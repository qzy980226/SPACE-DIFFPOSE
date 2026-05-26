"""
分析验证集中 MPJPE 最高的样本（完整 DiffPose 推理流程）。

用法：
  python analyze_top_mpjpe.py \
      --config configs/speedplus_v2_diffpose.yml \
      --model_pose_path <gcnpose_best.pth 路径> \
      --model_diff_path <diffpose_best.pth 路径> \
      --top_k 10 \
      --output worst_samples.json
"""

import argparse
import logging
import yaml
import os
import json
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

from models.gcnpose import GCNpose, adj_mx_from_edges as adj_mx_pose
from models.gcndiff import GCNdiff, adj_mx_from_edges as adj_mx_diff
from common.generators import PoseGenerator_gmm_speedplus
from common.utils_diff import get_beta_schedule, compute_alpha


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EDGES = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 7], [1, 4], [2, 5], [3, 6],
    [1, 9], [2, 10],
    [3, 8], [6, 8]
], dtype=torch.long)


def dict2namespace(config):
    ns = argparse.Namespace()
    for key, value in config.items():
        setattr(ns, key, dict2namespace(value) if isinstance(value, dict) else value)
    return ns


def load_config(config_path):
    with open(config_path, "r") as f:
        return dict2namespace(yaml.safe_load(f))


def per_sample_mpjpe(predicted, target, visibility_mask):
    """
    Args:
        predicted:       (B, J, 3)
        target:          (B, J, 3)
        visibility_mask: (B, J)

    Returns:
        (B,) 每样本 MPJPE，单位 mm
    """
    errors = torch.norm(predicted - target, dim=-1)          # (B, J)
    visible_count = visibility_mask.sum(dim=1).clamp(min=1)  # (B,)
    return (errors * visibility_mask).sum(dim=1) / visible_count * 1000.0


def build_model_pose(config, device):
    adj = adj_mx_pose(num_pts=11, edges=EDGES, sparse=False)
    config.model.coords_dim = [2, 3]
    if not hasattr(config.model, "use_visibility_embedding"):
        config.model.use_visibility_embedding = True
    model = GCNpose(adj.to(device), config).to(device)
    return torch.nn.DataParallel(model)


def build_model_diff(config, device):
    adj = adj_mx_diff(num_pts=11, edges=EDGES, sparse=False)
    config.model.coords_dim = [5, 5]  # build_model_pose 会把 config 改成 [2,3]，这里还原
    if not hasattr(config.model, "use_visibility_embedding"):
        config.model.use_visibility_embedding = True
    model = GCNdiff(adj.to(device), config).to(device)
    return torch.nn.DataParallel(model)


def generalized_steps_with_visibility(x, src_mask, seq, model, betas, visibility, eta=0.0):
    """DDIM 去噪采样（与 diffpose_frame.py 保持一致）"""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t      = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at      = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, next_t.long())
            xt = xs[-1]

            et = model(xt, src_mask, t.float(), 0, visibility)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs


def run_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    src_mask = torch.tensor([[[True] * 11]]).to(device)

    # ── 扩散时间步序列（与 test_hyber 一致）──────────────────────────────────────
    test_times      = config.testing.test_times
    test_timesteps  = config.testing.test_timesteps
    test_num_diff_t = config.testing.test_num_diffusion_timesteps

    if args.skip_type == "uniform":
        skip = test_num_diff_t // test_timesteps
        seq  = range(0, test_num_diff_t, skip)
    else:
        seq = (np.linspace(0, np.sqrt(test_num_diff_t * 0.8), test_timesteps) ** 2)
        seq = [int(s) for s in list(seq)]

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    # ── 加载数据集 ──────────────────────────────────────────────────────────────
    from common.speedplus_dataset_v2 import SpeedPlusV2Dataset
    from common.data_utils_v2 import create_2d_data_speedplus_v2, fetch_speedplus_v2

    dataset = SpeedPlusV2Dataset(
        train_json_path=config.data.train_json_path,
        kpts_mat_path=config.data.kpts_mat_path,
        gmm_data_path=config.data.gmm_data_path,
    )

    all_actions = list(dataset["spacecraft"].keys())
    split_idx   = int(len(all_actions) * args.train_split)
    test_actions = all_actions[split_idx:]
    logging.info(f"验证集样本数: {len(test_actions)}")

    keypoints_gmm  = create_2d_data_speedplus_v2(dataset)
    keypoints_test = {"spacecraft": {k: keypoints_gmm["spacecraft"][k] for k in test_actions}}

    poses_valid, poses_valid_2d, camerapara_valid, visibility_valid = fetch_speedplus_v2(
        ["spacecraft"], dataset, keypoints_test, stride=1
    )

    test_generator = PoseGenerator_gmm_speedplus(
        poses_valid, poses_valid_2d, camerapara_valid, visibility_valid,
        augment_uncertainty=False,
    )

    loader = data.DataLoader(
        test_generator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── 加载模型权重 ────────────────────────────────────────────────────────────
    model_pose = build_model_pose(config, device)
    assert args.model_pose_path and os.path.exists(args.model_pose_path), \
        f"GCNpose 权重不存在: {args.model_pose_path}"
    result = model_pose.load_state_dict(torch.load(args.model_pose_path, map_location=device)[0], strict=False)
    if result.missing_keys:
        logging.warning(f"GCNpose 缺失键（随机初始化）: {result.missing_keys}")
    if result.unexpected_keys:
        logging.warning(f"GCNpose 多余键（已忽略）: {result.unexpected_keys}")
    logging.info(f"已加载 GCNpose:  {args.model_pose_path}")

    model_diff = build_model_diff(config, device)
    assert args.model_diff_path and os.path.exists(args.model_diff_path), \
        f"GCNdiff 权重不存在: {args.model_diff_path}"
    result = model_diff.load_state_dict(torch.load(args.model_diff_path, map_location=device)[0], strict=False)
    if result.missing_keys:
        logging.warning(f"GCNdiff 缺失键（随机初始化）: {result.missing_keys}")
    if result.unexpected_keys:
        logging.warning(f"GCNdiff 多余键（已忽略）: {result.unexpected_keys}")
    logging.info(f"已加载 GCNdiff:  {args.model_diff_path}")

    # ── 推理（完整 DiffPose 流程，与 test_hyber 保持一致）──────────────────────
    torch.set_grad_enabled(False)
    model_pose.eval()
    model_diff.eval()
    cudnn.benchmark = True

    all_mpjpe = []

    for batch_data in loader:
        _, input_noise_scale, input_2d, targets_3d, _, visibility, _ = batch_data

        # Step 1: GCNpose 生成初始 3D 姿态（input_2d 尚在 CPU，与原始代码一致）
        inputs_xyz = model_pose(input_2d, src_mask, visibility)
        inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]

        input_2d   = input_2d.to(device)
        inputs_xyz = inputs_xyz.to(device)

        # Step 2: 拼接 uvxyz 作为扩散模型输入
        input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)  # (B, 11, 5)

        # Step 3: 重复 test_times 次用于分布估计
        input_uvxyz     = input_uvxyz.repeat(test_times, 1, 1)
        input_noise_scale = input_noise_scale.repeat(test_times, 1, 1).to(device)
        visibility_rep  = visibility.repeat(test_times, 1).to(device)
        input_uvxyz     = input_uvxyz.to(device)

        # Step 4: DDIM 扩散去噪
        x = input_uvxyz.clone()
        e = torch.randn_like(input_uvxyz) * input_noise_scale

        xs = generalized_steps_with_visibility(
            x, src_mask, seq, model_diff, betas,
            visibility=visibility_rep, eta=args.eta
        )
        output_uvxyz = xs[-1]  # 最后一步结果

        # Step 5: 平均多次采样，提取 xyz
        output_uvxyz = torch.mean(output_uvxyz.reshape(test_times, -1, 11, 5), dim=0)
        output_xyz   = output_uvxyz[:, :, 2:]  # (B, 11, 3)

        targets_3d = targets_3d.to(device)

        # Step 6: 中心化（与 test_hyber 一致）
        output_xyz[:, :, :] -= output_xyz[:, :1, :]
        targets_3d[:, :, :] -= targets_3d[:, :1, :]

        # Step 7: per-sample MPJPE
        visibility_dev = visibility.to(device)
        batch_mpjpe = per_sample_mpjpe(output_xyz, targets_3d, visibility_dev)
        all_mpjpe.extend(batch_mpjpe.cpu().tolist())

    # ── 统计 Top-K ──────────────────────────────────────────────────────────────
    assert len(all_mpjpe) == len(test_actions), (
        f"样本数量不匹配: {len(all_mpjpe)} vs {len(test_actions)}"
    )

    paired = sorted(zip(test_actions, all_mpjpe), key=lambda x: x[1], reverse=True)
    arr    = np.array(all_mpjpe)

    print(f"\n{'='*60}")
    print(f"  验证集 Top-{args.top_k} 最高 MPJPE 样本")
    print(f"{'='*60}")
    print(f"{'排名':<6}{'MPJPE (mm)':<14}{'样本名称'}")
    print(f"{'-'*60}")
    for rank, (name, err) in enumerate(paired[:args.top_k], 1):
        print(f"{rank:<6}{err:<14.2f}{name}")

    print(f"\n全验证集统计:")
    print(f"  均值:   {arr.mean():.2f} mm")
    print(f"  中位数: {np.median(arr):.2f} mm")
    print(f"  标准差: {arr.std():.2f} mm")
    print(f"  最大值: {arr.max():.2f} mm")
    print(f"  最小值: {arr.min():.2f} mm")

    if args.output:
        results = {
            "top_k": [{"rank": i+1, "name": n, "mpjpe_mm": round(e, 4)}
                      for i, (n, e) in enumerate(paired[:args.top_k])],
            "all_samples": [{"name": n, "mpjpe_mm": round(e, 4)} for n, e in paired],
            "stats": {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std()),
                "max":    float(arr.max()),
                "min":    float(arr.min()),
            },
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",           required=True,  help="YAML 配置文件路径")
    parser.add_argument("--model_pose_path",  required=True,  help="GCNpose 权重路径")
    parser.add_argument("--model_diff_path",  required=True,  help="GCNdiff 权重路径")
    parser.add_argument("--train_split",  type=float, default=0.8,
                        help="训练集比例，与训练时保持一致（默认 0.8）")
    parser.add_argument("--skip_type",    default="uniform", choices=["uniform", "quad"])
    parser.add_argument("--eta",          type=float, default=0.0)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--top_k",        type=int,   default=10)
    parser.add_argument("--output",       default=None, help="保存结果为 JSON（可选）")
    args = parser.parse_args()
    run_analysis(args)
