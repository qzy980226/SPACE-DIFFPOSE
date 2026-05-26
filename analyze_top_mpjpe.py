"""
分析验证集中 MPJPE 最高的样本。

用法：
  python analyze_top_mpjpe.py \
      --config configs/speedplus_v2_diffpose.yml \
      --model_pose_path <gcnpose_best.pth 路径> \
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

from models.gcnpose import GCNpose, adj_mx_from_edges
from common.generators import PoseGenerator_gmm_speedplus


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def per_sample_mpjpe(predicted, target, visibility_mask):
    """
    计算每个样本的 MPJPE（毫米），只统计可见关节。

    Args:
        predicted:       (B, J, 3)
        target:          (B, J, 3)
        visibility_mask: (B, J)  取值 0/1

    Returns:
        (B,) 每个样本的 MPJPE，单位 mm
    """
    errors = torch.norm(predicted - target, dim=-1)          # (B, J)
    visible_count = visibility_mask.sum(dim=1).clamp(min=1)  # (B,)
    per_sample = (errors * visibility_mask).sum(dim=1) / visible_count  # (B,) 米
    return per_sample * 1000.0  # 转毫米


def load_config(config_path):
    with open(config_path, "r") as f:
        return dict2namespace(yaml.safe_load(f))


def build_gcnpose(config, device):
    edges = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 7], [1, 4], [2, 5], [3, 6],
        [1, 9], [2, 10],
        [3, 8], [6, 8]
    ], dtype=torch.long)
    adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)
    config.model.coords_dim = [2, 3]
    if not hasattr(config.model, "use_visibility_embedding"):
        config.model.use_visibility_embedding = True
    model = GCNpose(adj.to(device), config).to(device)
    model = torch.nn.DataParallel(model)
    return model


def run_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    src_mask = torch.tensor([[[True] * 11]]).to(device)

    # ── 加载数据集 ──────────────────────────────────────────────────────────────
    from common.speedplus_dataset_v2 import SpeedPlusV2Dataset
    from common.data_utils_v2 import create_2d_data_speedplus_v2, fetch_speedplus_v2

    dataset = SpeedPlusV2Dataset(
        train_json_path=config.data.train_json_path,
        kpts_mat_path=config.data.kpts_mat_path,
        gmm_data_path=config.data.gmm_data_path,
    )

    all_actions = list(dataset["spacecraft"].keys())
    split_idx = int(len(all_actions) * args.train_split)
    test_actions = all_actions[split_idx:]          # 与训练时保持一致
    logging.info(f"验证集样本数: {len(test_actions)}")

    keypoints_gmm = create_2d_data_speedplus_v2(dataset)
    keypoints_test = {"spacecraft": {k: keypoints_gmm["spacecraft"][k] for k in test_actions}}

    poses_valid, poses_valid_2d, camerapara_valid, visibility_valid = fetch_speedplus_v2(
        ["spacecraft"], dataset, keypoints_test, stride=1
    )

    test_generator = PoseGenerator_gmm_speedplus(
        poses_valid, poses_valid_2d, camerapara_valid, visibility_valid,
        augment_uncertainty=False,
        global_variance_scale=1.0,
    )

    loader = data.DataLoader(
        test_generator,
        batch_size=args.batch_size,
        shuffle=False,          # 必须 False，保证索引顺序
        num_workers=4,
        pin_memory=True,
    )

    # ── 加载 GCNpose ────────────────────────────────────────────────────────────
    model_pose = build_gcnpose(config, device)
    if args.model_pose_path and os.path.exists(args.model_pose_path):
        states = torch.load(args.model_pose_path, map_location=device)
        model_pose.load_state_dict(states[0])
        logging.info(f"已加载 GCNpose: {args.model_pose_path}")
    else:
        logging.warning("未提供 GCNpose 权重，使用随机初始化（结果无意义）")

    # ── 推理 ────────────────────────────────────────────────────────────────────
    torch.set_grad_enabled(False)
    model_pose.eval()
    cudnn.benchmark = True

    all_mpjpe = []      # List[float]，每个元素对应一个样本

    for batch_data in loader:
        _, _, input_2d, target_3d, _, visibility, _ = batch_data

        input_2d   = input_2d.to(device)
        target_3d  = target_3d.to(device)
        visibility = visibility.to(device)

        pred_3d = model_pose(input_2d, src_mask, visibility)

        # 中心化（与训练一致）
        pred_3d   -= pred_3d[:, :1, :]
        target_3d -= target_3d[:, :1, :]

        batch_mpjpe = per_sample_mpjpe(pred_3d, target_3d, visibility)  # (B,)
        all_mpjpe.extend(batch_mpjpe.cpu().tolist())

    # ── 统计 Top-K ──────────────────────────────────────────────────────────────
    assert len(all_mpjpe) == len(test_actions), (
        f"样本数量不匹配: {len(all_mpjpe)} vs {len(test_actions)}"
    )

    paired = list(zip(test_actions, all_mpjpe))
    paired.sort(key=lambda x: x[1], reverse=True)

    top_k = args.top_k
    print(f"\n{'='*60}")
    print(f"  验证集 Top-{top_k} 最高 MPJPE 样本")
    print(f"{'='*60}")
    print(f"{'排名':<6}{'MPJPE (mm)':<14}{'样本名称'}")
    print(f"{'-'*60}")
    for rank, (name, err) in enumerate(paired[:top_k], 1):
        print(f"{rank:<6}{err:<14.2f}{name}")

    print(f"\n全验证集统计:")
    arr = np.array(all_mpjpe)
    print(f"  均值:   {arr.mean():.2f} mm")
    print(f"  中位数: {np.median(arr):.2f} mm")
    print(f"  标准差: {arr.std():.2f} mm")
    print(f"  最大值: {arr.max():.2f} mm")
    print(f"  最小值: {arr.min():.2f} mm")

    # ── 保存结果 ────────────────────────────────────────────────────────────────
    if args.output:
        results = {
            "top_k": [{"rank": i+1, "name": n, "mpjpe_mm": round(e, 4)}
                      for i, (n, e) in enumerate(paired[:top_k])],
            "all_samples": [{"name": n, "mpjpe_mm": round(e, 4)}
                            for n, e in paired],
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
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--model_pose_path", default=None, help="GCNpose 权重路径")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="训练集比例，与训练时保持一致（默认 0.8）")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=10, help="显示 MPJPE 最高的 K 个样本")
    parser.add_argument("--output", default=None, help="将结果保存为 JSON 文件（可选）")
    args = parser.parse_args()
    run_analysis(args)
