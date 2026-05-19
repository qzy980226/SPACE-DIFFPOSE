"""
遮挡鲁棒性评估工具
功能：测试模型在不同遮挡程度下的性能

使用方法：
    python evaluate_occlusion_robustness.py \
        --config configs/speedplus_v2_diffpose.yml \
        --model_path exp/speedplus_v2_diffpose/best_model.pth \
        --occlusion_ratios 0.2 0.4 0.6 0.8 \
        --doc occlusion_test \
        --exp exp
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff
from models.ema import EMAHelper
from common.utils import *
from common.utils_diff import get_beta_schedule, compute_alpha
from common.loss import mpjpe, p_mpjpe
from common.generators import PoseGenerator_gmm_speedplus


class OcclusionRobustnessEvaluator:
    """遮挡鲁棒性评估器"""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置日志
        self.setup_logging()

        # 初始化扩散参数
        self.setup_diffusion()

        # 创建模型
        self.create_models()

        # 加载数据集
        self.prepare_data()

        # 创建源掩码（用于Transformer）
        self.src_mask = torch.tensor([[[True] * 11]])

    def setup_logging(self):
        """设置日志"""
        log_path = os.path.join(self.args.exp, self.args.doc)
        os.makedirs(log_path, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_path, 'occlusion_eval.log')),
                logging.StreamHandler()
            ]
        )

        self.log_path = log_path

    def setup_diffusion(self):
        """初始化扩散过程参数"""
        betas = get_beta_schedule(
            beta_schedule=self.config.diffusion.beta_schedule,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    def create_models(self):
        """创建扩散模型和姿态模型"""
        # SPEED+的骨架连接
        edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 0],    # 前面板
            [4, 5], [5, 6], [6, 7], [7, 4],    # 后面板
            [0, 7], [1, 4], [2, 5], [3, 6],    # 连接边
            [1, 9], [2, 10], [3, 8], [6, 8]    # 附件
        ], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)

        # 确保配置正确
        if not hasattr(self.config.model, 'use_visibility_embedding'):
            self.config.model.use_visibility_embedding = True

        # 创建扩散模型
        self.config.model.coords_dim = [5, 5]
        self.model_diff = GCNdiff(adj.cuda(), self.config).cuda()

        # 创建姿态模型
        self.config.model.coords_dim = [2, 3]
        self.model_pose = GCNpose(adj.cuda(), self.config).cuda()

        # 加载检查点
        logging.info(f"Loading model from {self.args.model_path}")
        states = torch.load(self.args.model_path, map_location=self.device)

        # 加载扩散模型参数
        self.model_diff.load_state_dict(states[0])

        # 如果有EMA，使用EMA参数
        if self.config.model.ema and len(states) > 4:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
            ema_helper.load_state_dict(states[4])
            ema_helper.ema(self.model_diff)
            logging.info("Using EMA parameters")

        logging.info("Model loaded successfully")

    def prepare_data(self):
        """准备验证数据集"""
        if self.config.data.dataset == "speedplus_v2":
            from common.speedplus_dataset_v2 import SpeedPlusV2Dataset
            from common.data_utils_v2 import create_2d_data_speedplus_v2

            # 加载数据集
            dataset = SpeedPlusV2Dataset(
                train_json_path=self.config.data.train_json_path,
                kpts_mat_path=self.config.data.kpts_mat_path,
                gmm_data_path=self.config.data.gmm_data_path
            )

            keypoints_gmm = create_2d_data_speedplus_v2(dataset)

            # 划分验证集（使用后20%）
            all_actions = list(dataset['spacecraft'].keys())
            split_idx = int(len(all_actions) * 0.8)
            test_actions = all_actions[split_idx:]

            self.keypoints_test = {
                'spacecraft': {k: keypoints_gmm['spacecraft'][k] for k in test_actions}
            }
            self.subjects_test = ['spacecraft']
            self.dataset = dataset

            logging.info(f"Validation set size: {len(test_actions)}")
        else:
            raise KeyError('Invalid dataset')

    def create_dataloader_with_occlusion(self, occlusion_ratio):
        """
        创建带有人工遮挡的数据加载器

        Args:
            occlusion_ratio: 遮挡比例（0.0-1.0）
        """
        from common.data_utils_v2 import fetch_speedplus_v2

        # 获取验证数据
        poses_valid, poses_valid_2d, camerapara_valid, visibility_valid = fetch_speedplus_v2(
            self.subjects_test, self.dataset, self.keypoints_test, stride=1
        )

        # 创建带遮挡的数据生成器
        test_generator = OccludedPoseGenerator(
            poses_valid, poses_valid_2d, camerapara_valid, visibility_valid,
            occlusion_ratio=occlusion_ratio,
            uncertainty_scale=50  # 与训练时相同
        )

        data_loader = data.DataLoader(
            test_generator,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )

        return data_loader

    def generalized_steps_with_visibility(self, x, src_mask, seq, model, b, visibility, **kwargs):
        """扩散去噪步骤（支持可见性）"""
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())

                xt = xs[-1].to(self.device)

                # 预测噪声
                et = model(xt, src_mask, t, 0, visibility)

                # DDIM更新
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))

                c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))

        return xs, x0_preds

    def evaluate_with_occlusion(self, occlusion_ratio):
        """
        在指定遮挡比例下评估模型

        Args:
            occlusion_ratio: 遮挡比例（0.0表示无遮挡，1.0表示全部遮挡）

        Returns:
            mpjpe_error: 平均关节位置误差（mm）
            pa_mpjpe_error: Procrustes对齐后误差（mm）
        """
        cudnn.benchmark = True

        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating with {occlusion_ratio*100:.0f}% keypoints occluded")
        logging.info(f"{'='*60}")

        # 创建数据加载器
        data_loader = self.create_dataloader_with_occlusion(occlusion_ratio)

        # 测试配置
        test_times = self.config.testing.test_times
        test_timesteps = self.config.testing.test_timesteps
        test_num_diffusion_timesteps = self.config.testing.test_num_diffusion_timesteps

        # 设置扩散步骤
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        # 切换到评估模式
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()

        # 指标累加器
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()

        # 遍历所有批次
        for i, batch_data in enumerate(data_loader):
            # 解包数据
            _, input_noise_scale, input_2d, targets_3d, camera_para, \
                visibility, visibility_mask = batch_data

            # 使用姿态模型预测初始3D
            inputs_xyz = self.model_pose(input_2d, self.src_mask, visibility)
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :]  # 中心化

            input_2d = input_2d.to(self.device)
            inputs_xyz = inputs_xyz.to(self.device)

            # 拼接UV+XYZ
            input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)

            # 多次采样
            input_uvxyz = input_uvxyz.repeat(test_times, 1, 1)
            input_noise_scale = input_noise_scale.repeat(test_times, 1, 1)
            visibility_rep = visibility.repeat(test_times, 1)

            input_uvxyz = input_uvxyz.to(self.device)
            input_noise_scale = input_noise_scale.to(self.device)
            visibility_rep = visibility_rep.to(self.device)

            # 设置扩散时间步
            t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device) * test_num_diffusion_timesteps

            # 准备扩散参数
            x = input_uvxyz.clone()

            # 执行扩散去噪
            output_uvxyz = self.generalized_steps_with_visibility(
                x, self.src_mask, seq, self.model_diff, self.betas,
                visibility=visibility_rep, eta=self.args.eta
            )
            output_uvxyz = output_uvxyz[0][-1]

            # 平均多次采样
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times, -1, 11, 5), 0)
            output_xyz = output_uvxyz[:, :, 2:]

            output_xyz = output_xyz.to(self.device)
            targets_3d = targets_3d.to(self.device)

            # 中心化
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]

            # 计算误差
            if visibility.sum() > 0:
                # MPJPE
                mpjpe_error = mpjpe(output_xyz, targets_3d).item() * 1000.0
                epoch_loss_3d_pos.update(mpjpe_error, targets_3d.size(0))

                # PA-MPJPE
                p_mpjpe_error = p_mpjpe(
                    output_xyz.cpu().numpy(),
                    targets_3d.cpu().numpy()
                ).item() * 1000.0
                epoch_loss_3d_pos_procrustes.update(p_mpjpe_error, targets_3d.size(0))

            # 打印进度
            if (i + 1) % 10 == 0:
                logging.info(
                    f'Batch [{i+1}/{len(data_loader)}] | '
                    f'MPJPE: {epoch_loss_3d_pos.avg:.2f}mm | '
                    f'PA-MPJPE: {epoch_loss_3d_pos_procrustes.avg:.2f}mm'
                )

        # 最终结果
        mpjpe_final = epoch_loss_3d_pos.avg
        pa_mpjpe_final = epoch_loss_3d_pos_procrustes.avg

        logging.info(f"\nFinal Results:")
        logging.info(f"  MPJPE: {mpjpe_final:.2f} mm")
        logging.info(f"  PA-MPJPE: {pa_mpjpe_final:.2f} mm")

        return mpjpe_final, pa_mpjpe_final

    def run_evaluation(self, occlusion_ratios):
        """
        运行完整的遮挡鲁棒性评估

        Args:
            occlusion_ratios: 遮挡比例列表，如[0.0, 0.2, 0.4, 0.6, 0.8]
        """
        results = []

        logging.info("\n" + "="*80)
        logging.info("OCCLUSION ROBUSTNESS EVALUATION")
        logging.info("="*80)

        for ratio in occlusion_ratios:
            mpjpe_val, pa_mpjpe_val = self.evaluate_with_occlusion(ratio)
            results.append({
                'occlusion_ratio': ratio,
                'mpjpe': mpjpe_val,
                'pa_mpjpe': pa_mpjpe_val
            })

        # 打印汇总表格
        self.print_summary(results)

        # 保存结果
        self.save_results(results)

        return results

    def print_summary(self, results):
        """打印评估结果汇总"""
        logging.info("\n" + "="*80)
        logging.info("EVALUATION SUMMARY")
        logging.info("="*80)
        logging.info(f"{'Occlusion Ratio':<20} {'MPJPE (mm)':<15} {'PA-MPJPE (mm)':<15} {'Degradation':<15}")
        logging.info("-"*80)

        baseline_mpjpe = results[0]['mpjpe'] if results else 0

        for res in results:
            ratio = res['occlusion_ratio']
            mpjpe = res['mpjpe']
            pa_mpjpe = res['pa_mpjpe']

            if baseline_mpjpe > 0:
                degradation = (mpjpe - baseline_mpjpe) / baseline_mpjpe * 100
                degradation_str = f"+{degradation:.1f}%" if degradation > 0 else f"{degradation:.1f}%"
            else:
                degradation_str = "N/A"

            logging.info(f"{ratio*100:>6.0f}%{'':<13} {mpjpe:<15.2f} {pa_mpjpe:<15.2f} {degradation_str:<15}")

        logging.info("="*80)

    def save_results(self, results):
        """保存结果到文件"""
        import json

        result_path = os.path.join(self.log_path, 'occlusion_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

        logging.info(f"\nResults saved to {result_path}")

        # 保存为CSV
        csv_path = os.path.join(self.log_path, 'occlusion_results.csv')
        with open(csv_path, 'w') as f:
            f.write("Occlusion_Ratio,MPJPE_mm,PA_MPJPE_mm\n")
            for res in results:
                f.write(f"{res['occlusion_ratio']},{res['mpjpe']:.2f},{res['pa_mpjpe']:.2f}\n")

        logging.info(f"Results saved to {csv_path}")


class OccludedPoseGenerator(PoseGenerator_gmm_speedplus):
    """
    带遮挡模拟的姿态数据生成器
    继承自原始生成器，添加人工遮挡功能
    """

    def __init__(self, poses_3d, poses_2d_gmm, camerapara, visibility,
                 occlusion_ratio=0.0, uncertainty_scale=50, **kwargs):
        """
        Args:
            occlusion_ratio: 遮挡比例（0.0-1.0）
            uncertainty_scale: 被遮挡关键点的不确定性放大倍数
        """
        super().__init__(
            poses_3d, poses_2d_gmm, camerapara, visibility,
            augment_uncertainty=False,  # 禁用训练时的增强
            **kwargs
        )

        self.occlusion_ratio = occlusion_ratio
        self.uncertainty_scale = uncertainty_scale
        self.num_joints = 11

    def __getitem__(self, index):
        """重写getitem，添加人工遮挡"""
        # 调用父类方法获取原始数据
        out_pose_uvxyz, out_pose_noise_scale, out_pose_2d, out_pose_3d, \
            out_camerapara, out_visibility, visibility_mask = super().__getitem__(index)

        # 如果遮挡比例为0，直接返回
        if self.occlusion_ratio <= 0:
            return (out_pose_uvxyz, out_pose_noise_scale, out_pose_2d,
                   out_pose_3d, out_camerapara, out_visibility, visibility_mask)

        # 转换为numpy以便操作
        out_visibility_np = out_visibility.numpy()

        # 找出当前可见的关键点
        visible_joints = np.where(out_visibility_np > 0)[0]

        if len(visible_joints) > 0:
            # 计算要遮挡的关键点数量
            num_to_occlude = int(len(visible_joints) * self.occlusion_ratio)
            num_to_occlude = max(1, min(num_to_occlude, len(visible_joints)))  # 至少遮挡1个

            # 随机选择要遮挡的关键点
            occluded_joints = np.random.choice(
                visible_joints,
                size=num_to_occlude,
                replace=False
            )

            # 转换回tensor进行修改
            out_pose_uvxyz_np = out_pose_uvxyz.numpy()
            out_pose_noise_scale_np = out_pose_noise_scale.numpy()
            visibility_mask_np = visibility_mask.numpy()

            for joint_idx in occluded_joints:
                # 1. 设置2D坐标为零
                out_pose_uvxyz_np[joint_idx, 0:2] = [0.0, 0.0]

                # 2. 增大2D不确定性
                out_pose_noise_scale_np[joint_idx, 0:2] = [10.0, 10.0]

                # 3. 更新可见性标记
                out_visibility_np[joint_idx] = 0.0

                # 4. 更新可见性掩码
                visibility_mask_np[joint_idx, :] = 0.0

            # 转换回tensor
            out_pose_uvxyz = torch.from_numpy(out_pose_uvxyz_np).float()
            out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale_np).float()
            out_visibility = torch.from_numpy(out_visibility_np).float()
            visibility_mask = torch.from_numpy(visibility_mask_np).float()

        return (out_pose_uvxyz, out_pose_noise_scale, out_pose_2d,
               out_pose_3d, out_camerapara, out_visibility, visibility_mask)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Occlusion Robustness Evaluation')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (e.g., best_model.pth)')
    parser.add_argument('--occlusion_ratios', type=float, nargs='+',
                       default=[0.0, 0.2, 0.4, 0.6, 0.8],
                       help='Occlusion ratios to test')
    parser.add_argument('--doc', type=str, default='occlusion_eval',
                       help='Document name for logging')
    parser.add_argument('--exp', type=str, default='./exp',
                       help='Experiment directory')
    parser.add_argument('--skip_type', type=str, default='uniform',
                       choices=['uniform', 'quad'],
                       help='Skip type for diffusion sampling')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Eta for DDIM')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    import yaml
    from argparse import Namespace

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 递归转换为Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    config = dict_to_namespace(config_dict)

    # 创建评估器
    evaluator = OcclusionRobustnessEvaluator(args, config)

    # 运行评估
    results = evaluator.run_evaluation(args.occlusion_ratios)

    logging.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
