import os
import logging
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.generators import PoseGenerator_gmm_speedplus
# 修改：导入支持可见性的函数和损失函数
from common.data_utils import fetch_speedplus_with_visibility, create_dynamic_attention_masks, compute_visibility_statistics
from common.loss import mpjpe, p_mpjpe, mpjpe_masked, p_mpjpe_masked, compute_combined_loss, compute_loss_statistics

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        
        # 修改：移除固定的src_mask，改为动态mask
        # 原代码：self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True]]]).cuda()
        # 新代码：不再使用固定mask，将在forward时动态创建
        
        # Generate Diffusion sequence parameters（保持不变）
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def prepare_data(self):
        """
        修改：增加SPEED+数据集支持和可见性处理
        """
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # 加载数据集
        if config.data.dataset == "speedplus":
            from common.speedplus_dataset import SpeedPlusDataset
            
            # 加载SPEED+数据集
            dataset = SpeedPlusDataset(
                json_path=config.data.json_path,
                keypoints_path=config.data.keypoints_path
            )
            
            self.subjects_train = ['spacecraft']
            self.subjects_test = ['spacecraft']
            
            # 处理3D数据
            from common.data_utils import read_3d_data_speedplus, create_2d_data_speedplus
            self.dataset = read_3d_data_speedplus(dataset)
            
            # 创建2D投影数据
            self.keypoints_train = create_2d_data_speedplus(dataset)
            self.keypoints_test = self.keypoints_train  # 注意：这里需要根据实际情况分割数据集
            
            # 修改：获取并打印可见性统计信息
            visibility_stats = dataset.get_visibility_stats()
            print(f"==> 数据集可见性统计:")
            print(f"    总体可见性率: {visibility_stats['overall_visibility_rate']:.3f}")
            print(f"    平均每张图可见关键点: {visibility_stats['overall_visibility_rate'] * 11:.1f}/11")
            
            # SPEED+没有动作分类
            self.action_filter = None
            
        elif config.data.dataset == "human36m":
            # 原有的human3.6m处理逻辑（保持不变）
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            # ... 原有逻辑
        else:
            raise KeyError('Invalid dataset')

    def create_dynamic_mask(self, visibility_masks):
        """
        新增：根据批次的可见性创建动态注意力mask
        
        修改原因：需要为每个batch动态生成mask
        作用：将可见性信息转换为模型可用的注意力mask格式
        
        参数:
        visibility_masks: [batch_size, num_joints] 可见性mask
        
        返回:
        torch.Tensor: [batch_size, 1, num_joints] 注意力mask
        """
        batch_size, num_joints = visibility_masks.shape
        
        # 转换为注意力mask格式：[batch_size, 1, num_joints]
        attention_masks = visibility_masks.unsqueeze(1).float()
        
        return attention_masks.to(self.device)

    def prepare_dataloader(self):
        """
        修改：准备数据加载器，支持可见性mask
        """
        
        # 获取训练数据（包含可见性信息）
        if self.config.data.dataset == "speedplus":
            poses_train_2d, poses_train_3d, visibility_masks_train, actions_train, subjects_train = \
                fetch_speedplus_with_visibility(self.subjects_train, self.dataset, self.keypoints_train, 
                                               stride=self.config.data.get('stride', 1))
            
            poses_test_2d, poses_test_3d, visibility_masks_test, actions_test, subjects_test = \
                fetch_speedplus_with_visibility(self.subjects_test, self.dataset, self.keypoints_test, 
                                               stride=self.config.data.get('stride', 1))
        else:
            # 原有的human3.6m处理逻辑
            from common.data_utils import fetch_me
            poses_train_2d, poses_train_3d, actions_train, subjects_train = fetch_me(...)
            # 为human3.6m创建默认的全可见mask
            visibility_masks_train = [np.ones(17, dtype=bool) for _ in range(len(poses_train_2d))]
            visibility_masks_test = [np.ones(17, dtype=bool) for _ in range(len(poses_test_2d))]

        print(f'==> 训练数据: {len(poses_train_2d)} 个样本')
        print(f'==> 测试数据: {len(poses_test_2d)} 个样本')
        
        # 打印可见性统计
        if self.config.data.dataset == "speedplus":
            train_vis_stats = compute_visibility_statistics(visibility_masks_train)
            test_vis_stats = compute_visibility_statistics(visibility_masks_test)
            
            print(f'==> 训练集可见性统计:')
            print(f'    整体可见性率: {train_vis_stats["overall_visibility_rate"]:.3f}')
            print(f'    每样本最少可见关键点: {train_vis_stats["min_visible_per_sample"]}')
            print(f'    每样本最多可见关键点: {train_vis_stats["max_visible_per_sample"]}')

        # 归一化2D坐标
        from common.camera import normalize_screen_coordinates
        for i in range(len(poses_train_2d)):
            if self.config.data.dataset == "speedplus":
                cam = self.dataset.cameras()['spacecraft'][0]
                poses_train_2d[i] = normalize_screen_coordinates(poses_train_2d[i], cam['res_w'], cam['res_h'])
        
        for i in range(len(poses_test_2d)):
            if self.config.data.dataset == "speedplus":
                cam = self.dataset.cameras()['spacecraft'][0]
                poses_test_2d[i] = normalize_screen_coordinates(poses_test_2d[i], cam['res_w'], cam['res_h'])

        # 创建训练和测试的数据集
        class PoseDataset(torch.utils.data.Dataset):
            def __init__(self, poses_2d, poses_3d, visibility_masks, augment_visibility=False):
                self.poses_2d = poses_2d
                self.poses_3d = poses_3d
                self.visibility_masks = visibility_masks
                self.augment_visibility = augment_visibility
                
            def __len__(self):
                return len(self.poses_2d)
                
            def __getitem__(self, idx):
                pose_2d = torch.from_numpy(self.poses_2d[idx]).float()
                pose_3d = torch.from_numpy(self.poses_3d[idx]).float()
                visibility_mask = torch.from_numpy(self.visibility_masks[idx]).bool()
                
                # 数据增强：随机遮挡
                if self.augment_visibility and torch.rand(1) < 0.3:
                    visible_indices = torch.where(visibility_mask)[0]
                    if len(visible_indices) > 3:  # 确保至少保留3个可见关键点
                        num_to_occlude = min(2, len(visible_indices) - 3)
                        occlude_indices = visible_indices[torch.randperm(len(visible_indices))[:num_to_occlude]]
                        visibility_mask[occlude_indices] = False
                
                return pose_2d, pose_3d, visibility_mask

        # 创建数据加载器
        train_dataset = PoseDataset(poses_train_2d, poses_train_3d, visibility_masks_train, 
                                   augment_visibility=self.config.data.get('augment_visibility', False))
        test_dataset = PoseDataset(poses_test_2d, poses_test_3d, visibility_masks_test, 
                                  augment_visibility=False)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.get('num_workers', 4),
            drop_last=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            drop_last=False
        )

    def train_step(self, batch_data):
        """
        修改：单步训练，支持动态mask
        """
        poses_2d, poses_3d, visibility_masks = batch_data
        poses_2d = poses_2d.to(self.device)
        poses_3d = poses_3d.to(self.device)
        visibility_masks = visibility_masks.to(self.device)
        
        batch_size = poses_2d.shape[0]
        
        # 修改：创建动态注意力mask
        dynamic_masks = self.create_dynamic_mask(visibility_masks)
        
        # 随机采样时间步（保持不变）
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # 前向扩散过程（保持不变）
        noise = torch.randn_like(poses_3d)
        noisy_poses = self.q_sample(poses_3d, t, noise)
        
        # 修改：使用动态mask的模型前向传播
        predicted_noise = self.model(noisy_poses, dynamic_masks, t, poses_2d)
        
        # 修改：计算支持可见性mask的损失
        loss_dict = compute_combined_loss(
            predicted_noise, noise, visibility_masks,
            mpjpe_weight=self.config.training.get('mpjpe_weight', 1.0),
            p_mpjpe_weight=self.config.training.get('p_mpjpe_weight', 0.5),
            use_strict_masking=self.config.training.get('use_strict_masking', True),
            weight_invisible=self.config.training.get('weight_invisible', 0.1)
        )
        
        return loss_dict['total'], loss_dict

    def validate(self):
        """
        修改：验证函数，支持可见性统计
        """
        self.model.eval()
        total_loss = 0
        total_mpjpe = 0
        total_p_mpjpe = 0
        total_mpjpe_visible = 0
        total_p_mpjpe_visible = 0
        num_samples = 0
        
        visibility_stats_accumulator = {
            'total_visible': 0,
            'total_keypoints': 0,
            'samples_count': 0
        }

        with torch.no_grad():
            for batch_data in tqdm.tqdm(self.test_loader, desc="验证"):
                poses_2d, poses_3d, visibility_masks = batch_data
                poses_2d = poses_2d.to(self.device)
                poses_3d = poses_3d.to(self.device)
                visibility_masks = visibility_masks.to(self.device)
                
                batch_size = poses_2d.shape[0]
                
                # 创建动态mask
                dynamic_masks = self.create_dynamic_mask(visibility_masks)
                
                # 使用扩散模型进行采样（简化版本）
                predicted_poses = self.sample_poses(poses_2d, dynamic_masks, num_samples=1)
                
                # 计算各种损失指标
                stats = compute_loss_statistics(predicted_poses, poses_3d, visibility_masks)
                
                # 累积统计信息
                total_mpjpe += stats['mpjpe_all'].item() * batch_size
                total_p_mpjpe += stats['p_mpjpe_all'].item() * batch_size
                total_mpjpe_visible += stats['mpjpe_visible'].item() * batch_size
                total_p_mpjpe_visible += stats['p_mpjpe_visible'].item() * batch_size
                num_samples += batch_size
                
                # 可见性统计
                visibility_stats_accumulator['total_visible'] += torch.sum(visibility_masks.float()).item()
                visibility_stats_accumulator['total_keypoints'] += visibility_masks.numel()
                visibility_stats_accumulator['samples_count'] += batch_size

        # 计算平均指标
        avg_mpjpe = total_mpjpe / num_samples
        avg_p_mpjpe = total_p_mpjpe / num_samples
        avg_mpjpe_visible = total_mpjpe_visible / num_samples
        avg_p_mpjpe_visible = total_p_mpjpe_visible / num_samples
        
        visibility_rate = visibility_stats_accumulator['total_visible'] / visibility_stats_accumulator['total_keypoints']
        
        validation_results = {
            'mpjpe_all': avg_mpjpe,
            'p_mpjpe_all': avg_p_mpjpe,
            'mpjpe_visible': avg_mpjpe_visible,
            'p_mpjpe_visible': avg_p_mpjpe_visible,
            'visibility_rate': visibility_rate,
            'num_samples': num_samples
        }
        
        print(f"验证结果:")
        print(f"  MPJPE (全部): {avg_mpjpe:.2f}mm")
        print(f"  P-MPJPE (全部): {avg_p_mpjpe:.2f}mm") 
        print(f"  MPJPE (可见): {avg_mpjpe_visible:.2f}mm")
        print(f"  P-MPJPE (可见): {avg_p_mpjpe_visible:.2f}mm")
        print(f"  可见性率: {visibility_rate:.3f}")

        self.model.train()
        return validation_results

    def sample_poses(self, poses_2d, dynamic_masks, num_samples=1):
        """
        修改：使用扩散模型采样3D姿态，支持动态mask
        """
        batch_size = poses_2d.shape[0]
        
        # 从噪声开始
        sample_poses = torch.randn(batch_size, 11, 3, device=self.device)
        
        # 简化的采样过程（实际需要实现完整的DDPM/DDIM采样）
        for t in reversed(range(0, min(50, self.num_timesteps))):
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            # 模型预测（使用动态mask）
            predicted_noise = self.model(sample_poses, dynamic_masks, t_tensor, poses_2d)
            
            # 简化的去噪步骤
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
            
            # 简化的更新规则
            sample_poses = (sample_poses - predicted_noise * (1 - alpha_t_prev) / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t_prev / alpha_t)
            
            if t > 0:
                noise = torch.randn_like(sample_poses)
                sample_poses += torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise
        
        return sample_poses

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程（保持不变）"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # create diffusion model SPEED+版本（修改）
    def create_diffusion_model(self, model_path=None):
        """
        修改：创建支持动态mask的扩散模型
        """
        args, config = self.args, self.config
        
        if config.data.dataset == "speedplus":
            # SPEED+的边定义
            edges = torch.tensor([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 7], [1, 4], [2, 5], [3, 6],
                [1, 9], [2, 10],
                [3, 8], [6, 8]
            ], dtype=torch.long)
            adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)
        else:
            # Human3.6M的邻接矩阵
            adj = adj_mx_from_skeleton(self.skeleton)
            
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
        
        # 设置模型引用
        self.model = self.model_diff

    def create_pose_model(self, model_path=None):
        """
        修改：创建支持动态mask的姿态模型
        """
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2, 3]
        
        if config.data.dataset == "speedplus":
            edges = torch.tensor([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 7], [1, 4], [2, 5], [3, 6],
                [1, 9], [2, 10],
                [3, 8], [6, 8]
            ], dtype=torch.long)
            adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)
        else:
            adj = adj_mx_from_skeleton(self.skeleton)
            
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0])
        else:
            logging.info('initialize model randomly')

    def train(self):
        """
        修改：主训练循环，集成可见性功能
        """
        cudnn.benchmark = True
        args, config = self.args, self.config

        # 准备数据
        self.prepare_data()
        self.prepare_dataloader()
        
        # 创建模型
        self.create_diffusion_model()
        
        # 预计算扩散参数
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # 初始化优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # 训练循环
        for epoch in range(config.training.epochs):
            epoch_loss = 0
            num_batches = 0
            
            self.model.train()
            for batch_data in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                
                loss, loss_dict = self.train_step(batch_data)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}: Average Loss = {avg_epoch_loss:.4f}")
            
            # 定期验证
            if (epoch + 1) % config.training.get('val_interval', 10) == 0:
                validation_results = self.validate()
                
            scheduler.step()

        print("训练完成！")