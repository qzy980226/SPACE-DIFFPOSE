from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator_gmm_speedplus(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, camerapara, visibility,
                 augment_uncertainty=True, augment_prob=0.5,
                 uncertainty_scale=50, num_uncertain_joints=2,
                 # 混合增强策略参数
                 use_mixed_augmentation=False,
                 base_aug_config=None,
                 advanced_aug_config=None,
                 current_epoch=0,
                 # 全局方差缩放参数
                 global_variance_scale=1.0):
        """
        SPEED+ V2数据生成器，支持混合增强策略

        Args:
            augment_uncertainty: 是否启用数据增强
            augment_prob: 增强概率（向后兼容）
            uncertainty_scale: 不确定性放大倍数（向后兼容）
            num_uncertain_joints: 遮挡关键点数量（向后兼容）
            use_mixed_augmentation: 是否使用混合增强策略
            base_aug_config: 基础增强配置
            advanced_aug_config: 高级增强配置
            current_epoch: 当前训练epoch
            global_variance_scale: 全局方差缩放倍数（应用于所有关键点）
        """
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm)
        self._camerapara = np.concatenate(camerapara)
        self._visibility = np.concatenate(visibility)

        # 判断GMM组件数
        if len(self._poses_2d_gmm.shape) == 4:  # (N, 11, n_kernels, 5)
            self._kernel_n = self._poses_2d_gmm.shape[2]
        else:
            self._kernel_n = 1

        # 基础增强参数（向后兼容）
        self.augment_uncertainty = augment_uncertainty
        self.augment_prob = augment_prob
        self.uncertainty_scale = uncertainty_scale
        self.num_uncertain_joints = num_uncertain_joints

        # 混合增强策略参数
        self.use_mixed_augmentation = use_mixed_augmentation
        self.base_aug_config = base_aug_config or {}
        self.advanced_aug_config = advanced_aug_config or {}
        self.current_epoch = current_epoch

        # 全局方差缩放参数
        self.global_variance_scale = global_variance_scale

        # 中心化3D坐标
        self._poses_3d[:,:,:] = self._poses_3d[:,:,:] - self._poses_3d[:,:1,:]

        print(f'Generating {len(self._poses_3d)} poses for SPEED+ V2...')
        if self.global_variance_scale != 1.0:
            print(f'  Global variance scaling: {self.global_variance_scale}x')
        if self.use_mixed_augmentation:
            print(f'  Using mixed augmentation strategy (Epoch {current_epoch})')

    def _apply_mixed_augmentation(self, out_pose_2d_gmm, out_visibility):
        """
        应用混合增强策略

        Args:
            out_pose_2d_gmm: GMM参数 (11, n_kernels, 5)
            out_visibility: 可见性标记 (11,)

        Returns:
            增强后的GMM参数
        """
        visible_joints = np.where(out_visibility > 0)[0]

        if len(visible_joints) == 0:
            return out_pose_2d_gmm

        # 基础增强（训练全程）
        base_cfg = self.base_aug_config
        if base_cfg.get('enabled', True) and np.random.random() < base_cfg.get('augment_prob', 0.5):
            min_occluded = base_cfg.get('min_occluded_joints', 1)
            max_occluded = base_cfg.get('max_occluded_joints', 3)

            # 随机选择遮挡数量
            num_to_occlude = np.random.randint(min_occluded, max_occluded + 1)
            num_to_occlude = min(num_to_occlude, len(visible_joints))

            if num_to_occlude > 0:
                uncertain_joints = np.random.choice(
                    visible_joints, size=num_to_occlude, replace=False
                )
                for joint_idx in uncertain_joints:
                    scale = base_cfg.get('uncertainty_scale', 50)
                    out_pose_2d_gmm[joint_idx, :, 3:5] *= scale

        # 高级增强（从指定epoch开始）
        adv_cfg = self.advanced_aug_config
        start_epoch = adv_cfg.get('start_epoch', 30)

        if (adv_cfg.get('enabled', True) and
            self.current_epoch >= start_epoch and
            np.random.random() < adv_cfg.get('augment_prob', 0.3)):

            # 使用遮挡比例
            min_ratio = adv_cfg.get('min_occlusion_ratio', 0.4)
            max_ratio = adv_cfg.get('max_occlusion_ratio', 0.8)
            min_visible = adv_cfg.get('min_visible_joints', 2)

            # 随机选择遮挡比例
            occlusion_ratio = np.random.uniform(min_ratio, max_ratio)

            # 计算遮挡数量
            num_to_occlude = int(len(visible_joints) * occlusion_ratio)
            # 确保至少保留min_visible个可见点
            num_to_occlude = min(num_to_occlude, len(visible_joints) - min_visible)

            if num_to_occlude > 0:
                uncertain_joints = np.random.choice(
                    visible_joints, size=num_to_occlude, replace=False
                )
                for joint_idx in uncertain_joints:
                    scale = adv_cfg.get('uncertainty_scale', 100)
                    out_pose_2d_gmm[joint_idx, :, 3:5] *= scale

        return out_pose_2d_gmm

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_camerapara = self._camerapara[index]
        out_visibility = self._visibility[index].copy()  # (11,)

        # 处理2D GMM数据
        if len(self._poses_2d_gmm.shape) > 3:
            out_pose_2d_gmm = self._poses_2d_gmm[index].copy()

            # 选择增强策略
            if self.use_mixed_augmentation:
                # 使用混合增强策略
                out_pose_2d_gmm = self._apply_mixed_augmentation(out_pose_2d_gmm, out_visibility)
            elif self.augment_uncertainty and np.random.random() < self.augment_prob:
                # 向后兼容：使用旧的简单增强
                visible_joints = np.where(out_visibility > 0)[0]

                if len(visible_joints) >= self.num_uncertain_joints:
                    uncertain_joints = np.random.choice(
                        visible_joints,
                        size=min(self.num_uncertain_joints, len(visible_joints)),
                        replace=False
                    )

                    for joint_idx in uncertain_joints:
                        out_pose_2d_gmm[joint_idx, :, 3:5] *= self.uncertainty_scale
            
            # 从GMM中采样
            out_pose_2d_kernel = np.zeros([out_pose_2d_gmm.shape[0], 5])
            
            for i in range(out_pose_2d_gmm.shape[0]):
                if out_visibility[i] > 0:
                    if self._kernel_n == 1:
                        # 单GMM组件：直接使用
                        out_pose_2d_kernel[i] = out_pose_2d_gmm[i, 0]
                    else:
                        # 多GMM组件：根据gmm_weights采样
                        probs = out_pose_2d_gmm[i, :, 0]
                        probs = probs / probs.sum()
                        kernel_idx = np.random.choice(self._kernel_n, 1, p=probs).item()
                        out_pose_2d_kernel[i] = out_pose_2d_gmm[i, kernel_idx]
                else:
                    # 不可见关键点：零掩码
                    out_pose_2d_kernel[i, 0] = 0.1  # 低概率
                    out_pose_2d_kernel[i, 1:3] = [0, 0]
                    out_pose_2d_kernel[i, 3:5] = [10.0, 10.0]
            
            kernel_mean = out_pose_2d_kernel[:, 1:3]
            kernel_variance = out_pose_2d_kernel[:, 3:5]

        else:
            # 简单2D坐标格式
            kernel_mean = self._poses_2d_gmm[index].copy()
            kernel_variance = np.ones_like(kernel_mean) * 0.1

            kernel_mean[out_visibility <= 0] = [0, 0]
            kernel_variance[out_visibility <= 0] = [10.0, 10.0]

        # 应用全局方差缩放（仅对可见关键点）
        if self.global_variance_scale != 1.0:
            for i in range(len(out_visibility)):
                if out_visibility[i] > 0:
                    kernel_variance[i] *= self.global_variance_scale

        # 生成uvxyz和噪声尺度
        out_pose_uvxyz = np.concatenate((kernel_mean, out_pose_3d), axis=1)
        out_pose_noise_scale = np.concatenate(
            (kernel_variance, np.ones(out_pose_3d.shape)), axis=1
        )
        
        # 创建可见性掩码（使用上一个版本的方式，更简洁）
        visibility_mask = np.concatenate([
            np.tile(out_visibility[:, np.newaxis], (1, 2)),  # UV
            np.tile(out_visibility[:, np.newaxis], (1, 3))   # XYZ
        ], axis=1)
        
        # 转换为张量
        out_pose_uvxyz = torch.from_numpy(out_pose_uvxyz).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_2d = torch.from_numpy(kernel_mean).float()
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_camerapara = torch.from_numpy(out_camerapara).float()
        out_visibility = torch.from_numpy(out_visibility).float()
        visibility_mask = torch.from_numpy(visibility_mask).float()
        
        # 保持与上一个版本一致：返回7个值
        return (out_pose_uvxyz, out_pose_noise_scale, out_pose_2d, 
                out_pose_3d, out_camerapara, out_visibility, visibility_mask)

    def __len__(self):
        return len(self._poses_3d)
    
    def set_augmentation(self, enabled=True):
        """动态启用/禁用数据增强"""
        self.augment_uncertainty = enabled
        
    def set_augmentation_params(self, prob=None, scale=None, num_joints=None):
        """动态调整增强参数"""
        if prob is not None:
            self.augment_prob = prob
        if scale is not None:
            self.uncertainty_scale = scale
        if num_joints is not None:
            self.num_uncertain_joints = num_joints