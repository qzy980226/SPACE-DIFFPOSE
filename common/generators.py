from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator_gmm_speedplus(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, camerapara, visibility,
                 augment_uncertainty=True, augment_prob=0.5, 
                 uncertainty_scale=50, num_uncertain_joints=2):
        """
        保留所有参数理由：
        - augment_*: 数据增强功能，提升模型泛化
        - 这些参数在原项目中已存在，不应删除
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
        
        # 保留数据增强参数（重要：不应删除）
        self.augment_uncertainty = augment_uncertainty
        self.augment_prob = augment_prob
        self.uncertainty_scale = uncertainty_scale
        self.num_uncertain_joints = num_uncertain_joints
        
        # 中心化3D坐标
        self._poses_3d[:,:,:] = self._poses_3d[:,:,:] - self._poses_3d[:,:1,:]
        
        print(f'Generating {len(self._poses_3d)} poses for SPEED+ V2...')

    def __getitem__(self, index):
        """
        out_pose_3d = self._poses_3d[index]
        out_camerapara = self._camerapara[index]
        out_visibility = self._visibility[index]  # (11,)
        
        # 处理2D GMM数据
        if len(self._poses_2d_gmm.shape) > 3:
            # GMM格式处理
            out_pose_2d_gmm = self._poses_2d_gmm[index]  # (11, n_kernels, 5)
            
            if self.augment_uncertainty and np.random.random() < self.augment_prob:
                # 找出所有可见的关键点
                visible_joints = np.where(out_visibility > 0)[0]
                
                if len(visible_joints) >= self.num_uncertain_joints:
                    # 随机选择要增加不确定性的关键点
                    uncertain_joints = np.random.choice(
                        visible_joints, 
                        size=min(self.num_uncertain_joints, len(visible_joints)),
                        replace=False
                    )
                    
                    # 对选中的关键点增加不确定性
                    for joint_idx in uncertain_joints:
                        # 扩大所有核的方差
                        out_pose_2d_gmm[joint_idx, :, 3:5] *= self.uncertainty_scale
            
            
            out_pose_2d_kernel = np.zeros([out_pose_2d_gmm.shape[0], out_pose_2d_gmm.shape[2]])
            
            for i in range(out_pose_2d_gmm.shape[0]):
                if out_visibility[i] > 0:
                    # 可见关键点：正常采样
                    kernel_idx = np.random.choice(
                        self._kernel_n, 1, 
                        p=out_pose_2d_gmm[i,:,0]
                    ).item()
                    out_pose_2d_kernel[i] = out_pose_2d_gmm[i, kernel_idx]
                else:
                    # 不可见关键点：使用零掩码
                    out_pose_2d_kernel[i, 0] = 1.0  # 概率
                    out_pose_2d_kernel[i, 1:3] = [0, 0]  # 中心位置
                    out_pose_2d_kernel[i, 3:] = [10.0, 10.0]  # 大方差
            
            kernel_mean = out_pose_2d_kernel[:,1:3]
            kernel_variance = out_pose_2d_kernel[:,3:]
        else:
            # 简单2D坐标
            kernel_mean = self._poses_2d_gmm[index].copy()
            kernel_variance = np.ones_like(kernel_mean) * 0.1
            
            # 对不可见点应用零掩码
            kernel_mean[out_visibility <= 0] = [0, 0]
            kernel_variance[out_visibility <= 0] = [10.0, 10.0]
        """
        
    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_camerapara = self._camerapara[index]
        out_visibility = self._visibility[index]  # (11,)
        
        # 处理2D GMM数据
        if len(self._poses_2d_gmm.shape) > 3:
            out_pose_2d_gmm = self._poses_2d_gmm[index].copy()
            
            # 数据增强
            if self.augment_uncertainty and np.random.random() < self.augment_prob:
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