from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator_gmm_speedplus(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, camerapara, visibility=None):
        assert poses_3d is not None
            
        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm)
        self._camerapara = np.concatenate(camerapara)
        
        # 处理可见性信息
        if visibility is not None:
            self._visibility = np.concatenate(visibility)
        else:
            self._visibility = np.ones((self._poses_3d.shape[0], 11))
        
        self._kernel_n = self._poses_2d_gmm.shape[2] if len(self._poses_2d_gmm.shape) > 3 else 1

        # SPEED+: 中心化3D坐标
        self._poses_3d[:,:,:] = self._poses_3d[:,:,:] - self._poses_3d[:,:1,:]

        assert self._poses_3d.shape[0] == self._poses_2d_gmm.shape[0]
        print('Generating {} poses for SPEED+...'.format(len(self._poses_3d)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_camerapara = self._camerapara[index]
        out_visibility = self._visibility[index]  # (11,)
        
        # 处理2D GMM数据
        if len(self._poses_2d_gmm.shape) > 3:
            # GMM格式处理
            out_pose_2d_gmm = self._poses_2d_gmm[index]  # (11, n_kernels, 5)
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
        
        # 生成uvxyz格式，添加可见性作为额外维度
        out_pose_uvxyz = np.concatenate((kernel_mean, out_pose_3d), axis=1)
        out_pose_noise_scale = np.concatenate(
            (kernel_variance, np.ones(out_pose_3d.shape)), axis=1
        )
        
        # 可见性掩码用于损失计算
        visibility_mask = np.concatenate([
            np.tile(out_visibility[:, np.newaxis], (1, 2)),  # UV维度
            np.tile(out_visibility[:, np.newaxis], (1, 3))   # XYZ维度
        ], axis=1)
        
        # 转换为张量
        out_pose_uvxyz = torch.from_numpy(out_pose_uvxyz).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_2d = torch.from_numpy(kernel_mean).float()
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_camerapara = torch.from_numpy(out_camerapara).float()
        out_visibility = torch.from_numpy(out_visibility).float()
        visibility_mask = torch.from_numpy(visibility_mask).float()
        
        return out_pose_uvxyz, out_pose_noise_scale, out_pose_2d, out_pose_3d, out_camerapara, out_visibility, visibility_mask

    def __len__(self):
        return len(self._poses_3d)