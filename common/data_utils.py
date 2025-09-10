from __future__ import absolute_import, division

import numpy as np
import torch

from .camera import normalize_screen_coordinates
from common.camera import project_to_2d_speedplus

# 原有的speedplus_camera_dict（保持不变）
speedplus_camera_dict = {
    'speed_camera': [2988.58/1920*2, 2988.34/1920*2, 0.0, 0.0]
}

# ==================== 原有函数（保持不变） ====================
def read_3d_data_speedplus(dataset):
    """处理SPEED+数据集的3D数据（保持不变）"""
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = [anim['positions']]
            anim['positions_3d'] = positions_3d
    return dataset

def create_2d_data_speedplus(dataset):
    """为SPEED+创建2D投影数据（保持不变）"""
    keypoints_2d = {}
    
    for subject in dataset.subjects():
        keypoints_2d[subject] = {}
        for action in dataset[subject].keys():
            positions_3d = dataset[subject][action]['positions_3d'][0]
            cam = dataset.cameras()[subject][0]
            
            positions_2d = project_to_2d_speedplus(positions_3d, cam)
            positions_2d = normalize_screen_coordinates(
                positions_2d, w=cam['res_w'], h=cam['res_h']
            )
            
            keypoints_2d[subject][action] = [positions_2d]
    
    return keypoints_2d

def project_to_2d_speedplus(positions_3d, camera):
    """SPEED+的2D投影函数（保持不变）"""
    camera_params = []
    
    fx_norm = camera['focal_length'][0]
    fy_norm = camera['focal_length'][1]
    cx_norm = camera['center'][0]
    cy_norm = camera['center'][1]
    
    k1, k2, k3 = camera['radial_distortion']
    p1, p2 = camera['tangential_distortion']
    
    for i in range(positions_3d.shape[0]):
        params = np.array([fx_norm, fy_norm, cx_norm, cy_norm, k1, k2, p1, p2, k3])
        camera_params.append(params)
    
    camera_params = np.stack(camera_params)
    return project_to_2d_speedplus(positions_3d, camera_params, use_cv2=True)

# ==================== 新增函数（支持可见性） ====================
def fetch_speedplus_with_visibility(subjects, dataset, keypoints, stride=1, parse_3d_poses=True):
    """
    获取SPEED+数据，同时返回可见性信息
    
    修改原因：需要同时获取姿态数据和可见性mask
    作用：为训练提供完整的数据（包含可见性）
    """
    out_poses_3d = []
    out_poses_2d = []
    out_visibility_masks = []
    out_actions = []
    out_subjects = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            poses_2d = keypoints[subject][action]
            
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), '2D和3D姿态数量必须匹配'
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])
                    
            # 提取可见性mask
            if 'visibility_mask' in dataset[subject][action]:
                visibility_mask = dataset[subject][action]['visibility_mask']
                for i in range(len(poses_2d)):
                    out_visibility_masks.append(visibility_mask)
            else:
                # 如果没有可见性信息，默认所有关键点都可见
                default_mask = np.ones(11, dtype=bool)
                for i in range(len(poses_2d)):
                    out_visibility_masks.append(default_mask)

            out_subjects.extend([subject] * len(poses_2d))
            out_actions.extend([action] * len(poses_2d))

    if stride > 1:
        out_poses_2d = out_poses_2d[::stride]
        out_visibility_masks = out_visibility_masks[::stride]
        out_actions = out_actions[::stride]
        out_subjects = out_subjects[::stride]
        if parse_3d_poses:
            out_poses_3d = out_poses_3d[::stride]

    if len(out_poses_3d) == 0:
        out_poses_3d = None
        
    return out_poses_2d, out_poses_3d, out_visibility_masks, out_actions, out_subjects

def create_dynamic_attention_masks(visibility_masks):
    """
    为批处理数据创建动态注意力masks
    
    修改原因：模型需要批处理级别的mask输入
    作用：将numpy格式的可见性mask转换为torch tensor格式
    """
    batch_masks = []
    
    for visibility_mask in visibility_masks:
        mask = torch.tensor(visibility_mask, dtype=torch.bool)
        attention_mask = mask.unsqueeze(0)  # [1, 11]
        batch_masks.append(attention_mask)
    
    batch_attention_masks = torch.stack(batch_masks)  # [batch_size, 1, 11]
    return batch_attention_masks

def compute_visibility_statistics(visibility_masks):
    """
    计算一个批次中的可见性统计
    
    修改原因：需要监控训练过程中的可见性分布
    作用：提供可见性分析，帮助理解数据特征
    """
    if len(visibility_masks) == 0:
        return {}
    
    visibility_array = np.stack(visibility_masks)  # [batch_size, 11]
    
    batch_size = visibility_array.shape[0]
    total_keypoints = batch_size * 11
    visible_keypoints = np.sum(visibility_array)
    
    visibility_rate_per_sample = np.mean(visibility_array, axis=1)
    visibility_rate_per_keypoint = np.mean(visibility_array, axis=0)
    
    return {
        'batch_size': batch_size,
        'total_keypoints': total_keypoints,
        'visible_keypoints': visible_keypoints,
        'overall_visibility_rate': visible_keypoints / total_keypoints,
        'visibility_rate_per_sample': visibility_rate_per_sample,
        'visibility_rate_per_keypoint': visibility_rate_per_keypoint,
        'min_visible_per_sample': np.min(np.sum(visibility_array, axis=1)),
        'max_visible_per_sample': np.max(np.sum(visibility_array, axis=1))
    }

# ==================== 兼容性函数 ====================
def fetch_speedplus(subjects, dataset, keypoints, stride=1, parse_3d_poses=True):
    """原有的fetch函数，为了向后兼容"""
    poses_2d, poses_3d, visibility_masks, actions, subjects = fetch_speedplus_with_visibility(
        subjects, dataset, keypoints, stride, parse_3d_poses
    )
    
    # 返回原有格式（不包含可见性信息）
    return poses_2d, poses_3d, actions, subjects