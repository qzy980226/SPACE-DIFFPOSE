from __future__ import absolute_import, division

import numpy as np

from .camera import normalize_screen_coordinates
from common.camera import project_to_2d_speedplus
#数据集加载，预处理，格式转换
speedplus_camera_dict = {
    'speed_camera': [2988.58/1920*2, 2988.34/1920*2, 0.0, 0.0]  # 归一化焦距和主点
}

def read_3d_data_speedplus(dataset):
    """处理SPEED+数据集的3D数据，包含可见性信息"""
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            # SPEED+已经在相机坐标系中，不需要世界到相机的转换
            positions_3d = [anim['positions']]
            anim['positions_3d'] = positions_3d
            
            # 确保可见性信息也被保留
            if 'visibility' in anim:
                anim['visibility'] = anim['visibility']
            else:
                # 如果没有可见性信息，默认全部可见
                n_frames = positions_3d[0].shape[0]
                anim['visibility'] = np.ones((n_frames, 11))
    
    return dataset

def create_2d_data_speedplus(dataset):
    """为SPEED+创建2D投影数据"""
    keypoints_2d = {}
    
    for subject in dataset.subjects():
        keypoints_2d[subject] = {}
        for action in dataset[subject].keys():
            # 获取3D关键点和相机参数
            positions_3d = dataset[subject][action]['positions_3d'][0]
            cam = dataset.cameras()[subject][0]
            
            # 投影到2D
            positions_2d = project_to_2d_speedplus(positions_3d, cam)
            
            # 归一化屏幕坐标
            positions_2d = normalize_screen_coordinates(
                positions_2d, w=cam['res_w'], h=cam['res_h']
            )
            
            keypoints_2d[subject][action] = [positions_2d]
    
    return keypoints_2d

def project_to_2d_speedplus(positions_3d, camera):
    """SPEED+的2D投影函数"""
    # 构建相机参数向量
    camera_params = []
    
    # 归一化的内参
    fx_norm = camera['focal_length'][0]
    fy_norm = camera['focal_length'][1]
    cx_norm = camera['center'][0]
    cy_norm = camera['center'][1]
    
    # 畸变系数
    k1, k2, k3 = camera['radial_distortion']
    p1, p2 = camera['tangential_distortion']
    
    # 组合成参数向量
    for i in range(positions_3d.shape[0]):
        params = np.array([fx_norm, fy_norm, cx_norm, cy_norm, k1, k2, p1, p2, k3])
        camera_params.append(params)
    
    camera_params = np.stack(camera_params)
    
    # 使用camera.py中的投影函数
    return project_to_2d_speedplus(positions_3d, camera_params, use_cv2=True)

def create_2d_data_speedplus_gmm(dataset, sigma=1.0):
    """为SPEED+创建2D GMM数据，包含零掩码处理"""
    keypoints_2d_gmm = {}
    
    for subject in dataset.subjects():
        keypoints_2d_gmm[subject] = {}
        for action in dataset[subject].keys():
            data = dataset[subject][action]
            positions_2d = data.get('positions_2d', data['positions_3d'][0])
            visibility = data.get('visibility', np.ones((1, 11)))[0]
            
            # 创建GMM表示
            n_frames = positions_2d.shape[0]
            n_joints = 11
            n_kernels = 5  # GMM核数量
            
            gmm_data = np.zeros((n_frames, n_joints, n_kernels, 5))  # [prob, mean_x, mean_y, var_x, var_y]
            
            for frame_idx in range(n_frames):
                for joint_idx in range(n_joints):
                    if visibility[joint_idx] > 0:
                        # 可见关键点：创建GMM
                        pos_2d = positions_2d[frame_idx, joint_idx]
                        
                        # 主核（高概率）
                        gmm_data[frame_idx, joint_idx, 0, 0] = 0.7  # 概率
                        gmm_data[frame_idx, joint_idx, 0, 1:3] = pos_2d  # 均值
                        gmm_data[frame_idx, joint_idx, 0, 3:5] = sigma  # 方差
                        
                        # 辅助核（低概率）
                        for k in range(1, n_kernels):
                            gmm_data[frame_idx, joint_idx, k, 0] = 0.3 / (n_kernels - 1)
                            # 添加小的偏移
                            offset = np.random.randn(2) * sigma * 2
                            gmm_data[frame_idx, joint_idx, k, 1:3] = pos_2d + offset
                            gmm_data[frame_idx, joint_idx, k, 3:5] = sigma * 2
                    else:
                        # 不可见关键点：零掩码
                        # 所有核的概率设为均匀分布，位置设为图像中心
                        for k in range(n_kernels):
                            gmm_data[frame_idx, joint_idx, k, 0] = 1.0 / n_kernels
                            gmm_data[frame_idx, joint_idx, k, 1:3] = [0, 0]  # 归一化坐标的中心
                            gmm_data[frame_idx, joint_idx, k, 3:5] = sigma * 10  # 大方差表示不确定性
            
            keypoints_2d_gmm[subject][action] = gmm_data
            
    return keypoints_2d_gmm



def fetch_speedplus(subjects, dataset, keypoints, stride=1, parse_3d_poses=True):
    """获取SPEED+数据，包含可见性信息"""
    out_poses_3d = []
    out_poses_2d = []
    out_camera_para = []
    out_visibility = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            poses_2d = keypoints[subject][action]
            visibility = dataset[subject][action].get('visibility', 
                                                      np.ones((poses_2d.shape[0], 11)))
            
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])
                out_visibility.append(visibility[i] if i < len(visibility) else visibility[0])
            
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                camera = dataset.cameras()[subject][0]
                
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])
                    cam_param = [
                        camera['focal_length'][0],
                        camera['focal_length'][1], 
                        camera['center'][0],
                        camera['center'][1]
                    ]
                    out_camera_para.append([cam_param] * poses_3d[i].shape[0])
    
    return out_poses_3d, out_poses_2d, out_camera_para, out_visibility