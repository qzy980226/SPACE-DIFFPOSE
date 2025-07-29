from __future__ import absolute_import, division

import numpy as np

from .camera import normalize_screen_coordinates
from common.camera import project_to_2d_speedplus
#数据集加载，预处理，格式转换
speedplus_camera_dict = {
    'speed_camera': [2988.58/1920*2, 2988.34/1920*2, 0.0, 0.0]  # 归一化焦距和主点
}

def read_3d_data_speedplus(dataset):
    """处理SPEED+数据集的3D数据"""
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            # SPEED+已经在相机坐标系中，不需要世界到相机的转换
            positions_3d = [anim['positions']]
            anim['positions_3d'] = positions_3d
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


def fetch_speedplus(subjects, dataset, keypoints, stride=1, parse_3d_poses=True):
    """获取SPEED+数据，移除action相关"""
    out_poses_3d = []
    out_poses_2d = []
    out_camera_para = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            poses_2d = keypoints[subject][action]
            
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])
            
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                camera = dataset.cameras()[subject][0]
                
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])
                    # SPEED+相机参数格式
                    cam_param = [
                        camera['focal_length'][0],
                        camera['focal_length'][1], 
                        camera['center'][0],
                        camera['center'][1]
                    ]
                    out_camera_para.append([cam_param] * poses_3d[i].shape[0])
    
    return out_poses_3d, out_poses_2d, out_camera_para