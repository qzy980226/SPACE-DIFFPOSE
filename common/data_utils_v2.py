"""
SPEED+ V2数据处理工具
修改原因：需要处理GMM格式的数据
"""
import numpy as np

def create_2d_data_speedplus_v2(dataset):
    """
    创建SPEED+ V2的2D GMM数据
    修改原因：直接使用预计算的GMM参数，而不是生成
    """
    keypoints_2d_gmm = {}
    
    for subject in dataset.subjects():
        keypoints_2d_gmm[subject] = {}
        for action in dataset[subject].keys():
            data = dataset[subject][action]
            
            # 构建GMM格式数据
            # 格式：(n_frames, n_joints, n_kernels, 5)
            # 5维：[概率, mean_x, mean_y, var_x, var_y]
            
            n_frames = 1  # 单帧
            n_joints = 11
            n_kernels = 1  # 单个GMM组件
            
            gmm_data = np.zeros((n_frames, n_joints, n_kernels, 5))
            
            for j in range(n_joints):
                # 使用gmm_weights作为概率
                gmm_data[0, j, 0, 0] = data['gmm_weights'][j]
                
                # 归一化坐标（从像素到[-1, 1]）
                mean_x = (data['gmm_means'][j][0] / 1920) * 2 - 1
                mean_y = (data['gmm_means'][j][1] / 1200) * 2 - 1
                gmm_data[0, j, 0, 1:3] = [mean_x, mean_y]
                
                # 提取方差（协方差矩阵的对角线）
                var_x = data['gmm_covariances'][j][0, 0]
                var_y = data['gmm_covariances'][j][1, 1]
                gmm_data[0, j, 0, 3:5] = [var_x, var_y]
            
            keypoints_2d_gmm[subject][action] = gmm_data
    
    return keypoints_2d_gmm

def fetch_speedplus_v2(subjects, dataset, keypoints, stride=1):
    """
    获取SPEED+ V2数据
    修改原因：适配新的数据格式，包含GMM weights作为概率
    """
    out_poses_3d = []
    out_poses_2d_gmm = []
    out_camera_para = []
    out_visibility = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            data = dataset[subject][action]
            
            # 3D姿态
            out_poses_3d.append(data['positions'])
            
            # 2D GMM数据
            out_poses_2d_gmm.append(keypoints[subject][action])
            
            # 可见性（从success_flags）
            out_visibility.append(data['visibility'].reshape(1, -1))
            
            # 相机参数（使用内置的）
            from common.speedplus_dataset import speedplus_camera
            cam_params = [
                speedplus_camera['focal_length'][0],
                speedplus_camera['focal_length'][1],
                speedplus_camera['center'][0], 
                speedplus_camera['center'][1]
            ]
            out_camera_para.append([cam_params])
    
    return out_poses_3d, out_poses_2d_gmm, out_camera_para, out_visibility