from __future__ import absolute_import, division

import numpy as np
import json
import os
import cv2
import pandas as pd
import torch
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

# SPEED+航天器的11个关键点定义
speedplus_skeleton = Skeleton(
    parents=[-1] * 11,  
    joints_left=[],     
    joints_right=[]
)

# 关键点连接关系（保持不变）
speedplus_edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 7], [1, 4], [2, 5], [3, 6],
    [1, 9], [2, 10],
    [3, 8], [6, 8]
], dtype=np.int32)

# SPEED+相机参数（保持不变）
speedplus_camera = {
    'id': 'speed_camera',
    'center': [960.0, 600.0],
    'focal_length': [2988.58, 2988.34],
    'radial_distortion': [-0.2238, 0.5141, -0.1312],
    'tangential_distortion': [-0.0007, -0.0002],
    'res_w': 1920,
    'res_h': 1200
}

class SpeedPlusDataset(MocapDataset):
    def __init__(self, json_path, keypoints_path, remove_static_joints=False):
        super(SpeedPlusDataset, self).__init__(skeleton=speedplus_skeleton, fps=None)
        
        # 加载3D关键点模板（保持不变）
        self.keypoints_3d_template = self._load_keypoints_template(keypoints_path)
        
        # 加载姿态数据（修改：增加可见性信息）
        self._data = self._load_annotations(json_path)
        
        # 设置相机参数（保持不变）
        self._cameras = self._setup_cameras()
    
    # ==================== 保持不变的方法 ====================
    def _load_keypoints_template(self, excel_path):
        """从Excel文件加载3D关键点模板（保持不变）"""
        df = pd.read_excel(excel_path)
        keypoints = np.zeros((11, 3))
        for i in range(11):
            col_name = f'P{i}'
            keypoints[i, 0] = df.loc[df.index[0], col_name]  # X
            keypoints[i, 1] = df.loc[df.index[1], col_name]  # Y
            keypoints[i, 2] = df.loc[df.index[2], col_name]  # Z
        return keypoints
    
    def _quaternion_to_rotation_matrix(self, q):
        """四元数转旋转矩阵（保持不变）"""
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])
        return R
    
    def _pose_to_keypoints(self, rotation, translation):
        """将姿态转换为3D关键点坐标（保持不变）"""
        R = self._quaternion_to_rotation_matrix(rotation)
        keypoints_3d = (R @ self.keypoints_3d_template.T).T + translation
        return keypoints_3d
    
    def _setup_cameras(self):
        """设置相机参数（保持不变）"""
        cameras = {'spacecraft': [speedplus_camera.copy()]}
        
        cam = cameras['spacecraft'][0]
        cam['center'] = normalize_screen_coordinates(
            np.array(cam['center']), w=cam['res_w'], h=cam['res_h']
        ).astype('float32')
        
        cam['focal_length'] = np.array(cam['focal_length']) / cam['res_w'] * 2.0
        cam['radial_distortion'] = np.array(cam['radial_distortion'])
        cam['tangential_distortion'] = np.array(cam['tangential_distortion'])
        
        cam['intrinsic'] = np.concatenate((
            cam['focal_length'],
            cam['center'],
            cam['radial_distortion'][:2],  
            cam['tangential_distortion'],   
            cam['radial_distortion'][2:3]   
        ))
        
        cam['K'] = np.array([
            [2988.58, 0, 960],
            [0, 2988.34, 600],
            [0, 0, 1]
        ])
        cam['dist_coeffs'] = np.array([-0.2238, 0.5141, -0.0007, -0.0002, -0.1312])
        
        return cameras
    
    # ==================== 修改的方法 ====================
    def _load_annotations(self, json_path):
        """
        加载JSON标注数据，直接读取可见性信息
        
        修改原因：需要从JSON中提取可见性信息
        作用：将数据集提供的可见性信息加载到内存中
        """
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        data = {}
        data['spacecraft'] = {}
        
        for idx, ann in enumerate(annotations):
            image_name = ann['filename'].split('.')[0]
            translation = np.array(ann['r_Vo2To_vbs_true'])
            rotation = np.array(ann['q_vbs2tango_true'])
            
            # 转换为3D关键点（保持不变）
            keypoints_3d = self._pose_to_keypoints(rotation, translation)
            
            # 【关键修改】直接从JSON中读取可见性信息
            if 'visibility' in ann:
                visibility_mask = np.array(ann['visibility'], dtype=bool)
            else:
                # 如果没有可见性信息，默认全部可见
                visibility_mask = np.ones(11, dtype=bool)
                print(f"Warning: No visibility info for {image_name}, assuming all visible")
            
            data['spacecraft'][image_name] = {
                'positions': keypoints_3d[np.newaxis, :, :],  
                'cameras': [speedplus_camera],
                'rotation': rotation,
                'translation': translation,
                'visibility_mask': visibility_mask  # 【新增】存储可见性mask
            }
        
        return data
    
    # ==================== 新增的必要方法 ====================
    def create_attention_mask(self, visibility_mask):
        """
        根据可见性创建注意力mask
        
        修改原因：需要将布尔型可见性mask转换为模型可用的注意力mask格式
        作用：为GraFormer的注意力机制提供正确格式的mask
        """
        mask = torch.tensor(visibility_mask, dtype=torch.bool)
        attention_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 11]
        return attention_mask
    
    def get_visibility_stats(self):
        """
        获取数据集的可见性统计信息
        
        修改原因：提供可见性分析功能，帮助理解数据集特征
        作用：统计各关键点的可见率，为模型训练提供参考
        """
        total_samples = 0
        total_keypoints = 0
        visible_keypoints = 0
        visibility_per_keypoint = np.zeros(11)
        
        for subject in self.subjects():
            for action in self[subject].keys():
                if 'visibility_mask' in self[subject][action]:
                    visibility_mask = self[subject][action]['visibility_mask']
                    total_samples += 1
                    total_keypoints += 11
                    visible_keypoints += np.sum(visibility_mask)
                    visibility_per_keypoint += visibility_mask.astype(float)
        
        if total_samples > 0:
            visibility_per_keypoint /= total_samples
            overall_visibility_rate = visible_keypoints / total_keypoints
            
            print(f"数据集可见性统计:")
            print(f"总样本数: {total_samples}")
            print(f"整体可见性率: {overall_visibility_rate:.3f}")
            print(f"各关键点可见性率:")
            for i in range(11):
                print(f"  关键点{i}: {visibility_per_keypoint[i]:.3f}")
        
        return {
            'total_samples': total_samples,
            'overall_visibility_rate': overall_visibility_rate,
            'keypoint_visibility_rates': visibility_per_keypoint
        }