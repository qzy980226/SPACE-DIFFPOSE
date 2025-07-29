from __future__ import absolute_import, division

import numpy as np
import json
import os
import cv2
import pandas as pd
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates
from common.camera import denormalize_screen_coordinates

# SPEED+航天器的11个关键点定义
speedplus_skeleton = Skeleton(
    parents=[-1] * 11,  
    joints_left=[],     
    joints_right=[]
)

# 关键点连接关系
speedplus_edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 7], [1, 4], [2, 5], [3, 6],
    [1, 9], [2, 10],
    [3, 8], [6, 8]
], dtype=np.int32)

# SPEED+相机参数
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
        
        # 加载3D关键点
        self.keypoints_3d_template = self._load_keypoints_template(keypoints_path)
        
        # 加载姿态数据
        self._data = self._load_annotations(json_path)
        
        # 设置相机参数
        self._cameras = self._setup_cameras()
    
    def _load_keypoints_template(self, excel_path):
        """从Excel文件加载3D关键点模板"""
        df = pd.read_excel(excel_path)
        keypoints = np.zeros((11, 3))
        for i in range(11):
            col_name = f'P{i}'
            keypoints[i, 0] = df.loc[df.index[0], col_name]  # X
            keypoints[i, 1] = df.loc[df.index[1], col_name]  # Y
            keypoints[i, 2] = df.loc[df.index[2], col_name]  # Z
        return keypoints
    
    def _quaternion_to_rotation_matrix(self, q):
        """四元数转旋转矩阵"""
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])
        return R
    
    def _pose_to_keypoints(self, rotation, translation):
        """将姿态（旋转+平移）转换为3D关键点坐标"""
        R = self._quaternion_to_rotation_matrix(rotation)
        keypoints_3d = (R @ self.keypoints_3d_template.T).T + translation
        return keypoints_3d
    
    def _load_annotations(self, json_path):
        """加载JSON格式的标注数据"""
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        data = {}
        data['spacecraft'] = {}
        
        # 读取数据
        for idx, ann in enumerate(annotations):
            image_name = ann['filename'].split('.')[0]
            translation = np.array(ann['r_Vo2To_vbs_true'])
            rotation = np.array(ann['q_vbs2tango_true'])
            
            # 转换为3D关键点
            keypoints_3d = self._pose_to_keypoints(rotation, translation)
            
            data['spacecraft'][image_name] = {
                'positions': keypoints_3d[np.newaxis, :, :],  
                'cameras': [speedplus_camera],
                'rotation': rotation,
                'translation': translation
            }
        
        return data
    
    def _setup_cameras(self):
        cameras = {'spacecraft': [speedplus_camera.copy()]}
        
        # 转换为与Human3.6M兼容的格式
        cam = cameras['spacecraft'][0]
        
        # 归一化相机参数
        cam['center'] = normalize_screen_coordinates(
            np.array(cam['center']), w=cam['res_w'], h=cam['res_h']
        ).astype('float32')
        
        # 归一化焦距（相对于图像宽度）
        cam['focal_length'] = np.array(cam['focal_length']) / cam['res_w'] * 2.0
        
        # 转换为numpy数组
        cam['radial_distortion'] = np.array(cam['radial_distortion'])
        cam['tangential_distortion'] = np.array(cam['tangential_distortion'])
        
        # 创建完整的内参向量（9维：fx, fy, cx, cy, k1, k2, p1, p2, k3）
        cam['intrinsic'] = np.concatenate((
            cam['focal_length'],
            cam['center'],
            cam['radial_distortion'][:2],  # k1, k2
            cam['tangential_distortion'],   # p1, p2
            cam['radial_distortion'][2:3]   # k3
        ))
        
        # 添加用于反投影的原始参数
        cam['K'] = np.array([
            [2988.58, 0, 960],
            [0, 2988.34, 600],
            [0, 0, 1]
        ])
        cam['dist_coeffs'] = np.array([-0.2238, 0.5141, -0.0007, -0.0002, -0.1312])
        
        return cameras
    
    