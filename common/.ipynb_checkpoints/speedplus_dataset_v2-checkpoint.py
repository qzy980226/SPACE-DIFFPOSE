"""
新的SPEED+ V2数据集加载器
修改原因：需要处理新的数据格式（train.json + kpts.mat + GMM npz文件）
"""
from __future__ import absolute_import, division

import numpy as np
import json
import os
import scipy.io as sio
from glob import glob
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

# SPEED+航天器的11个关键点定义
speedplus_skeleton = Skeleton(
    parents=[-1] * 11,  
    joints_left=[],     
    joints_right=[]
)

class SpeedPlusV2Dataset(MocapDataset):
    def __init__(self, train_json_path, kpts_mat_path, gmm_data_path):
        super(SpeedPlusV2Dataset, self).__init__(skeleton=speedplus_skeleton, fps=None)
        
        # 加载3D关键点模板
        self.keypoints_3d_template = self._load_kpts_mat(kpts_mat_path)
        
        # 加载训练数据
        self.annotations = self._load_train_json(train_json_path)
        
        # 加载GMM参数
        self.gmm_data = self._load_gmm_data(gmm_data_path)
        
        
        # 设置相机参数（使用内置的SPEED+相机参数）
        self._cameras = self._setup_cameras()
        
        # 构建数据集
        self._data = self._build_dataset()
        
        
    
    def _load_kpts_mat(self, mat_path):
        """加载3D关键点模板，从3×11转换为11×3"""
        mat_data = sio.loadmat(mat_path)
        kpts = mat_data['corners']  
        if kpts.shape == (3, 11):
            kpts = kpts.T  # 转置为11×3
        return kpts
    
    def _load_train_json(self, json_path):
        """加载训练标注"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _load_gmm_data(self, gmm_path):
        """加载所有GMM参数文件"""
        gmm_files = sorted(glob(os.path.join(gmm_path, 'img*_gmm_params.npz')))
        gmm_dict = {}
        
        for gmm_file in gmm_files:
            # 提取图像名（如img000001）
            basename = os.path.basename(gmm_file)
            img_name = basename.split('_gmm_params')[0]
            
            # 加载GMM参数
            data = np.load(gmm_file, allow_pickle=True)
            gmm_dict[img_name] = {
                'gmm_means': data['gmm_means'],
                'gmm_covariances': data['gmm_covariances'],
                'gmm_weights': data['gmm_weights'],  # 用作概率
                'success_flags': data['success_flags']  # 可见性
            }
        
        return gmm_dict
    
    def _quaternion_to_rotation_matrix(self, q):
        """四元数转旋转矩阵"""
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])
        return R
    
    def _build_dataset(self):
        """构建数据集结构"""
        data = {'spacecraft': {}}
        
        for ann in self.annotations:
            img_name = ann['filename'].split('.')[0]  # 去除扩展名
            
            # 获取姿态参数
            translation = np.array(ann['r_Vo2To_vbs_true'])
            quaternion = np.array(ann['q_vbs2tango_true'])
            
            # 转换为3D关键点
            R = self._quaternion_to_rotation_matrix(quaternion)
            keypoints_3d = (R @ self.keypoints_3d_template.T).T + translation
            
            # 获取GMM参数
            if img_name in self.gmm_data:
                gmm = self.gmm_data[img_name]
                data['spacecraft'][img_name] = {
                    'positions': keypoints_3d[np.newaxis, :, :],  # (1, 11, 3)
                    'gmm_means': gmm['gmm_means'],  # (11, 2)
                    'gmm_covariances': gmm['gmm_covariances'],  # (11, 2, 2)
                    'gmm_weights': gmm['gmm_weights'],  # (11,) 作为概率
                    'visibility': gmm['success_flags'].astype(np.float32),  # (11,)
                    'rotation': quaternion,
                    'translation': translation,
                    'cameras': self._cameras['spacecraft']
                }
        
        return data
    
    def _setup_cameras(self):
        """使用内置的SPEED+相机参数"""
        from common.speedplus_dataset import speedplus_camera
        return {'spacecraft': [speedplus_camera.copy()]}