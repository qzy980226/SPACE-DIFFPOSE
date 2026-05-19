"""
Lightbox数据集验证脚本（SPEED+格式）
适用于实际的lightbox数据：
- test.json: 真实四元数与平移向量
- kpts.mat: 3D模型关键点
- camera.json: 相机参数
- GMM heatmap数据: .json格式

输出格式（针对每张图像）：
1. 预测的旋转四元数 (w, x, y, z)
2. 预测的平移向量 (tx, ty, tz)
3. 该图像的可见性掩码MPJPE
"""

import os
import logging
import argparse
import json
import glob
import numpy as np
import torch
import torch.utils.data as data
from scipy.spatial.transform import Rotation
from scipy.io import loadmat
from tqdm import tqdm
import cv2

from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff
from common.utils_diff import get_beta_schedule, compute_alpha
from common.loss import mpjpe, p_mpjpe  # 使用支持visibility_mask的版本
# 注意：不要 from common.utils import *，避免覆盖 p_mpjpe


class LightboxDataset(torch.utils.data.Dataset):
    """
    Lightbox数据集加载器（SPEED+格式）

    数据文件：
    - test.json: {'filename': {'q_vbs2tango': [...], 'r_Vo2To_vbs_true': [...]}}
    - kpts.mat: {'kpts': (3, 11)} - 3D模型关键点
    - camera.json: {'fx', 'fy', 'cx', 'cy', 'width', 'height'}
    - GMM heatmaps: {filename}.json
    """

    def __init__(self, gmm_dir, test_json_path, kpts_mat_path, camera_json_path):
        """
        Args:
            gmm_dir: GMM格式.json文件所在目录
            test_json_path: test.json路径（真实姿态）
            kpts_mat_path: kpts.mat路径（3D模型）
            camera_json_path: camera.json路径（相机参数）
        """
        self.gmm_dir = gmm_dir

        # 加载所有GMM文件
        self.gmm_files = sorted(glob.glob(os.path.join(gmm_dir, '*.json')))
        if len(self.gmm_files) == 0:
            raise ValueError(f"No .json files found in {gmm_dir}")

        # 加载test.json（真实姿态）
        print(f"Loading test.json from {test_json_path}")
        with open(test_json_path, 'r') as f:
            test_data_raw = json.load(f)

        # test.json是数组格式，转换为字典格式 {filename: data}
        if isinstance(test_data_raw, list):
            self.test_data = {item['filename']: item for item in test_data_raw}
            print(f"  Loaded {len(self.test_data)} annotations from test.json (array format)")
        else:
            # 如果已经是字典格式
            self.test_data = test_data_raw
            print(f"  Loaded {len(self.test_data)} annotations from test.json (dict format)")

        # 加载kpts.mat（3D模型）
        print(f"Loading kpts.mat from {kpts_mat_path}")
        mat_data = loadmat(kpts_mat_path)

        # 尝试不同的键名（根据speedplus_dataset_v2.py，键名是'corners'）
        if 'corners' in mat_data:
            kpts = mat_data['corners']
        elif 'kpts' in mat_data:
            kpts = mat_data['kpts']
        else:
            # 列出所有可用的键
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            raise KeyError(f"Could not find keypoints in .mat file. Available keys: {available_keys}")

        # 转置为 (11, 3) 如果需要
        if kpts.shape == (3, 11):
            self.kpts_3d_model = kpts.T
        else:
            self.kpts_3d_model = kpts

        print(f"  3D model keypoints shape: {self.kpts_3d_model.shape}")

        # 加载camera.json（相机参数）
        print(f"Loading camera.json from {camera_json_path}")
        with open(camera_json_path, 'r') as f:
            camera_data = json.load(f)

        # SPEED+格式：键名可能是 cx/cy 或 ccx/ccy
        cx = camera_data.get('cx', camera_data.get('ccx', 0))
        cy = camera_data.get('cy', camera_data.get('ccy', 0))

        self.camera_intrinsics = np.array([
            [camera_data['fx'], 0, cx],
            [0, camera_data['fy'], cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.camera_params = camera_data

        print(f"  Camera intrinsics:\n{self.camera_intrinsics}")
        print(f"Loaded {len(self.gmm_files)} GMM files from Lightbox dataset")

    def _convert_gmm_to_2d_poses(self, gmm_params, visibility, n_components=2):
        """
        将GMM参数转换为SPEED+格式的2D姿态数据

        Args:
            gmm_params: dict {kp_idx: {'weights', 'means', 'covariances', ...}}
            visibility: [11] bool array
            n_components: GMM分量数量

        Returns:
            poses_2d_gmm: (11, n_components, 5) - [weight, u, v, sigma_u, sigma_v]
        """
        poses_2d_gmm = np.zeros((11, n_components, 5), dtype=np.float32)

        for kp_idx in range(11):
            # JSON中的键是字符串，需要转换
            kp_key = str(kp_idx)
            if visibility[kp_idx] and kp_key in gmm_params:
                # 提取GMM参数（转换为numpy数组以支持索引）
                params = gmm_params[kp_key]
                weights = np.array(params['weights'])      # [n_comp]
                means = np.array(params['means'])          # [n_comp, 2] - (x, y)
                covs = np.array(params['covariances'])     # [n_comp, 2] - diagonal covariance

                # 确保有足够的分量
                actual_n_comp = len(weights)
                n_to_use = min(actual_n_comp, n_components)

                for i in range(n_to_use):
                    poses_2d_gmm[kp_idx, i, 0] = weights[i]          # weight
                    poses_2d_gmm[kp_idx, i, 1] = means[i][0]         # u (x坐标)
                    poses_2d_gmm[kp_idx, i, 2] = means[i][1]         # v (y坐标)
                    poses_2d_gmm[kp_idx, i, 3] = np.sqrt(covs[i][0]) # sigma_u
                    poses_2d_gmm[kp_idx, i, 4] = np.sqrt(covs[i][1]) # sigma_v

                # 如果实际分量少于n_components，用第一个分量填充
                if actual_n_comp < n_components:
                    for i in range(actual_n_comp, n_components):
                        poses_2d_gmm[kp_idx, i] = poses_2d_gmm[kp_idx, 0]
            else:
                # 不可见关键点：设置低权重和高方差
                for i in range(n_components):
                    poses_2d_gmm[kp_idx, i] = [0.1, 0, 0, 10.0, 10.0]

        return poses_2d_gmm

    def _sample_from_gmm(self, poses_2d_gmm, visibility):
        """
        从GMM分布中采样2D关键点坐标

        Args:
            poses_2d_gmm: (11, n_components, 5)
            visibility: (11,) bool array

        Returns:
            kernel_mean: (11, 2) - 采样的2D坐标
            kernel_variance: (11, 2) - 采样的方差
        """
        n_components = poses_2d_gmm.shape[1]
        out_pose_2d_kernel = np.zeros([11, 5])

        for i in range(11):
            if visibility[i]:
                if n_components == 1:
                    out_pose_2d_kernel[i] = poses_2d_gmm[i, 0]
                else:
                    probs = poses_2d_gmm[i, :, 0]
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        kernel_idx = np.random.choice(n_components, p=probs)
                        out_pose_2d_kernel[i] = poses_2d_gmm[i, kernel_idx]
                    else:
                        out_pose_2d_kernel[i] = poses_2d_gmm[i, 0]
            else:
                out_pose_2d_kernel[i] = [0.1, 0, 0, 10.0, 10.0]

        kernel_mean = out_pose_2d_kernel[:, 1:3]      # (u, v)
        kernel_variance = out_pose_2d_kernel[:, 3:5]  # (sigma_u, sigma_v)

        return kernel_mean, kernel_variance

    def __getitem__(self, index):
        """
        返回一个样本

        Returns:
            input_2d: (11, 2) - 2D关键点坐标
            input_noise_scale: (11, 2) - 噪声尺度
            target_3d: (11, 3) - 3D模型关键点（用于计算MPJPE）
            quaternion_gt: (4,) - 真实旋转四元数
            translation_gt: (3,) - 真实平移向量
            visibility: (11,) - 可见性标记
            image_id: str - 图像ID
        """
        gmm_file = self.gmm_files[index]

        # 加载GMM数据（JSON格式）
        with open(gmm_file, 'r') as f:
            gmm_data = json.load(f)

        # 从JSON中提取数据
        gmm_params = gmm_data['gmm_parameters']
        visibility_np = np.array(gmm_data['visibility'], dtype=np.float32)

        # 获取图像文件名（优先使用JSON中的image_name，否则从文件名推断）
        if 'image_name' in gmm_data:
            image_filename = gmm_data['image_name']
        else:
            image_filename = os.path.splitext(os.path.basename(gmm_file))[0] + '.jpg'

        # 转换GMM为SPEED+格式并采样
        poses_2d_gmm = self._convert_gmm_to_2d_poses(gmm_params, visibility_np > 0)
        kernel_mean, kernel_variance = self._sample_from_gmm(poses_2d_gmm, visibility_np > 0)

        # 获取真实姿态（从test.json）
        if image_filename in self.test_data:
            gt_data = self.test_data[image_filename]
            # 四元数: test.json中是 [x, y, z, w] 格式，转换为 [w, x, y, z]
            # 键名可能是 'q_vbs2tango_true' 或 'q_vbs2tango'
            q_key = 'q_vbs2tango_true' if 'q_vbs2tango_true' in gt_data else 'q_vbs2tango'
            q_vbs2tango = gt_data[q_key]
            quaternion_gt = np.array([q_vbs2tango[3], q_vbs2tango[0], q_vbs2tango[1], q_vbs2tango[2]], dtype=np.float32)

            # 平移向量
            translation_gt = np.array(gt_data['r_Vo2To_vbs_true'], dtype=np.float32)
        else:
            print(f"Warning: {image_filename} not found in test.json")
            quaternion_gt = np.array([1, 0, 0, 0], dtype=np.float32)
            translation_gt = np.zeros(3, dtype=np.float32)

        # 使用3D模型关键点（中心化）
        target_3d = self.kpts_3d_model.copy()
        target_3d = target_3d - target_3d[0:1, :]

        # 创建可见性掩码
        visibility_mask = np.concatenate([
            np.tile(visibility_np[:, np.newaxis], (1, 2)),  # UV
            np.tile(visibility_np[:, np.newaxis], (1, 3))   # XYZ
        ], axis=1)

        # 转换为Tensor
        input_2d = torch.from_numpy(kernel_mean).float()
        input_noise_scale = torch.from_numpy(kernel_variance).float()
        target_3d = torch.from_numpy(target_3d).float()
        quaternion_gt = torch.from_numpy(quaternion_gt).float()
        translation_gt = torch.from_numpy(translation_gt).float()
        visibility = torch.from_numpy(visibility_np).float()

        return (input_2d, input_noise_scale, target_3d, quaternion_gt,
                translation_gt, visibility, image_filename)

    def __len__(self):
        return len(self.gmm_files)


class LightboxValidator:
    """Lightbox数据集验证器"""

    def __init__(self, args, config, device='cuda'):
        self.args = args
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # GraFormer mask (11个关键点)
        self.src_mask = torch.tensor([[[True] * 11]]).to(self.device)

        # 设置扩散参数
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        # 创建模型
        self._create_models()

    def _create_models(self):
        """创建GCNpose和GCNdiff模型"""
        # SPEED+骨架边（11个关键点）
        edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 7], [1, 4], [2, 5], [3, 6],
            [1, 9], [2, 10],
            [3, 8], [6, 8]
        ], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)

        # 创建GCNpose模型（2D->3D）
        pose_config = self.config.copy()
        pose_config.model.coords_dim = [2, 3]
        if not hasattr(pose_config.model, 'use_visibility_embedding'):
            pose_config.model.use_visibility_embedding = True

        self.model_pose = GCNpose(adj.to(self.device), pose_config).to(self.device)
        self.model_pose = torch.nn.DataParallel(self.model_pose)

        # 创建GCNdiff模型（扩散模型）
        diff_config = self.config.copy()
        diff_config.model.coords_dim = [5, 5]
        if not hasattr(diff_config.model, 'use_visibility_embedding'):
            diff_config.model.use_visibility_embedding = True

        self.model_diff = GCNdiff(adj.to(self.device), diff_config).to(self.device)
        self.model_diff = torch.nn.DataParallel(self.model_diff)

        print("Models created successfully")

    def load_checkpoint(self, pose_ckpt_path, diff_ckpt_path):
        """加载训练好的模型权重"""
        if pose_ckpt_path and os.path.exists(pose_ckpt_path):
            states = torch.load(pose_ckpt_path, map_location=self.device)
            self.model_pose.load_state_dict(states[0])
            logging.info(f'Loaded GCNpose from: {pose_ckpt_path}')
        else:
            logging.warning(f'GCNpose checkpoint not found: {pose_ckpt_path}')

        if diff_ckpt_path and os.path.exists(diff_ckpt_path):
            states = torch.load(diff_ckpt_path, map_location=self.device)
            self.model_diff.load_state_dict(states[0])
            logging.info(f'Loaded GCNdiff from: {diff_ckpt_path}')
        else:
            logging.warning(f'GCNdiff checkpoint not found: {diff_ckpt_path}')

    def generalized_steps_with_visibility(self, x, src_mask, seq, model, b, visibility, eta=0.0):
        """扩散去噪过程（DDIM采样）"""
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(self.device)
                next_t = (torch.ones(n) * j).to(self.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1]

                et = model(xt, src_mask, t.float(), 0, visibility)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()

                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next)

        return xs

    def pose_3d_to_quaternion_translation_pnp(self, pose_2d, pose_3d_pred, pose_3d_model,
                                               camera_matrix, visibility_mask):
        """
        使用PnP算法从2D-3D对应关系估计旋转四元数和平移向量

        Args:
            pose_2d: (11, 2) - 2D关键点坐标
            pose_3d_pred: (11, 3) - 预测的3D关键点（中心化的）
            pose_3d_model: (11, 3) - 3D模型关键点（未中心化的原始坐标）
            camera_matrix: (3, 3) - 相机内参矩阵
            visibility_mask: (11,) - 可见性标记

        Returns:
            quaternion: (4,) - 旋转四元数 [w, x, y, z]
            translation: (3,) - 平移向量
        """
        # 筛选可见的关键点
        visible_indices = visibility_mask > 0.5

        if visible_indices.sum() < 4:
            # 如果可见点少于4个，无法使用PnP，回退到SVD方法
            centroid = pose_3d_pred.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(pose_3d_pred - centroid, full_matrices=False)
                rotation_matrix = Vt.T
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, -1] *= -1
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()
                quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
            except:
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            translation = centroid
            return quaternion, translation

        # 提取可见的2D和3D点
        points_2d = pose_2d[visible_indices].astype(np.float64)
        points_3d = pose_3d_model[visible_indices].astype(np.float64)

        # 确保相机矩阵是float64
        camera_matrix = camera_matrix.astype(np.float64)

        # 使用cv2.solvePnP求解姿态
        try:
            # 使用RANSAC提高鲁棒性
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d,
                points_2d,
                camera_matrix,
                None,  # 无畸变
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                confidence=0.99
            )

            if not success or inliers is None or len(inliers) < 4:
                # RANSAC失败，尝试普通PnP
                success, rvec, tvec = cv2.solvePnP(
                    points_3d,
                    points_2d,
                    camera_matrix,
                    None,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if success:
                # 将旋转向量转换为旋转矩阵
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # 转换为四元数
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()  # [x, y, z, w]
                quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # [w, x, y, z]

                # 平移向量
                translation = tvec.flatten()

                return quaternion, translation
            else:
                # PnP失败，回退到SVD方法
                raise ValueError("PnP failed")

        except Exception as e:
            # 出错时回退到SVD方法
            centroid = pose_3d_pred.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(pose_3d_pred - centroid, full_matrices=False)
                rotation_matrix = Vt.T
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, -1] *= -1
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()
                quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
            except:
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            translation = centroid
            return quaternion, translation

    def validate(self, dataset, output_path=None, test_times=1, test_timesteps=2,
                 test_num_diffusion_timesteps=24, skip_type='uniform', eta=0.0):
        """在Lightbox数据集上进行验证"""

        data_loader = data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

        # 设置扩散步骤
        if skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError(f"Unknown skip_type: {skip_type}")

        self.model_pose.eval()
        self.model_diff.eval()
        torch.set_grad_enabled(False)

        results = []
        mpjpe_all, p_mpjpe_all = [], []
        rotation_errors, translation_errors = [], []

        print(f"\nValidating on {len(dataset)} images from Lightbox dataset...")

        for i, batch_data in enumerate(tqdm(data_loader, desc="Validating")):
            input_2d, input_noise_scale, targets_3d, quaternion_gt, translation_gt, \
                visibility, image_id = batch_data

            # 移到设备
            input_2d = input_2d.to(self.device)
            input_noise_scale = input_noise_scale.to(self.device)
            targets_3d = targets_3d.to(self.device)
            quaternion_gt = quaternion_gt.to(self.device)
            translation_gt = translation_gt.to(self.device)
            visibility = visibility.to(self.device)

            # Step 1: GCNpose (2D -> 3D)
            inputs_xyz = self.model_pose(input_2d, self.src_mask, visibility)
            inputs_xyz = inputs_xyz - inputs_xyz[:, :1, :]

            # Step 2: 构建uvxyz
            input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)

            # Step 3: 扩散去噪
            input_uvxyz = input_uvxyz.repeat(test_times, 1, 1)
            visibility_rep = visibility.repeat(test_times, 1)

            x = input_uvxyz.clone()
            output_uvxyz = self.generalized_steps_with_visibility(
                x, self.src_mask, seq, self.model_diff, self.betas,
                visibility=visibility_rep, eta=eta
            )
            output_uvxyz = output_uvxyz[-1]

            # 平均多个样本
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times, -1, 11, 5), 0)
            output_xyz = output_uvxyz[:, :, 2:]

            # 中心化
            output_xyz = output_xyz - output_xyz[:, :1, :]
            targets_3d = targets_3d - targets_3d[:, :1, :]

            # Step 4: 使用PnP算法转换为四元数和平移向量
            output_xyz_np = output_xyz[0].cpu().numpy()
            input_2d_np = input_2d[0].cpu().numpy()
            visibility_mask_np = visibility[0].cpu().numpy()

            # 获取相机内参和原始3D模型
            camera_matrix = dataset.camera_intrinsics
            pose_3d_model = dataset.kpts_3d_model  # 未中心化的原始坐标

            quaternion_pred, translation_pred = self.pose_3d_to_quaternion_translation_pnp(
                input_2d_np,
                output_xyz_np,
                pose_3d_model,
                camera_matrix,
                visibility_mask_np
            )

            # Step 5: 计算误差
            visibility_mask = visibility[0]

            if visibility_mask.sum() > 0:
                # MPJPE
                mpjpe_error = mpjpe(output_xyz, targets_3d, visibility_mask.unsqueeze(0)).item() * 1000.0
                p_mpjpe_error = p_mpjpe(
                    output_xyz.cpu().numpy(),
                    targets_3d.cpu().numpy(),
                    visibility_mask.unsqueeze(0).cpu().numpy()
                ) * 1000.0

                mpjpe_all.append(mpjpe_error)
                p_mpjpe_all.append(p_mpjpe_error)

                # 旋转误差
                quaternion_gt_np = quaternion_gt[0].cpu().numpy()
                q_pred_scipy = [quaternion_pred[1], quaternion_pred[2], quaternion_pred[3], quaternion_pred[0]]
                q_gt_scipy = [quaternion_gt_np[1], quaternion_gt_np[2], quaternion_gt_np[3], quaternion_gt_np[0]]

                try:
                    rot_pred = Rotation.from_quat(q_pred_scipy)
                    rot_gt = Rotation.from_quat(q_gt_scipy)
                    rot_error = (rot_pred.inv() * rot_gt).magnitude()
                    rotation_errors.append(np.degrees(rot_error))
                    rot_error_deg = float(np.degrees(rot_error))
                except:
                    rot_error_deg = -1

                # 平移误差
                translation_gt_np = translation_gt[0].cpu().numpy()
                translation_error = np.linalg.norm(translation_pred - translation_gt_np)
                translation_errors.append(translation_error)
                trans_error_val = float(translation_error)
            else:
                mpjpe_error = p_mpjpe_error = rot_error_deg = trans_error_val = -1

            # 保存结果
            result = {
                'image_id': image_id[0] if isinstance(image_id, (list, tuple)) else str(image_id),
                'predicted_quaternion': quaternion_pred.tolist(),
                'predicted_translation': translation_pred.tolist(),
                'visibility_masked_mpjpe_mm': float(mpjpe_error),
                'visibility_masked_p_mpjpe_mm': float(p_mpjpe_error),
                'rotation_error_deg': rot_error_deg,
                'translation_error': trans_error_val,
                'num_visible_joints': int(visibility_mask.sum().item()),
                'ground_truth_quaternion': quaternion_gt[0].cpu().numpy().tolist(),
                'ground_truth_translation': translation_gt[0].cpu().numpy().tolist()
            }
            results.append(result)

            # 每50张打印进度
            if (i + 1) % 50 == 0 and mpjpe_all:
                print(f"\nProcessed {i+1}/{len(dataset)} images")
                print(f"  Avg MPJPE: {np.mean(mpjpe_all):.2f}mm")
                if rotation_errors:
                    print(f"  Avg Rotation Error: {np.mean(rotation_errors):.2f}°")
                if translation_errors:
                    print(f"  Avg Translation Error: {np.mean(translation_errors):.4f}")

        # 计算总体统计
        summary = {
            'total_images': len(dataset),
            'average_mpjpe_mm': float(np.mean(mpjpe_all)) if mpjpe_all else -1,
            'average_p_mpjpe_mm': float(np.mean(p_mpjpe_all)) if p_mpjpe_all else -1,
            'average_rotation_error_deg': float(np.mean(rotation_errors)) if rotation_errors else -1,
            'average_translation_error': float(np.mean(translation_errors)) if translation_errors else -1,
            'std_mpjpe_mm': float(np.std(mpjpe_all)) if mpjpe_all else -1,
            'std_rotation_error_deg': float(np.std(rotation_errors)) if rotation_errors else -1,
            'std_translation_error': float(np.std(translation_errors)) if translation_errors else -1
        }

        output_data = {'summary': summary, 'per_image_results': results}

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {output_path}")

        # 打印总结
        print("\n" + "="*70)
        print("LIGHTBOX VALIDATION SUMMARY")
        print("="*70)
        print(f"Total images: {summary['total_images']}")
        if summary['average_mpjpe_mm'] > 0:
            print(f"Average MPJPE: {summary['average_mpjpe_mm']:.2f} ± {summary['std_mpjpe_mm']:.2f} mm")
            print(f"Average P-MPJPE: {summary['average_p_mpjpe_mm']:.2f} mm")
        if summary['average_rotation_error_deg'] > 0:
            print(f"Average Rotation Error: {summary['average_rotation_error_deg']:.2f} ± {summary['std_rotation_error_deg']:.2f}°")
        if summary['average_translation_error'] > 0:
            print(f"Average Translation Error: {summary['average_translation_error']:.4f} ± {summary['std_translation_error']:.4f}")
        print("="*70)

        return output_data


def main():
    parser = argparse.ArgumentParser(description='Validate on Lightbox dataset (SPEED+ format)')

    # 数据路径
    parser.add_argument('--gmm_dir', type=str, required=True,
                        help='Directory containing GMM .json files')
    parser.add_argument('--test_json', type=str, required=True,
                        help='Path to test.json (ground truth poses)')
    parser.add_argument('--kpts_mat', type=str, required=True,
                        help='Path to kpts.mat (3D model keypoints)')
    parser.add_argument('--camera_json', type=str, required=True,
                        help='Path to camera.json (camera parameters)')

    # 模型路径
    parser.add_argument('--pose_ckpt', type=str, required=True,
                        help='Path to GCNpose checkpoint')
    parser.add_argument('--diff_ckpt', type=str, required=True,
                        help='Path to GCNdiff checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')

    # 输出路径
    parser.add_argument('--output', type=str, default='lightbox_results.json',
                        help='Path to save results')

    # 测试参数
    parser.add_argument('--test_times', type=int, default=1,
                        help='Number of sampling times')
    parser.add_argument('--test_timesteps', type=int, default=2,
                        help='Number of diffusion steps')
    parser.add_argument('--test_num_diffusion_timesteps', type=int, default=24,
                        help='Total diffusion timesteps')
    parser.add_argument('--skip_type', type=str, default='uniform',
                        choices=['uniform', 'quad'])
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # 加载配置
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    class DictObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, DictObj(v) if isinstance(v, dict) else v)
        def copy(self):
            return self

    config = DictObj(config)

    # 创建数据集
    print("Loading Lightbox dataset...")
    dataset = LightboxDataset(
        gmm_dir=args.gmm_dir,
        test_json_path=args.test_json,
        kpts_mat_path=args.kpts_mat,
        camera_json_path=args.camera_json
    )

    # 创建验证器
    print("Creating validator...")
    validator = LightboxValidator(args, config, device=args.device)

    # 加载模型
    print("Loading checkpoints...")
    validator.load_checkpoint(args.pose_ckpt, args.diff_ckpt)

    # 执行验证
    print("Starting validation...")
    validator.validate(
        dataset=dataset,
        output_path=args.output,
        test_times=args.test_times,
        test_timesteps=args.test_timesteps,
        test_num_diffusion_timesteps=args.test_num_diffusion_timesteps,
        skip_type=args.skip_type,
        eta=args.eta
    )

    print("\nValidation completed!")


if __name__ == '__main__':
    main()
