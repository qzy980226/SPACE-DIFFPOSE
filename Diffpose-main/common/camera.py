from __future__ import absolute_import, division

import numpy as np
import torch
import cv2

from common.utils import wrap
from common.quaternion import qrot, qinverse

#3D世界坐标与2D图像坐标变换
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # 归一化使得[0, w]映射到[-1, 1]，同时保持纵横比
    return X / w * 2 - [1, h / w]

def denormalize_screen_coordinates(X, w, h):
    """
    将归一化的坐标转换回像素坐标
    """
    assert X.shape[-1] == 2
    return (X + [1, h / w]) * w / 2

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera_speedplus(X, q, t):
    """
    使用四元数表示旋转
    """
    # 四元数转旋转矩阵
    qx, qy, qz, qw = q
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])
    
    # 应用旋转和平移
    if isinstance(X, torch.Tensor):
        R = torch.from_numpy(R).float()
        t = torch.from_numpy(t).float() if isinstance(t, np.ndarray) else t
        return torch.matmul(X, R.T) + t
    else:
        return (R @ X.T).T + t


def camera_to_world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1,)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c

def project_to_2d_speedplus(X, camera_params, use_cv2=True):
    """
    5个畸变参数
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    

    # 使用OpenCV进行畸变投影，更准确地处理5个畸变参数
    batch_size = X.shape[0]
    projected_2d = []
        
    for i in range(batch_size):
        # 反归一化相机参数
        fx = camera_params[i, 0] * 1920 / 2  # 假设原始分辨率
        fy = camera_params[i, 1] * 1920 / 2
        cx = (camera_params[i, 2] + 1) * 1920 / 2
        cy = (camera_params[i, 3] + 1) * 1200 / 2
            
        # 相机矩阵
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
        # 畸变系数
        dist_coeffs = camera_params[i, 4:9].cpu().numpy()  # k1, k2, p1, p2, k3
            
        # 投影
        points_3d = X[i].cpu().numpy() if hasattr(X[i], 'cpu') else X[i]
        points_2d, _ = cv2.projectPoints(
            points_3d,
            np.zeros(3),  # 旋转向量（已在相机坐标系）
            np.zeros(3),  # 平移向量（已在相机坐标系）
            K,
            dist_coeffs
        )
        points_2d = points_2d.reshape(-1, 2)
            
        # 归一化回[-1, 1]范围
        points_2d[:, 0] = points_2d[:, 0] / 1920 * 2 - 1
        points_2d[:, 1] = points_2d[:, 1] / 1200 * 2 - 1200/1920
            
        projected_2d.append(points_2d)
        
    projected_2d = np.stack(projected_2d)
    return torch.from_numpy(projected_2d).float() if isinstance(X, torch.Tensor) else projected_2d
