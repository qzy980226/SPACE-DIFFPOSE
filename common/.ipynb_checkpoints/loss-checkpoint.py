from __future__ import absolute_import, division

import torch
import numpy as np


def mpjpe(predicted, target, visibility_mask=None):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.

    Args:
        predicted: Predicted 3D poses, shape (..., num_joints, 3)
        target: Target 3D poses, shape (..., num_joints, 3)
        visibility_mask: Optional visibility mask, shape (..., num_joints, 3) or (..., num_joints).
                        If provided, only visible joints are included in the error calculation.

    Returns:
        MPJPE in meters (to get mm, multiply by 1000)
    """
    assert predicted.shape == target.shape

    # 计算每个关节的欧氏距离
    errors = torch.norm(predicted - target, dim=len(target.shape) - 1)  # (..., num_joints)

    if visibility_mask is not None:
        # 确保 visibility_mask 在与 predicted 相同的设备上
        if hasattr(visibility_mask, 'device') and visibility_mask.device != predicted.device:
            visibility_mask = visibility_mask.to(predicted.device)

        # 如果可见性掩码是3D的 (batch, num_joints, 3)，取任意一个维度
        if len(visibility_mask.shape) == len(predicted.shape):
            # 取第一个坐标维度的可见性（假设XYZ的可见性一致）
            visibility_mask = visibility_mask[..., 0]  # (..., num_joints)

        # 只计算可见关节的误差
        visible_errors = errors * visibility_mask
        num_visible = visibility_mask.sum()

        if num_visible > 0:
            return visible_errors.sum() / num_visible
        else:
            # 如果没有可见关节，返回0（避免除零）
            return torch.tensor(0.0, device=predicted.device)
    else:
        # 没有可见性掩码，计算所有关节的平均误差
        return torch.mean(errors)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target, visibility_mask=None):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.

    Args:
        predicted: Predicted 3D poses, shape (batch, num_joints, 3)
        target: Target 3D poses, shape (batch, num_joints, 3)
        visibility_mask: Optional visibility mask, shape (batch, num_joints) or (batch, num_joints, 3).
                        If provided, only visible joints are used for alignment and error calculation.

    Returns:
        P-MPJPE in meters (to get mm, multiply by 1000)
    """
    assert predicted.shape == target.shape

    if visibility_mask is not None:
        # 确保是numpy数组
        if not isinstance(visibility_mask, np.ndarray):
            visibility_mask = visibility_mask.cpu().numpy() if hasattr(visibility_mask, 'cpu') else np.array(visibility_mask)

        # 将可见性掩码转换为 (batch, num_joints) 形状
        if len(visibility_mask.shape) == 3:
            # 如果是 (batch, num_joints, 3)，取第一个维度
            visibility_mask = visibility_mask[:, :, 0]

        # 为每个batch样本单独计算P-MPJPE
        errors = []
        for i in range(predicted.shape[0]):
            vis_mask = visibility_mask[i]  # (num_joints,)
            visible_indices = np.where(vis_mask > 0)[0]

            if len(visible_indices) < 3:
                # 如果可见点少于3个，无法进行刚体对齐，跳过或返回0
                errors.append(0.0)
                continue

            # 只使用可见关节进行对齐
            pred_visible = predicted[i, visible_indices, :]  # (num_visible, 3)
            target_visible = target[i, visible_indices, :]   # (num_visible, 3)

            # Procrustes对齐
            muX = np.mean(target_visible, axis=0, keepdims=True)
            muY = np.mean(pred_visible, axis=0, keepdims=True)

            X0 = target_visible - muX
            Y0 = pred_visible - muY

            normX = np.sqrt(np.sum(X0 ** 2))
            normY = np.sqrt(np.sum(Y0 ** 2))

            if normX < 1e-8 or normY < 1e-8:
                errors.append(0.0)
                continue

            X0 /= normX
            Y0 /= normY

            H = np.matmul(X0.T, Y0)
            U, s, Vt = np.linalg.svd(H)
            V = Vt.T
            R = np.matmul(V, U.T)

            # Avoid improper rotations
            if np.linalg.det(R) < 0:
                V[:, -1] *= -1
                R = np.matmul(V, U.T)

            tr = np.sum(s)
            a = tr * normX / normY  # Scale
            t = muX - a * np.matmul(muY, R)  # Translation

            # 对齐预测
            pred_visible_aligned = a * np.matmul(pred_visible, R) + t

            # 计算误差（仅可见关节）
            error = np.mean(np.linalg.norm(pred_visible_aligned - target_visible, axis=1))
            errors.append(error)

        return np.mean(errors)

    else:
        # 原始实现：计算所有关节
        muX = np.mean(target, axis=1, keepdims=True)
        muY = np.mean(predicted, axis=1, keepdims=True)

        X0 = target - muX
        Y0 = predicted - muY

        normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
        normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

        X0 /= normX
        Y0 /= normY

        H = np.matmul(X0.transpose(0, 2, 1), Y0)
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= sign_detR.flatten()
        R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

        tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

        a = tr * normX / normY  # Scale
        t = muX - a * np.matmul(muY, R)  # Translation

        # Perform rigid transformation on the input
        predicted_aligned = a * np.matmul(predicted, R) + t

        # Return MPJPE
        return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


def normalized_2d_keypoint_error(predicted, target):
    """
    Normalized 2D keypoint distance error.
    Computes the mean Euclidean distance between predicted and target 2D keypoints.

    Args:
        predicted: Predicted 2D keypoints, shape (..., num_joints, 2)
        target: Target 2D keypoints, shape (..., num_joints, 2)

    Returns:
        Mean Euclidean distance error across all keypoints
    """
    assert predicted.shape == target.shape
    assert predicted.shape[-1] == 2, "Expected 2D coordinates (last dimension should be 2)"

    # Compute Euclidean distance for each keypoint
    # torch.norm computes L2 norm along the last dimension (x, y coordinates)
    if isinstance(predicted, torch.Tensor):
        distances = torch.norm(predicted - target, dim=-1)
        return torch.mean(distances)
    else:
        distances = np.linalg.norm(predicted - target, axis=-1)
        return np.mean(distances)
