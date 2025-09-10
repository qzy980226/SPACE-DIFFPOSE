import torch
import numpy as np

# ==================== 原有函数（保持不变） ====================
def mpjpe(predicted, target):
    """
    Mean per-joint position error (保持不变)
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (保持不变)
    """
    assert predicted.shape == target.shape
    
    muX = predicted.mean(1, keepdim=True)
    muY = target.mean(1, keepdim=True)
    
    X0 = predicted - muX
    Y0 = target - muY

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdim=True))
    
    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(-2, -1), Y0)
    U, s, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(-2, -1))

    # Avoid improper rotations (reflections)
    sign_detR = torch.sign(torch.det(R))
    V[:, :, -1] *= sign_detR.unsqueeze(-1)
    s[:, -1] *= sign_detR.flatten()
    R = torch.matmul(V, U.transpose(-2, -1))

    tr = torch.sum(s, dim=1, keepdim=True).unsqueeze(-1)

    a = tr * normX / normY  # Scale
    t = muX - a * torch.matmul(muY, R)  # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a * torch.matmul(predicted, R) + t
    
    # Return MPJPE
    return torch.mean(torch.norm(predicted_aligned - target, dim=len(target.shape)-1))

# ==================== 新增函数（支持可见性mask） ====================
def mpjpe_masked(predicted, target, visibility_mask):
    """
    Mean per-joint position error with visibility masking
    
    修改原因：需要只对可见关键点计算MPJPE
    作用：避免不可见关键点的噪声干扰损失计算
    
    参数:
    predicted: [batch_size, num_joints, 3] 预测的3D关键点
    target: [batch_size, num_joints, 3] 目标3D关键点
    visibility_mask: [batch_size, num_joints] 或 [batch_size, 1, num_joints] 可见性mask
    
    返回:
    torch.Tensor: 标量，仅可见关键点的平均误差
    """
    assert predicted.shape == target.shape
    
    # 确保mask的维度正确
    if visibility_mask.dim() == 3:
        visibility_mask = visibility_mask.squeeze(1)  # [batch_size, num_joints]
    
    # 计算每个关键点的欧几里得距离
    joint_errors = torch.norm(predicted - target, dim=2)  # [batch_size, num_joints]
    
    # 只计算可见关键点的误差
    masked_errors = joint_errors * visibility_mask.float()
    
    # 计算可见关键点的数量
    visible_joints = torch.sum(visibility_mask.float(), dim=1)  # [batch_size]
    
    # 避免除零：如果没有可见关键点，设为1
    visible_joints = torch.clamp(visible_joints, min=1.0)
    
    # 计算每个样本的平均误差
    sample_errors = torch.sum(masked_errors, dim=1) / visible_joints  # [batch_size]
    
    # 返回批次平均误差
    return torch.mean(sample_errors)

def p_mpjpe_masked(predicted, target, visibility_mask):
    """
    Procrustes-aligned MPJPE with visibility masking
    
    修改原因：刚体对齐时应只考虑可见关键点
    作用：提高对齐精度，避免不可见关键点影响对齐结果
    
    参数:
    predicted: [batch_size, num_joints, 3] 预测的3D关键点
    target: [batch_size, num_joints, 3] 目标3D关键点
    visibility_mask: [batch_size, num_joints] 可见性mask
    
    返回:
    torch.Tensor: 标量，对齐后仅可见关键点的平均误差
    """
    assert predicted.shape == target.shape
    
    # 确保mask的维度正确
    if visibility_mask.dim() == 3:
        visibility_mask = visibility_mask.squeeze(1)  # [batch_size, num_joints]
    
    batch_size = predicted.shape[0]
    batch_errors = []
    
    for b in range(batch_size):
        pred_b = predicted[b]  # [num_joints, 3]
        target_b = target[b]   # [num_joints, 3]
        mask_b = visibility_mask[b]  # [num_joints]
        
        # 提取可见关键点
        visible_indices = mask_b.bool()
        visible_count = torch.sum(visible_indices)
        
        if visible_count < 3:
            # 如果可见关键点少于3个，无法进行刚体对齐，使用普通MPJPE
            error = torch.mean(torch.norm(pred_b[visible_indices] - target_b[visible_indices], dim=1))
        else:
            # 提取可见关键点
            pred_visible = pred_b[visible_indices]    # [visible_count, 3]
            target_visible = target_b[visible_indices]  # [visible_count, 3]
            
            # 执行Procrustes对齐
            pred_aligned = procrustes_align(pred_visible.unsqueeze(0), target_visible.unsqueeze(0))
            pred_aligned = pred_aligned.squeeze(0)  # [visible_count, 3]
            
            # 计算对齐后的误差
            error = torch.mean(torch.norm(pred_aligned - target_visible, dim=1))
        
        batch_errors.append(error)
    
    return torch.mean(torch.stack(batch_errors))

def procrustes_align(X, Y):
    """
    执行Procrustes对齐
    
    修改原因：p_mpjpe_masked需要独立的对齐函数
    作用：对3D点集进行刚体对齐（尺度+旋转+平移）
    """
    muX = X.mean(1, keepdim=True)
    muY = Y.mean(1, keepdim=True)
    
    X0 = X - muX
    Y0 = Y - muY

    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdim=True))
    
    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(-2, -1), Y0)
    U, s, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(-2, -1))

    # 避免不正当的旋转（反射）
    sign_detR = torch.sign(torch.det(R))
    V[:, :, -1] *= sign_detR.unsqueeze(-1)
    s[:, -1] *= sign_detR.flatten()
    R = torch.matmul(V, U.transpose(-2, -1))  # 旋转矩阵

    tr = torch.sum(s, dim=1, keepdim=True).unsqueeze(-1)

    a = tr * normX / normY  # 缩放
    t = muX - a * torch.matmul(muY, R)  # 平移
    
    # 对输入进行刚体变换
    X_aligned = a * torch.matmul(X, R) + t
    
    return X_aligned

def compute_combined_loss(predicted, target, visibility_mask, 
                         mpjpe_weight=1.0, p_mpjpe_weight=0.5,
                         use_strict_masking=True, weight_invisible=0.1):
    """
    计算组合损失（MPJPE + P-MPJPE）
    
    修改原因：提供灵活的损失计算策略
    作用：支持多种mask策略和损失组合
    
    参数:
    predicted: [batch_size, num_joints, 3] 预测的3D关键点
    target: [batch_size, num_joints, 3] 目标3D关键点
    visibility_mask: [batch_size, num_joints] 可见性mask
    mpjpe_weight: MPJPE损失权重
    p_mpjpe_weight: P-MPJPE损失权重
    use_strict_masking: 是否使用严格mask（True为完全忽略，False为加权）
    weight_invisible: 不可见关键点权重（仅当use_strict_masking=False时使用）
    
    返回:
    dict: 包含各项损失的字典
    """
    losses = {}
    
    if use_strict_masking:
        # 严格mask：完全忽略不可见关键点
        if mpjpe_weight > 0:
            losses['mpjpe'] = mpjpe_masked(predicted, target, visibility_mask)
        if p_mpjpe_weight > 0:
            losses['p_mpjpe'] = p_mpjpe_masked(predicted, target, visibility_mask)
    else:
        # 加权损失：对不可见关键点给予较小权重
        if mpjpe_weight > 0:
            losses['mpjpe'] = compute_visibility_weighted_loss(
                predicted, target, visibility_mask, weight_invisible
            )
        if p_mpjpe_weight > 0:
            # P-MPJPE的加权版本较复杂，这里简化为使用mask版本
            losses['p_mpjpe'] = p_mpjpe_masked(predicted, target, visibility_mask)
    
    # 计算总损失
    total_loss = 0
    if 'mpjpe' in losses:
        total_loss += mpjpe_weight * losses['mpjpe']
    if 'p_mpjpe' in losses:
        total_loss += p_mpjpe_weight * losses['p_mpjpe']
    
    losses['total'] = total_loss
    
    return losses

def compute_visibility_weighted_loss(predicted, target, visibility_mask, weight_invisible=0.1):
    """
    计算可见性加权的损失
    
    修改原因：提供软mask策略作为严格mask的替代
    作用：对不可见关键点给予较小权重而不是完全忽略
    """
    assert predicted.shape == target.shape
    
    if visibility_mask.dim() == 3:
        visibility_mask = visibility_mask.squeeze(1)
    
    # 创建权重矩阵
    weights = visibility_mask.float() + (1 - visibility_mask.float()) * weight_invisible
    
    # 计算每个关键点的误差
    joint_errors = torch.norm(predicted - target, dim=2)  # [batch_size, num_joints]
    
    # 应用权重
    weighted_errors = joint_errors * weights
    
    # 计算平均权重误差
    total_weight = torch.sum(weights, dim=1)  # [batch_size]
    total_weight = torch.clamp(total_weight, min=1e-6)  # 避免除零
    
    sample_errors = torch.sum(weighted_errors, dim=1) / total_weight
    
    return torch.mean(sample_errors)

def compute_loss_statistics(predicted, target, visibility_mask):
    """
    计算各种损失统计信息，用于分析模型性能
    
    修改原因：需要全面的性能分析功能
    作用：比较可见和全体关键点的性能，提供详细统计
    """
    stats = {}
    
    # 全体关键点的损失
    stats['mpjpe_all'] = mpjpe(predicted, target)
    stats['p_mpjpe_all'] = p_mpjpe(predicted, target)
    
    # 仅可见关键点的损失
    stats['mpjpe_visible'] = mpjpe_masked(predicted, target, visibility_mask)
    stats['p_mpjpe_visible'] = p_mpjpe_masked(predicted, target, visibility_mask)
    
    # 每个关键点的平均误差
    joint_errors = torch.norm(predicted - target, dim=2)  # [batch_size, num_joints]
    stats['per_joint_error'] = torch.mean(joint_errors, dim=0)  # [num_joints]
    
    # 可见性统计
    if visibility_mask.dim() == 3:
        visibility_mask = visibility_mask.squeeze(1)
    
    visible_count = torch.sum(visibility_mask.float(), dim=1)  # [batch_size]
    stats['avg_visible_joints'] = torch.mean(visible_count)
    stats['min_visible_joints'] = torch.min(visible_count)
    stats['max_visible_joints'] = torch.max(visible_count)
    
    return stats