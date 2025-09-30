import os
import logging
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.data_utils import fetch_speedplus
from common.utils_diff import get_beta_schedule, generalized_steps, generalized_steps_with_visibility  # 添加新函数
from common.generators import PoseGenerator_gmm_speedplus
from common.loss import mpjpe, p_mpjpe

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask SPEED+版本
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True]]]).cuda()
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def prepare_data(self):
        """args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # 加载数据集
        if config.data.dataset == "speedplus":
            from common.speedplus_dataset import SpeedPlusDataset
            from common.data_utils import create_2d_data_speedplus_gmm
            
            # 加载SPEED+数据集
            dataset = SpeedPlusDataset(
                json_path=config.data.json_path,
                keypoints_path=config.data.keypoints_path
            )
            
            self.subjects_train = ['spacecraft']
            self.subjects_test = ['spacecraft']
            
            # 处理3D数据
            from common.data_utils import read_3d_data_speedplus
            self.dataset = read_3d_data_speedplus(dataset)
            
            # 创建2D GMM数据（包含零掩码）
            self.keypoints_train = create_2d_data_speedplus_gmm(dataset)
            self.keypoints_test = self.keypoints_train
            
            # SPEED+没有动作分类
            self.action_filter = None
        else:
            raise KeyError('Invalid dataset')
        """
        args, config = self.args, self.config
    
        if config.data.dataset == "speedplus_v2":
            from common.speedplus_dataset_v2 import SpeedPlusV2Dataset
            from common.data_utils_v2 import create_2d_data_speedplus_v2, fetch_speedplus_v2
            
            # 加载SPEED+ V2数据集
            dataset = SpeedPlusV2Dataset(
                train_json_path=config.data.train_json_path,
                kpts_mat_path=config.data.kpts_mat_path,
                gmm_data_path=config.data.gmm_data_path
            )
            
            self.subjects_train = ['spacecraft']
            self.subjects_test = ['spacecraft']
            self.dataset = dataset
            
            keypoints_gmm = create_2d_data_speedplus_v2(dataset)
        
            # 分割训练/测试（理由：需要验证集评估）
            all_actions = list(dataset['spacecraft'].keys())
            split_idx = int(len(all_actions) * 0.8)
            
            train_actions = all_actions[:split_idx]
            test_actions = all_actions[split_idx:]
            
            self.keypoints_train = {
                'spacecraft': {k: keypoints_gmm['spacecraft'][k] for k in train_actions}
            }
            self.keypoints_test = {
                'spacecraft': {k: keypoints_gmm['spacecraft'][k] for k in test_actions}
            }
            
            self.action_filter = None

    # create diffusion model SPEED+版本
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 7], [1, 4], [2, 5], [3, 6],
            [1, 9], [2, 10],
            [3, 8], [6, 8]
        ], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)
        
        # 确保配置中有use_visibility_embedding
        if not hasattr(config.model, 'use_visibility_embedding'):
            config.model.use_visibility_embedding = True
        
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model            
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
            logging.info('Loaded diffusion model from: ' + model_path)
                
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 7], [1, 4], [2, 5], [3, 6],
            [1, 9], [2, 10],
            [3, 8], [6, 8]
        ], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)
        
        if not hasattr(config.model, 'use_visibility_embedding'):
            config.model.use_visibility_embedding = True
        
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0])
        else:
            logging.info('initialize model randomly')
              
    def train(self):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        best_p1, best_epoch = 1000, 0
        stride = self.args.downsample
        
        # 创建数据加载器
        if config.data.dataset == "speedplus":
            poses_train, poses_train_2d, camerapara_train, visibility_train = fetch_speedplus(
                self.subjects_train, self.dataset, self.keypoints_train, stride
            )
            
            train_generator = PoseGenerator_gmm_speedplus(
                poses_train, poses_train_2d, camerapara_train, visibility_train,
                augment_uncertainty=True,  # 训练时启用增强
                augment_prob=0.5,          # 50%的概率应用增强
                uncertainty_scale=50,       # 方差扩大50倍
                num_uncertain_joints=2      # 每次选择2个关键点
            )
            
            data_loader = data.DataLoader(
                train_generator,
                batch_size=config.training.batch_size, shuffle=True,
                num_workers=config.training.num_workers, pin_memory=True
            )
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
    
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            for i, batch_data in enumerate(data_loader):
                data_time += time.time() - data_start
                step += 1

                # 解包数据（包含可见性）
                targets_uvxyz, targets_noise_scale, _, targets_3d, camera_para, \
                    visibility, visibility_mask = batch_data
                
                # 生成噪声样本
                n = targets_3d.size(0)
                x = targets_uvxyz
                e = torch.randn_like(x)
                b = self.betas            
                t = torch.randint(low=0, high=self.num_timesteps,
                                size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                # 应用可见性掩码到噪声
                e = e * targets_noise_scale
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                if i == 0 and epoch == 0:
                    print(f"\n第一个样本的前11个关键点:")
                    print(f"targets_uvxyz[0, :11]:\n{targets_uvxyz[0, :11]}")
                    print(f"  Visibility: {visibility[0].cpu().numpy()}")
                    print(f"  Visible joints: {visibility[0].sum().item()}/11")
                
                # 预测噪声，传入可见性信息
                output_noise = self.model_diff(x, src_mask, t.float(), 0, visibility)
                
                # 计算损失，应用可见性掩码
                loss_diff = ((e - output_noise) * visibility_mask.unsqueeze(-1)).square().sum(dim=(1, 2))
                # 归一化：除以可见关键点数量
                n_visible = visibility_mask.sum(dim=1, keepdim=True).clamp(min=1)
                loss_diff = (loss_diff / n_visible).mean()
                
                # 梯度更新
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma)
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
                logging.info('test the performance of current model')

                # 测试并获取结果
                p1, p2 = self.test_hyber(is_train=True)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch = epoch
                    # 保存最佳模型
                    torch.save(states, os.path.join(self.args.log_path, "best_model.pth"))
                    
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJPE: {:.2f} PA-MPJPE: {:.2f} |'.format(
                    best_epoch, best_p1, epoch, p1, p2
                ))
        
        # 训练结束，返回最佳结果
        logging.info('Training completed. Best MPJPE: {:.2f} at epoch {}'.format(best_p1, best_epoch))
        return best_p1, best_epoch
    
    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, \
            config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "speedplus":
            poses_valid, poses_valid_2d, camerapara_valid, visibility_valid = fetch_speedplus(
                self.subjects_test, self.dataset, self.keypoints_test, stride
            )

            test_generator = PoseGenerator_gmm_speedplus(
                poses_valid, poses_valid_2d, camerapara_valid, visibility_valid,
                augment_uncertainty=False  # 测试时禁用增强
            )

            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm_speedplus(
                    poses_valid, poses_valid_2d, camerapara_valid, visibility_valid
                ),
                batch_size=config.training.batch_size, shuffle=False,
                num_workers=config.training.num_workers, pin_memory=True
            )
        else:
            raise KeyError('Invalid dataset')

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        # 设置扩散步骤
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        action_error_sum = define_error_list(actions=["spacecraft"])

        for i, batch_data in enumerate(data_loader):
            data_time += time.time() - data_start
            
            # 解包数据（包含可见性）
            _, input_noise_scale, input_2d, targets_3d, camera_para, \
                visibility, visibility_mask = batch_data
            input_action = None

            # 使用可见性信息构建初始3D姿态
            inputs_xyz = self.model_pose(input_2d, src_mask, visibility)            
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :] 
            input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)
                        
            # 生成多个样本用于分布估计
            input_uvxyz = input_uvxyz.repeat(test_times, 1, 1)
            input_noise_scale = input_noise_scale.repeat(test_times, 1, 1)
            visibility_rep = visibility.repeat(test_times, 1)
            
            # 设置扩散时间步
            t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device) * test_num_diffusion_timesteps
            
            # 准备扩散参数
            x = input_uvxyz.clone()
            e = torch.randn_like(input_uvxyz)
            b = self.betas   
            e = e * input_noise_scale        
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            # x = x * a.sqrt() + e * (1.0 - a).sqrt()  # 测试时可以不加噪声
            
            # 执行扩散去噪过程（传入可见性信息）
            output_uvxyz = self.generalized_steps_with_visibility(
                x, src_mask, seq, self.model_diff, self.betas, 
                visibility=visibility_rep, eta=self.args.eta
            )
            output_uvxyz = output_uvxyz[0][-1]  # 取最后一步的结果
            
            # 平均多个样本的结果
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times, -1, 11, 5), 0)
            output_xyz = output_uvxyz[:, :, 2:]  # 提取xyz坐标
            
            # 中心化处理
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            
            # 计算误差（考虑可见性）
            if visibility.sum() > 0:  # 确保至少有一个可见点
                # 对所有点计算误差（用于整体评估）
                mpjpe_error = mpjpe(output_xyz, targets_3d).item() * 1000.0
                epoch_loss_3d_pos.update(mpjpe_error, targets_3d.size(0))
                
                # 计算P-MPJPE
                p_mpjpe_error = p_mpjpe(
                    output_xyz.cpu().numpy(), 
                    targets_3d.cpu().numpy()
                ).item() * 1000.0
                epoch_loss_3d_pos_procrustes.update(p_mpjpe_error, targets_3d.size(0))
                
                # 更新动作误差统计
                action_error_sum = test_calculation(
                    output_xyz, targets_3d, input_action, action_error_sum, 
                    data_type="speedplus", subject=None, MAE=False
                )
            
            data_start = time.time()
            
            if i % 100 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'.format(
                    batch=i + 1, size=len(data_loader), data=data_time, 
                    e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg
                ))
        
        # 输出最终结果
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'.format(
            batch=i + 1, size=len(data_loader), data=data_time, 
            e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg
        ))
        
        # 打印误差并返回结果
        p1, p2 = print_error(data_type="speedplus", action_error_sum=action_error_sum, is_train=is_train)

        return p1, p2
    
    def generalized_steps_with_visibility(self, x, src_mask, seq, model, b, visibility, **kwargs):
        """修改的扩散步骤，包含可见性信息"""
        from common.utils_diff import compute_alpha
        
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
            
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).cuda()
                next_t = (torch.ones(n) * j).cuda()
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1]
                
                # 传入可见性信息到模型
                et = model(xt, src_mask, t.float(), 0, visibility)
                
                # DDIM采样步骤
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t)
                
                # 计算方差参数
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                
                # 更新到下一个时间步
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next)

        return xs, x0_preds