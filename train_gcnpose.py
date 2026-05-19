"""
训练GCNpose作为预训练模型
用于为扩散模型提供初始3D姿态估计能力
"""

import argparse
import logging
import yaml
import os
import sys
import time
import json
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

from models.gcnpose import GCNpose, adj_mx_from_edges
from common.utils import *
from common.loss import mpjpe, p_mpjpe
from common.generators import PoseGenerator_gmm_speedplus


class GCNposeTrainer(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # GraFormer mask for SPEED+ (11 keypoints)
        self.src_mask = torch.tensor([[[True] * 11]]).to(device)

    def prepare_data(self):
        """准备SPEED+数据集"""
        args, config = self.args, self.config

        if config.data.dataset == "speedplus_v2":
            from common.speedplus_dataset_v2 import SpeedPlusV2Dataset
            from common.data_utils_v2 import create_2d_data_speedplus_v2, fetch_speedplus_v2

            # 加载数据集
            dataset = SpeedPlusV2Dataset(
                train_json_path=config.data.train_json_path,
                kpts_mat_path=config.data.kpts_mat_path,
                gmm_data_path=config.data.gmm_data_path
            )

            self.subjects_train = ['spacecraft']
            self.subjects_test = ['spacecraft']
            self.dataset = dataset

            keypoints_gmm = create_2d_data_speedplus_v2(dataset)

            # 分割训练/测试
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

            logging.info(f"Loaded {len(train_actions)} training samples, {len(test_actions)} test samples")
        else:
            raise KeyError(f'Invalid dataset: {config.data.dataset}')

    def create_model(self, model_path=None):
        """创建GCNpose模型"""
        config = self.config

        # SPEED+ 航天器关键点边结构
        edges = torch.tensor([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 7], [1, 4], [2, 5], [3, 6],
            [1, 9], [2, 10],
            [3, 8], [6, 8]
        ], dtype=torch.long)

        adj = adj_mx_from_edges(num_pts=11, edges=edges, sparse=False)

        # 设置输入输出维度
        config.model.coords_dim = [2, 3]  # 输入2D, 输出3D

        # 确保可见性嵌入开启
        if not hasattr(config.model, 'use_visibility_embedding'):
            config.model.use_visibility_embedding = True

        # 创建模型
        self.model = GCNpose(adj.to(self.device), config).to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # 加载预训练权重（如果提供）
        if model_path and os.path.exists(model_path):
            logging.info(f'Loading pretrained model from: {model_path}')
            states = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(states[0])
        else:
            logging.info('Initializing model randomly')

    def train(self):
        """训练GCNpose模型"""
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        stride = args.downsample

        # 准备数据加载器
        from common.data_utils_v2 import fetch_speedplus_v2

        poses_train, poses_train_2d, camerapara_train, visibility_train = fetch_speedplus_v2(
            self.subjects_train, self.dataset, self.keypoints_train, stride
        )

        # 训练数据加载器（不使用数据增强）
        train_generator = PoseGenerator_gmm_speedplus(
            poses_train, poses_train_2d, camerapara_train, visibility_train,
            augment_uncertainty=False  # 纯监督训练
        )

        train_loader = data.DataLoader(
            train_generator,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True
        )

        # 优化器
        optimizer = get_optimizer(config, self.model.parameters())

        # 训练参数
        best_mpjpe = 1000.0
        best_epoch = 0
        lr_init = config.optim.lr
        decay = config.optim.decay
        gamma = config.optim.lr_gamma

        # 初始化训练历史记录
        training_history = {
            "config": {
                "batch_size": config.training.batch_size,
                "learning_rate": lr_init,
                "n_epochs": config.training.n_epochs,
                "optimizer": config.optim.optimizer,
                "lr_decay": decay,
                "lr_gamma": gamma
            },
            "epochs": []
        }

        logging.info(f"Starting GCNpose training for {config.training.n_epochs} epochs")
        logging.info(f"Batch size: {config.training.batch_size}, Learning rate: {lr_init}")

        for epoch in range(config.training.n_epochs):
            epoch_start = time.time()

            # 训练模式
            torch.set_grad_enabled(True)
            self.model.train()

            epoch_loss_3d = AverageMeter()

            for i, batch_data in enumerate(train_loader):
                # 解包数据
                _, _, input_2d, target_3d, camera_para, visibility, visibility_mask = batch_data

                input_2d = input_2d.to(self.device)
                target_3d = target_3d.to(self.device)
                visibility = visibility.to(self.device)
                visibility_mask = visibility_mask.to(self.device)

                # 前向传播
                pred_3d = self.model(input_2d, src_mask, visibility)

                # 中心化（以第一个关键点为原点）
                pred_3d = pred_3d - pred_3d[:, :1, :]
                target_3d = target_3d - target_3d[:, :1, :]

                # 提取3D坐标的可见性掩码 (只取XYZ部分，不要UV部分)
                # visibility_mask.shape = (batch, 11, 5) -> 只取后3列
                visibility_mask_3d = visibility_mask[:, :, 2:5]  # (batch, 11, 3)

                # 计算损失（应用可见性掩码）
                loss = ((pred_3d - target_3d) * visibility_mask_3d).square().sum(dim=(1, 2))
                n_visible = visibility_mask_3d.sum(dim=(1, 2)).clamp(min=1)
                loss = (loss / n_visible).mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.optim.grad_clip)
                optimizer.step()

                # 记录损失
                epoch_loss_3d.update(loss.item() * 1000.0, target_3d.size(0))

                # 打印训练进度
                if i % 10 == 0 and i != 0:
                    logging.info(f'Epoch [{epoch+1}/{config.training.n_epochs}] '
                               f'Iter [{i+1}/{len(train_loader)}] '
                               f'Loss: {epoch_loss_3d.avg:.2f} mm')

            # 学习率衰减
            if epoch % decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma)
                logging.info(f'Learning rate decayed to: {lr_now}')

            # 每个epoch结束后验证
            logging.info(f'Epoch {epoch+1} completed in {time.time() - epoch_start:.2f}s')
            logging.info(f'Training Loss: {epoch_loss_3d.avg:.2f} mm')

            # 测试性能
            test_mpjpe, test_pmpjpe = self.test()

            # 记录当前epoch的训练历史
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss_mm": round(epoch_loss_3d.avg, 2),
                "val_mpjpe_mm": round(test_mpjpe, 2),
                "val_p_mpjpe_mm": round(test_pmpjpe, 2),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time_sec": round(time.time() - epoch_start, 2),
                "is_best": False
            }

            # 保存模型
            states = [
                self.model.state_dict(),
                optimizer.state_dict(),
                epoch,
            ]

            torch.save(states, os.path.join(args.log_path, f"gcnpose_epoch_{epoch}.pth"))
            torch.save(states, os.path.join(args.log_path, "gcnpose_latest.pth"))

            # 保存最佳模型
            if test_mpjpe < best_mpjpe:
                best_mpjpe = test_mpjpe
                best_epoch = epoch
                epoch_record["is_best"] = True
                torch.save(states, os.path.join(args.log_path, "gcnpose_best.pth"))
                logging.info(f'*** New best model saved! MPJPE: {best_mpjpe:.2f} mm ***')

            # 添加到训练历史
            training_history["epochs"].append(epoch_record)
            training_history["best_epoch"] = best_epoch + 1
            training_history["best_mpjpe_mm"] = round(best_mpjpe, 2)

            # 保存训练历史到JSON文件
            history_path = os.path.join(args.log_path, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)

            logging.info(f'Best: Epoch {best_epoch}, MPJPE: {best_mpjpe:.2f} mm')
            logging.info('-' * 80)

        logging.info(f'Training completed! Best MPJPE: {best_mpjpe:.2f} mm at epoch {best_epoch}')
        return best_mpjpe, best_epoch

    def test(self):
        """测试GCNpose性能"""
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        stride = args.downsample

        # 准备测试数据
        from common.data_utils_v2 import fetch_speedplus_v2

        poses_test, poses_test_2d, camerapara_test, visibility_test = fetch_speedplus_v2(
            self.subjects_test, self.dataset, self.keypoints_test, stride
        )

        test_generator = PoseGenerator_gmm_speedplus(
            poses_test, poses_test_2d, camerapara_test, visibility_test,
            augment_uncertainty=False
        )

        test_loader = data.DataLoader(
            test_generator,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True
        )

        # 测试模式
        torch.set_grad_enabled(False)
        self.model.eval()

        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        action_error_sum = define_error_list(actions=["spacecraft"])

        for i, batch_data in enumerate(test_loader):
            # 解包数据
            _, _, input_2d, target_3d, camera_para, visibility, visibility_mask = batch_data

            input_2d = input_2d.to(self.device)
            target_3d = target_3d.to(self.device)
            visibility = visibility.to(self.device)

            # 预测
            pred_3d = self.model(input_2d, src_mask, visibility)

            # 中心化
            pred_3d = pred_3d - pred_3d[:, :1, :]
            target_3d = target_3d - target_3d[:, :1, :]

            # 准备可见性掩码 (只使用关节级别的可见性)
            # visibility.shape = (batch, 11)
            visibility_mask_joints = visibility  # (batch, 11)

            # 计算MPJPE（使用可见性掩码）
            mpjpe_error = mpjpe(pred_3d, target_3d, visibility_mask_joints).item() * 1000.0
            epoch_loss_3d_pos.update(mpjpe_error, target_3d.size(0))

            # 计算P-MPJPE（使用可见性掩码）
            p_mpjpe_error = p_mpjpe(
                pred_3d.cpu().numpy(),
                target_3d.cpu().numpy(),
                visibility_mask_joints.cpu().numpy()
            ) * 1000.0
            epoch_loss_3d_pos_procrustes.update(p_mpjpe_error, target_3d.size(0))

            # 更新错误统计
            action_error_sum = test_calculation(
                pred_3d, target_3d, None, action_error_sum,
                data_type="speedplus", subject=None, MAE=False
            )

        logging.info(f'Test Results: MPJPE: {epoch_loss_3d_pos.avg:.2f} mm, '
                    f'P-MPJPE: {epoch_loss_3d_pos_procrustes.avg:.2f} mm')

        # 打印详细错误
        p1, p2 = print_error(data_type="speedplus", action_error_sum=action_error_sum, is_train=True)

        return p1, p2


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train GCNpose for pretraining")

    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--exp", type=str, default="exp",
                       help="Path for saving outputs")
    parser.add_argument("--doc", type=str, required=True,
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=19960903,
                       help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--downsample", type=int, default=1,
                       help="Downsample factor")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Pretrained model path (optional)")
    parser.add_argument("--verbose", type=str, default="info",
                       help="Logging level")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # 加载配置文件
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)

    # 转换为namespace
    config = dict2namespace(config)

    # 覆盖配置
    config.training.batch_size = args.batch_size
    config.optim.lr = args.lr
    if args.n_epochs:
        config.training.n_epochs = args.n_epochs

    # 创建日志目录
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # 保存配置
    with open(os.path.join(args.log_path, "config.yml"), "w") as f:
        yaml.dump(namespace2dict(config), f, default_flow_style=False)

    # 设置日志
    level = getattr(logging, args.verbose.upper(), None)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "train.log"))
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(namespace):
    """将namespace转换回dict（用于保存）"""
    if isinstance(namespace, argparse.Namespace):
        return {k: namespace2dict(v) for k, v in vars(namespace).items()}
    else:
        return namespace


def main():
    args, config = parse_args_and_config()

    logging.info("=" * 80)
    logging.info("Training GCNpose as Pretrained Model for DiffPose")
    logging.info("=" * 80)
    logging.info(f"Experiment: {args.doc}")
    logging.info(f"Log path: {args.log_path}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Batch size: {config.training.batch_size}")
    logging.info(f"Learning rate: {config.optim.lr}")
    logging.info(f"Epochs: {config.training.n_epochs}")
    logging.info("=" * 80)

    try:
        # 创建训练器
        trainer = GCNposeTrainer(args, config)

        # 准备数据
        logging.info("Preparing dataset...")
        trainer.prepare_data()

        # 创建模型
        logging.info("Creating model...")
        trainer.create_model(args.model_path)

        # 训练
        logging.info("Starting training...")
        best_mpjpe, best_epoch = trainer.train()

        logging.info("=" * 80)
        logging.info(f"Training completed successfully!")
        logging.info(f"Best model: Epoch {best_epoch}, MPJPE {best_mpjpe:.2f} mm")
        logging.info(f"Saved to: {args.log_path}/gcnpose_best.pth")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
