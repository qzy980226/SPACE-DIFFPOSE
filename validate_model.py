"""
完整的模型验证脚本
用于在SPEED+ V2数据集上验证训练好的DiffPose模型
"""
import argparse
import logging
import yaml
import os
import sys
import torch
import numpy as np

from runners.diffpose_frame import Diffpose


def dict2namespace(config):
    """将字典转换为命名空间"""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def setup_logging(verbose='info'):
    """设置日志"""
    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"level {verbose} not supported")

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)


def parse_validation_args():
    """解析验证参数"""
    parser = argparse.ArgumentParser(description="DiffPose Model Validation")

    # 基础配置
    parser.add_argument("--config", type=str,
                        default="speedplus_v2_diffpose.yml",
                        help="配置文件名（位于configs目录）")
    parser.add_argument("--checkpoint", type=str,
                        default="exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth",
                        help="模型检查点路径")
    parser.add_argument("--pose_model", type=str,
                        default=None,
                        help="GCNpose预训练模型路径（可选）")

    # 扩散模型参数
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="采样跳跃类型: uniform 或 quad")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM采样的eta参数，控制随机性")
    parser.add_argument("--test_times", type=int, default=1,
                        help="每个样本的测试次数（用于多次采样平均）")
    parser.add_argument("--test_timesteps", type=int, default=2,
                        help="扩散采样的时间步数")
    parser.add_argument("--test_num_diffusion_timesteps", type=int, default=24,
                        help="扩散过程的总时间步数")

    # 其他参数
    parser.add_argument("--seed", type=int, default=19960903,
                        help="随机种子")
    parser.add_argument("--downsample", type=int, default=1,
                        help="下采样因子")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="批次大小")
    parser.add_argument("--verbose", type=str, default="info",
                        help="日志级别: info | debug | warning | critical")

    return parser.parse_args()


def main():
    """主验证函数"""
    # 解析参数
    args = parse_validation_args()

    # 设置日志
    setup_logging(args.verbose)
    logging.info("=" * 60)
    logging.info("开始模型验证")
    logging.info("=" * 60)

    # 检查检查点是否存在
    if not os.path.exists(args.checkpoint):
        logging.error(f"检查点文件不存在: {args.checkpoint}")
        logging.error("请确认以下路径之一:")
        logging.error("  - exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth")
        logging.error("  - exp/speedplus_v2_diffpose_uvxyz_gt/ckpt.pth")
        sys.exit(1)

    logging.info(f"使用检查点: {args.checkpoint}")

    # 加载配置文件
    config_path = os.path.join("configs", args.config)
    if not os.path.exists(config_path):
        logging.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # 转换为命名空间
    config = dict2namespace(config_dict)

    # 覆盖测试参数
    config.testing.test_times = args.test_times
    config.testing.test_timesteps = args.test_timesteps
    config.testing.test_num_diffusion_timesteps = args.test_num_diffusion_timesteps

    logging.info(f"配置信息:")
    logging.info(f"  - 数据集: {config.data.dataset}")
    logging.info(f"  - 测试次数: {config.testing.test_times}")
    logging.info(f"  - 扩散步数: {config.testing.test_timesteps}")
    logging.info(f"  - 总时间步: {config.testing.test_num_diffusion_timesteps}")
    logging.info(f"  - 批次大小: {args.batch_size}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 检测设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"使用设备: {device}")
    config.device = device

    # 创建虚拟args（用于兼容Diffpose类）
    runner_args = argparse.Namespace()
    runner_args.skip_type = args.skip_type
    runner_args.eta = args.eta
    runner_args.downsample = args.downsample
    runner_args.log_path = os.path.dirname(args.checkpoint)
    runner_args.seed = args.seed

    try:
        logging.info("-" * 60)
        logging.info("初始化DiffPose运行器")
        logging.info("-" * 60)

        # 创建DiffPose运行器
        runner = Diffpose(runner_args, config, device=device)

        logging.info("创建扩散模型...")
        runner.create_diffusion_model(args.checkpoint)

        logging.info("创建姿态估计模型...")
        runner.create_pose_model(args.pose_model)

        logging.info("准备数据集...")
        runner.prepare_data()

        logging.info("-" * 60)
        logging.info("开始验证...")
        logging.info("-" * 60)

        # 执行验证（使用test_hyber函数）
        mpjpe, p_mpjpe = runner.test_hyber(is_train=False)

        logging.info("=" * 60)
        logging.info("验证完成!")
        logging.info("=" * 60)
        logging.info(f"最终结果:")
        logging.info(f"  MPJPE (Mean Per Joint Position Error): {mpjpe:.2f} mm")
        logging.info(f"  P-MPJPE (Procrustes Aligned MPJPE):    {p_mpjpe:.2f} mm")
        logging.info("=" * 60)

        # 保存结果到文件
        result_file = os.path.join(os.path.dirname(args.checkpoint), "validation_results.txt")
        with open(result_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DiffPose模型验证结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"检查点: {args.checkpoint}\n")
            f.write(f"配置文件: {args.config}\n")
            f.write(f"测试次数: {args.test_times}\n")
            f.write(f"扩散步数: {args.test_timesteps}\n")
            f.write(f"总时间步: {args.test_num_diffusion_timesteps}\n")
            f.write("-" * 60 + "\n")
            f.write(f"MPJPE:   {mpjpe:.2f} mm\n")
            f.write(f"P-MPJPE: {p_mpjpe:.2f} mm\n")
            f.write("=" * 60 + "\n")

        logging.info(f"验证结果已保存至: {result_file}")

        return 0

    except Exception as e:
        logging.error("=" * 60)
        logging.error("验证过程中出现错误!")
        logging.error("=" * 60)
        logging.error(f"错误信息: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
