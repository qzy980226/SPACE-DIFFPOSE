"""预处理SPEED+数据，生成训练所需的NPZ文件"""
import numpy as np
import json
import os
from common.speedplus_dataset import SpeedPlusDataset
from common.data_utils import create_2d_data_speedplus_gmm

def prepare_speedplus_data(json_path, keypoints_path, output_dir, train_split=0.8):
    # 加载数据集
    dataset = SpeedPlusDataset(json_path, keypoints_path)
    
    # 生成2D GMM数据
    keypoints_2d_gmm = create_2d_data_speedplus_gmm(dataset)
    
    # 分割训练/测试集
    all_actions = list(dataset['spacecraft'].keys())
    n_train = int(len(all_actions) * train_split)
    
    train_actions = all_actions[:n_train]
    test_actions = all_actions[n_train:]
    
    # 保存为NPZ格式
    train_data = {
        'positions_3d': {},
        'positions_2d': {}
    }
    test_data = {
        'positions_3d': {},
        'positions_2d': {}
    }
    
    # 分割数据
    train_data['positions_3d']['spacecraft'] = {
        action: dataset['spacecraft'][action] 
        for action in train_actions
    }
    test_data['positions_3d']['spacecraft'] = {
        action: dataset['spacecraft'][action] 
        for action in test_actions
    }
    
    # 保存文件
    np.savez_compressed(
        os.path.join(output_dir, 'data_3d_speedplus.npz'),
        positions_3d=dataset._data
    )
    np.savez_compressed(
        os.path.join(output_dir, 'data_2d_speedplus_gmm.npz'),
        positions_2d=keypoints_2d_gmm
    )
    
    print(f"Data prepared: {len(train_actions)} training, {len(test_actions)} testing samples")

if __name__ == "__main__":
    prepare_speedplus_data(
        json_path="./data/speedplus/annotations.json",
        keypoints_path="./data/speedplus/keypoints_3d.xlsx",
        output_dir="./data/",
        train_split=0.8
    )