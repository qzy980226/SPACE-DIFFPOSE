import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


class SpacecraftHeatmapGenerator:
    def __init__(self, keypoints, camera_intrinsics, quaternion):
        """
        初始化热图生成器

        参数:
        keypoints: (11, 3) numpy数组，包含11个关键点的3D坐标
        camera_intrinsics: 3x3相机内参矩阵
        quaternion: 四元数 [w, x, y, z] 或 [x, y, z, w]
        """
        self.keypoints = np.array(keypoints)
        self.camera_intrinsics = np.array(camera_intrinsics)
        self.quaternion = np.array(quaternion)

        # 确保有11个关键点
        assert self.keypoints.shape == (11, 3), "需要11个关键点的3D坐标"

    def apply_rotation(self, points, quaternion):
        """使用四元数旋转点"""
        # scipy使用[x, y, z, w]格式，如果输入是[w, x, y, z]需要转换
        # 这里假设输入是[w, x, y, z]格式
        if len(quaternion) == 4:
            # 转换为scipy格式 [x, y, z, w]
            quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        else:
            quat_scipy = quaternion

        rotation = Rotation.from_quat(quat_scipy)
        return rotation.apply(points)

    def generate_3d_heatmap(self, keypoint_idx, grid_size=50, sigma=1.0, bounds=None):
        """
        为指定的关键点生成3D热图

        参数:
        keypoint_idx: 关键点索引 (0-10)
        grid_size: 3D网格的分辨率
        sigma: 高斯函数的标准差，控制热图的扩散范围
        bounds: 3D空间边界 [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

        返回:
        grid_points: 网格点坐标
        heatmap_values: 对应的热图值
        """
        if bounds is None:
            # 自动计算边界，留出一些余量
            margin = 2.0
            bounds = []
            for i in range(3):
                min_val = self.keypoints[:, i].min() - margin
                max_val = self.keypoints[:, i].max() + margin
                bounds.append([min_val, max_val])

        # 创建3D网格
        x = np.linspace(bounds[0][0], bounds[0][1], grid_size)
        y = np.linspace(bounds[1][0], bounds[1][1], grid_size)
        z = np.linspace(bounds[2][0], bounds[2][1], grid_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 获取目标关键点
        target_point = self.keypoints[keypoint_idx]

        # 计算每个网格点到关键点的距离
        distances = np.sqrt((X - target_point[0]) ** 2 +
                            (Y - target_point[1]) ** 2 +
                            (Z - target_point[2]) ** 2)

        # 使用高斯函数将距离转换为热图值
        heatmap_values = np.exp(-distances ** 2 / (2 * sigma ** 2))

        return (X, Y, Z), heatmap_values

    def visualize_heatmap_slice(self, keypoint_idx, slice_axis='z', slice_idx=None,
                                grid_size=50, sigma=1.0):
        """
        可视化3D热图的2D切片

        参数:
        keypoint_idx: 关键点索引
        slice_axis: 切片轴 ('x', 'y', 或 'z')
        slice_idx: 切片索引，None表示使用中间切片
        """
        (X, Y, Z), heatmap = self.generate_3d_heatmap(keypoint_idx, grid_size, sigma)

        if slice_idx is None:
            slice_idx = grid_size // 2

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if slice_axis == 'z':
            im = ax.imshow(heatmap[:, :, slice_idx].T, cmap='hot', origin='lower',
                           extent=[X.min(), X.max(), Y.min(), Y.max()])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            title = f'Z = {Z[0, 0, slice_idx]:.2f}'
        elif slice_axis == 'y':
            im = ax.imshow(heatmap[:, slice_idx, :].T, cmap='hot', origin='lower',
                           extent=[X.min(), X.max(), Z.min(), Z.max()])
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            title = f'Y = {Y[0, slice_idx, 0]:.2f}'
        else:  # x
            im = ax.imshow(heatmap[slice_idx, :, :].T, cmap='hot', origin='lower',
                           extent=[Y.min(), Y.max(), Z.min(), Z.max()])
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            title = f'X = {X[slice_idx, 0, 0]:.2f}'

        # 标记所有关键点在切片上的投影
        for i, kp in enumerate(self.keypoints):
            if slice_axis == 'z':
                ax.plot(kp[0], kp[1], 'wo' if i == keypoint_idx else 'bo',
                        markersize=10 if i == keypoint_idx else 6)
            elif slice_axis == 'y':
                ax.plot(kp[0], kp[2], 'wo' if i == keypoint_idx else 'bo',
                        markersize=10 if i == keypoint_idx else 6)
            else:
                ax.plot(kp[1], kp[2], 'wo' if i == keypoint_idx else 'bo',
                        markersize=10 if i == keypoint_idx else 6)

        ax.set_title(f'关键点 {keypoint_idx} 的热图切片 ({title})')
        plt.colorbar(im, ax=ax, label='热图值')
        plt.tight_layout()
        plt.show()

    def visualize_3d_isosurface(self, keypoint_idx, threshold=0.5, grid_size=30, sigma=1.0):
        """
        使用Plotly可视化3D热图的等值面

        参数:
        keypoint_idx: 关键点索引
        threshold: 等值面阈值
        """
        (X, Y, Z), heatmap = self.generate_3d_heatmap(keypoint_idx, grid_size, sigma)

        # 创建等值面
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=heatmap.flatten(),
            isomin=threshold,
            isomax=1.0,
            surface_count=3,
            colorscale='Hot',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        # 添加关键点
        fig.add_trace(go.Scatter3d(
            x=self.keypoints[:, 0],
            y=self.keypoints[:, 1],
            z=self.keypoints[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=['red' if i == keypoint_idx else 'blue' for i in range(11)],
            ),
            text=[f'关键点 {i}' for i in range(11)],
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f'关键点 {keypoint_idx} 的3D热图等值面',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        fig.show()

    def generate_all_heatmaps(self, grid_size=50, sigma=1.0):
        """
        为所有11个关键点生成热图

        返回:
        heatmaps: 字典，键为关键点索引，值为(grid_coords, heatmap_values)
        """
        heatmaps = {}

        for i in range(11):
            grid_coords, values = self.generate_3d_heatmap(i, grid_size, sigma)
            heatmaps[i] = (grid_coords, values)

        return heatmaps

    def save_heatmap_data(self, keypoint_idx, filename, grid_size=50, sigma=1.0):
        """
        保存热图数据到文件
        """
        (X, Y, Z), heatmap = self.generate_3d_heatmap(keypoint_idx, grid_size, sigma)

        # 保存为npz格式
        np.savez(filename,
                 X=X, Y=Y, Z=Z,
                 heatmap=heatmap,
                 keypoint=self.keypoints[keypoint_idx],
                 all_keypoints=self.keypoints,
                 camera_intrinsics=self.camera_intrinsics,
                 quaternion=self.quaternion)

        print(f"热图数据已保存到 {filename}")


# 使用示例
if __name__ == "__main__":
    # 模拟数据 - 请替换为您的实际数据

    # 11个关键点的3D坐标
    keypoints = np.array([
        [0, 0, 0],  # 航天器中心
        [1, 0, 0],  # 右侧
        [-1, 0, 0],  # 左侧
        [0, 1, 0],  # 上方
        [0, -1, 0],  # 下方
        [0, 0, 1],  # 前方
        [0, 0, -1],  # 后方
        [0.5, 0.5, 0],  # 右上
        [-0.5, 0.5, 0],  # 左上
        [0.5, -0.5, 0],  # 右下
        [-0.5, -0.5, 0]  # 左下
    ])

    # 相机内参矩阵 (示例)
    camera_intrinsics = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    # 四元数 [w, x, y, z] (示例：无旋转)
    quaternion = [1, 0, 0, 0]

    # 创建热图生成器
    generator = SpacecraftHeatmapGenerator(keypoints, camera_intrinsics, quaternion)

    # 为第一个关键点生成并可视化热图
    print("生成关键点0的热图...")

    # 可视化2D切片
    generator.visualize_heatmap_slice(keypoint_idx=0, slice_axis='z', sigma=0.5)

    # 可视化3D等值面 (需要安装plotly)
    # generator.visualize_3d_isosurface(keypoint_idx=0, threshold=0.3, sigma=0.5)

    # 生成所有关键点的热图
    print("\n生成所有11个关键点的热图...")
    all_heatmaps = generator.generate_all_heatmaps(grid_size=30, sigma=0.5)

    # 保存热图数据
    # generator.save_heatmap_data(0, "keypoint_0_heatmap.npz", grid_size=50, sigma=0.5)

    print("\n热图生成完成！")