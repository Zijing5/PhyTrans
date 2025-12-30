import numpy as np
import trimesh
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    """
    计算points1中每个点到points2中最近点的距离向量
    points1: 形状为(B, N, 3) - B个环境，每个环境有N个点
    points2: 形状为(B, M, 3) - B个环境，每个环境有M个点
    返回: 形状为(B, N, 3) - 每个点的最小距离向量
    """
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)  # (B, N, M, 3)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)  # (B, N, M)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)  # (B, N)
    
    # 创建索引网格
    B_indices, N_indices = torch.meshgrid(
        torch.arange(points1.shape[0], device=points1.device),
        torch.arange(points1.shape[1], device=points1.device),
        indexing='ij'
    )
    
    # 获取最小距离向量
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()  # (B, N, 3)
    return min_dis_mat


def generate_box_pointcloud(box_length, box_width, box_height, count=1024, seed=2024):
    """
    从立方体尺寸生成点云
    
    参数:
        box_length: 立方体长度
        box_width: 立方体宽度
        box_height: 立方体高度
        count: 采样点数量
        seed: 随机种子，保证结果可复现
        
    返回:
        经过中心化处理的点云张量
    """
    # 生成立方体的8个顶点
    half_l = box_length / 2
    half_w = box_width / 2
    half_h = box_height / 2
    
    vertices = np.array([
        [ half_l,  half_w,  half_h],  # 0
        [ half_l,  half_w, -half_h],  # 1
        [ half_l, -half_w,  half_h],  # 2
        [ half_l, -half_w, -half_h],  # 3
        [-half_l,  half_w,  half_h],  # 4
        [-half_l,  half_w, -half_h],  # 5
        [-half_l, -half_w,  half_h],  # 6
        [-half_l, -half_w, -half_h]   # 7
    ])
    
    # 定义立方体的12个面（每个面由2个三角形组成）
    faces = np.array([
        # 前面
        [0, 2, 6], [0, 6, 4],
        # 后面
        [1, 5, 7], [1, 7, 3],
        # 右面
        [0, 4, 5], [0, 5, 1],
        # 左面
        [2, 3, 7], [2, 7, 6],
        # 顶面
        [0, 1, 3], [0, 3, 2],
        # 底面
        [4, 6, 7], [4, 7, 5]
    ])
    
    # 创建trimesh网格
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 从表面均匀采样点
    points, _ = trimesh.sample.sample_surface_even(mesh, count=count, seed=seed)
    
    # 计算中心点并中心化
    center = np.mean(points, 0)
    points_centered = points - center
    
    # 转换为torch张量
    return torch.from_numpy(points_centered).float()



class PointCloudVisualizer:
    """点云可视化工具类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def set_data(self, points, colors=None):
        """
        设置要可视化的点云数据
        
        参数:
            points (np.ndarray): 形状为 (N, 3) 的点云数据
            colors (np.ndarray, 可选): 形状为 (N, 3) 或 (N, 4) 的颜色数据
        """
        self.points = points
        self.colors = colors
        
    def visualize(self, title="点云可视化", point_size=5, axis_equal=True):
        """
        显示点云
        
        参数:
            title (str): 图表标题
            point_size (float): 点的大小
            axis_equal (bool): 是否设置坐标轴等比例
        """
        if not hasattr(self, 'points'):
            raise ValueError("请先设置点云数据")
            
        # 绘制点云
        scatter = self.ax.scatter(
            self.points[:, 0], self.points[:, 1], self.points[:, 2],
            s=point_size, c=self.colors, cmap='viridis'
        )
        
        # 设置图表属性
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 设置坐标轴比例相等
        if axis_equal:
            self._set_axes_equal()
            
        # 添加颜色条（如果有颜色数据）
        if self.colors is not None and self.colors.ndim == 2 and self.colors.shape[1] in [3, 4]:
            self.fig.colorbar(scatter, ax=self.ax)
            
        plt.show()
        
    def _set_axes_equal(self):
        """使3D坐标轴等比例"""
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        # 扩展范围使三个坐标轴等比例
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        
        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])    
        
if __name__ == "__main__":
    # 生成一个边长为1的立方体点云
    box_length = 1.0
    box_width = 1.0
    box_height = 1.0
    point_count = 2048
    
    point_cloud = generate_box_pointcloud(box_length, box_width, box_height, count=point_count)
    
    # 可视化点云
    visualizer = PointCloudVisualizer()
    visualizer.set_data(point_cloud.numpy())
    visualizer.visualize(title="立方体点云", point_size=10)