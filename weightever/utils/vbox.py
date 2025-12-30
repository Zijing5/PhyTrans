import torch
from isaacgym.torch_utils import *  # 假设存在四元数旋转函数
from weightever.utils import torch_utils

def compute_initial_b_part_bbox(vbox_pos_offset, vbox_length, vbox_width, vbox_height):
    """
    计算初始状态下（A在原点，无旋转）部件B的bbox 8个顶点
    
    参数:
        vbox_pos_offset: 部件B的几何中心 (shape: [num_envs, 3])
        vbox_length: B的长度 (shape: [num_envs] 或标量)
        vbox_width: B的宽度 (shape: [num_envs] 或标量)
        vbox_height: B的高度 (shape: [num_envs] 或标量)
        
    返回:
        8个顶点的坐标 (shape: [num_envs, 8, 3])
    """
    # 计算半尺寸
    half_len = vbox_length / 2.0
    half_wid = vbox_width / 2.0
    half_hgt = vbox_height / 2.0
    
    # 扩展维度以匹配batch操作 (num_envs, 1)
    half_len = half_len.unsqueeze(1) if isinstance(half_len, torch.Tensor) else torch.tensor([half_len], device=vbox_pos_offset.device)
    half_wid = half_wid.unsqueeze(1) if isinstance(half_wid, torch.Tensor) else torch.tensor([half_wid], device=vbox_pos_offset.device)
    half_hgt = half_hgt.unsqueeze(1) if isinstance(half_hgt, torch.Tensor) else torch.tensor([half_hgt], device=vbox_pos_offset.device)
    
    # 计算8个顶点（基于B的中心和半尺寸）
    # 注意：初始方向与A一致，因此轴方向相同
    lfus = vbox_pos_offset + torch.cat([-half_len, half_wid, half_hgt], dim=1)  # 左前上
    lfds = vbox_pos_offset + torch.cat([-half_len, half_wid, -half_hgt], dim=1) # 左前下
    lbus = vbox_pos_offset + torch.cat([-half_len, -half_wid, half_hgt], dim=1) # 左后上
    lbds = vbox_pos_offset + torch.cat([-half_len, -half_wid, -half_hgt], dim=1)# 左后下
    rfus = vbox_pos_offset + torch.cat([half_len, half_wid, half_hgt], dim=1)   # 右前上
    rfds = vbox_pos_offset + torch.cat([half_len, half_wid, -half_hgt], dim=1)  # 右前下
    rbus = vbox_pos_offset + torch.cat([half_len, -half_wid, half_hgt], dim=1)  # 右后上
    rbds = vbox_pos_offset + torch.cat([half_len, -half_wid, -half_hgt], dim=1) # 右后下
    
    # 组合为 [num_envs, 8, 3] 形状
    bbox_vertices = torch.stack([lfus, lfds, lbus, lbds, rfus, rfds, rbus, rbds], dim=1)
    return bbox_vertices


def compute_transformed_b_part_bbox(initial_b_vertices, box_pos, box_rot, root_states):
    """
    计算A经过位移(box_pos)和旋转(box_rot)后，部件B的bbox 8个顶点
    
    参数:
        initial_b_vertices: 初始状态下B的8个顶点 (shape: [num_envs, 8, 3])
        box_pos: A的位移 (shape: [num_envs, 3])
        box_rot: A的旋转四元数 (shape: [num_envs, 4])
        
    返回:
        变换后B的8个顶点 (shape: [num_envs, 8, 3])
    """
    num_envs = initial_b_vertices.shape[0]
    root_pos = root_states[:,0:3]
    root_rot = root_states[:,3:7]

    # 1. 扩展旋转四元数以匹配顶点数量
    # 从 [num_envs, 4] 扩展为 [num_envs, 8, 4]，再展平为 [num_envs*8, 4]
    box_rot_extended = box_rot.unsqueeze(1).repeat(1, 8, 1).view(-1, 4)
    
    # 2. 展平顶点坐标以便批量旋转 [num_envs*8, 3]
    vertices_flat = initial_b_vertices.view(-1, 3)
    
    # 3. 应用旋转（绕A的原点旋转）
    rotated_vertices = quat_rotate(box_rot_extended, vertices_flat)  # [num_envs*8, 3]
    
    # 4. 恢复形状并应用位移（A的位置偏移）
    rotated_vertices = rotated_vertices  # [num_envs*8, 3]
    world_vertices = rotated_vertices + box_pos.unsqueeze(1).repeat(1, 8, 1).view(-1, 3)  # 加上A的位移

    root_rot_extended = root_rot.unsqueeze(1).repeat(1, 8, 1).view(-1, 4)
    root_pos_extended = root_pos.unsqueeze(1).repeat(1, 8, 1).view(-1, 3)
    heading_rot_extended = torch_utils.calc_heading_quat_inv(root_rot_extended)
    local_vertices = quat_rotate(heading_rot_extended, world_vertices-root_pos_extended)
    local_vertices = local_vertices.view(num_envs,8,3)
    
    return local_vertices


# 使用示例
if __name__ == "__main__":
    # 假设环境数量为2
    num_envs = 2
    device = torch.device("cpu")
    
    # 部件B的初始参数（相对于A的局部坐标系）
    vbox_pos = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.4, 0.3]], device=device)  # B的中心
    vbox_length = torch.tensor([1.0, 1.2], device=device)
    vbox_width = torch.tensor([0.8, 0.9], device=device)
    vbox_height = torch.tensor([0.6, 0.7], device=device)
    
    # 步骤1：计算初始状态下B的bbox
    initial_bbox = compute_initial_b_part_bbox(vbox_pos, vbox_length, vbox_width, vbox_height)
    print("初始B的bbox顶点形状:", initial_bbox.shape)  # 应为 [2, 8, 3]
    
    # A的位移和旋转（示例）
    box_pos = torch.tensor([[2.0, 3.0, 0.0], [1.5, 2.5, 0.0]], device=device)  # A的位置
    box_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, torch.sin(torch.pi/4), 0.0, torch.cos(torch.pi/4)]], device=device)  # 旋转四元数（第二个环境绕y轴旋转45度）
    
    # 步骤2：计算A变换后B的bbox
    transformed_bbox = compute_transformed_b_part_bbox(initial_bbox, box_pos, box_rot)
    print("变换后B的bbox顶点形状:", transformed_bbox.shape)  # 应为 [2, 8, 3]
