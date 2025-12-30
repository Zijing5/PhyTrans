import torch

def quat_inverse(quat):
    """计算四元数的逆（共轭，假设四元数已归一化）"""
    # quat: (env, 4) 格式为 (x, y, z, w)
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)  # 逆四元数: (-x, -y, -z, w)

def quat_multiply(q1, q2):
    """四元数乘法：q1 * q2"""
    # q1, q2: (env, 1, 4)
    x1, y1, z1, w1  = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)  # (env, 1, 4)

def world_to_bbox_local(hand_pos, box_pos, box_quat):
    """
    将世界坐标系中的手部位置转换到bbox的局部坐标系
    hand_pos: (env, 3) 世界坐标系中的手部位置
    box_pos: (env, 3) bbox在世界坐标系中的位置
    box_quat: (env, 4) bbox的旋转四元数 (x, y, z, w)
    返回: (env, 3) 局部坐标系中的手部位置
    """
    env_num = hand_pos.shape[0]
    # 1. 平移：手部位置减去bbox中心（世界坐标系）
    hand_rel_pos = hand_pos - box_pos  # (env, 3)
    
    # 2. 旋转：用逆四元数将相对位置转回局部坐标系
    inv_quat = quat_inverse(box_quat)  # (env, 4) 逆四元数
    
    # 将手部相对位置转换为四元数 (x, y, z, 0)
    p = torch.cat([hand_rel_pos.unsqueeze(1), torch.zeros(env_num, 1, 1, device=hand_pos.device)], dim=-1)  # (env, 1, 4)
    

    # 4. 正确旋转顺序：inv_quat × p × box_quat（q⁻¹ × p × q）
    # 步骤1：inv_quat 左乘 p
    q_inv = inv_quat.unsqueeze(1)  # (env,1,4)
    p_rot1 = quat_multiply(q_inv, p)  # (env,1,4) → q⁻¹ × p
    
    # 步骤2：结果右乘 box_quat
    q = box_quat.unsqueeze(1)  # (env,1,4)
    p_rot2 = quat_multiply(p_rot1, q)  # (env,1,4) → (q⁻¹ × p) × q
    
    # 5. 提取旋转后的3D坐标（取虚部x,y,z，忽略实部w）
    hand_local = p_rot2[..., :3].squeeze(1)  # (env, 3)，正确取前3个分量（虚部）
    
    return hand_local

def point_to_bbox_local_distance(hand_local, box_bps):
    """
    在局部坐标系中计算手部到bbox表面的最短距离
    hand_local: (env, 3) 局部坐标系中的手部位置
    box_bps: (env, 8, 3) 局部坐标系中的bbox顶点（无旋转无位移）
    返回: (env,) 每个环境中手部到bbox表面的最短距离
    """
    # 计算bbox在局部坐标系中的轴对齐边界
    bbox_min = torch.min(box_bps, dim=1).values  # (env, 3)
    bbox_max = torch.max(box_bps, dim=1).values  # (env, 3)
    
    # 计算手部在bbox内部的投影点
    proj_local = torch.clamp(hand_local, bbox_min, bbox_max)  # (env, 3)
    
    # 计算到表面的最短距离
    distance = torch.norm(hand_local - proj_local, dim=1)  # (env,)
    return distance

def get_hand_to_bbox_distances(box_states, vbox_bps, rhand_pos, lhand_pos):
    """计算左右手到bbox表面的最短距离"""
    box_pos = box_states[:, :3]  # (env, 3)
    box_quat = box_states[:, 3:7]  # (env, 4)
    
    # 转换手部位置到bbox局部坐标系 rhand_pos
    rhand_local = world_to_bbox_local(rhand_pos, box_pos, box_quat)
    lhand_local = world_to_bbox_local(lhand_pos, box_pos, box_quat)
    


    rhand_dist = point_to_bbox_local_distance(rhand_local, vbox_bps)
    lhand_dist = point_to_bbox_local_distance(lhand_local, vbox_bps)

    
    return rhand_dist, lhand_dist