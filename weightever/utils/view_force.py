import numpy as np
from isaacgym import gymapi, gymutil, gymtorch

# 全局标志：控制力的显示状态
# show_joint_forces = True    # 关节力显示开关
# show_contact_forces = True  # 接触力显示开关
def generate_multi_env_lines(envs, actor_handles,gym, show_joint, show_contact,contact_forces, dof_forces_tensor, rigid_body_states,scale=0.01):
    """为多个环境生成力线条（返回按环境分组的线条数据）"""
    # 存储格式：[(env1_verts, env1_colors, env1_num), (env2_verts, ...), ...]
    all_env_lines = []
    
    for env_idx, env_ptr in enumerate(envs):
        # 获取当前环境的角色actor_handle（假设actor_handles与envs一一对应）
        actor_handle = actor_handles[env_idx]
        
        # 为当前环境生成关节力线条
        j_verts, j_cols, j_num = [], [], 0
        if show_joint:
            j_verts, j_cols, j_num = _generate_joint_force_lines(
                env_ptr, actor_handle,gym, dof_forces_tensor, rigid_body_states, scale=scale
            )
        
        # 为当前环境生成接触力线条
        c_verts, c_cols, c_num = [], [], 0
        if show_contact:
            c_verts, c_cols, c_num = _generate_contact_force_lines(
                env_ptr, actor_handle, gym, contact_forces, rigid_body_states, scale=10*scale
            )
        
        # 合并当前环境的所有线条
        total_verts = j_verts + c_verts
        total_cols = j_cols + c_cols
        total_num = j_num + c_num
        
        all_env_lines.append( (total_verts, total_cols, total_num) )
    
    return all_env_lines


def _generate_joint_force_lines(env, actor_handle, gym, dof_forces_tensor, rigid_body_states, scale=0.01):
    """生成关节力可视化线条（从关节位置指向力的方向）"""
    line_vertices = []
    line_colors = []
    num_lines = 0

    # 获取角色的关节信息
    dof_count = gym.get_actor_dof_count(env, actor_handle)
    if dof_count == 0:
        return line_vertices, line_colors, num_lines

    # 获取关节力（扭矩）
    # dof_forces_tensor = gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim)).clone().cpu().numpy()
    # gym.release_dof_force_tensor(sim, dof_force_tensor)

    # 获取关节位置（用于线条起点）
    # dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_POS)
    # dof_pos_tensor = gym.acquire_dof_state_tensor(sim) # 306,2
    # dof_pos = gymtorch.wrap_tensor(dof_pos_tensor)[:,0].reshape(51,3,2) # 转为 PyTorch 张量
    # gym.refresh_dof_state_tensor(sim)  # 确保数据是最新的
    rigid_body_count = gym.get_actor_rigid_body_count(env, actor_handle)
    # rigid_body_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)) # 获取刚体位置

    # view_cloud_point = True
    # if view_cloud_point:
    #     visualizer = PointCloudVisualizer()
    #     visualizer.set_data(dof_pos[:,:,0].cpu().numpy())
    #     visualizer.visualize(title="point cloud", point_size=10)



    # rigid_body_pos = dof_pos.clone()*0.0
    # gym.compute_forward_kinematics(
    #     sim,
    #     gymtorch.unwrap_tensor(dof_pos),  # 传入原始张量（非 wrapped）
    #     gymtorch.unwrap_tensor(rigid_body_pos),  # 输出：刚体全局位置
    #     actor_handle  # 指定人形机器人的 Actor 句柄（关键！确保计算对应机器人）
    # )


    # 遍历关节生成线条（简化：取刚体位置作为关节力起点）
    # base_body_num = 1
    force_limit = 1e-2
    for i in range(rigid_body_count): # 51+1
        if i == 0:
            continue  # 忽略根关节
        body_idx = gym.get_actor_rigid_body_index(env, actor_handle, i, gymapi.DOMAIN_SIM) 
        dof_idx = gym.get_actor_dof_index(env, actor_handle, 3*(i-1), gymapi.DOMAIN_SIM) 

        force = [dof_forces_tensor[dof_idx], dof_forces_tensor[dof_idx+1], dof_forces_tensor[dof_idx+2]]
        if np.linalg.norm(force) < force_limit:  # 忽略极小的力
            continue
        force = [0.0 if np.abs(i)<force_limit else i for i in force] # set zero if the direction is too small


        start_pos0 = rigid_body_states[body_idx,:3]  # 关节起点（刚体位置）
        start_pos = gymapi.Vec3(start_pos0[0], start_pos0[1], start_pos0[2])  # 转为 Vec3 格式


        # 计算力的方向和终点（简化：沿关节轴方向，长度按力大小缩放）
        # 注意：关节力方向需根据关节类型调整，这里简化为沿局部X轴
        force_mag = np.linalg.norm(force)
        force_normalized = np.array(force)/force_mag
        force_dir = gymapi.Vec3(force_normalized[0],force_normalized[1],force_normalized[2])  # 示例方向，需根据实际关节轴调整
        end_pos = gymapi.Vec3(
            start_pos.x + force_dir.x * force_mag * scale,
            start_pos.y + force_dir.y * force_mag * scale,
            start_pos.z + force_dir.z * force_mag * scale
        )

        # 添加线条顶点和颜色（关节力用蓝色）
        line_vertices.extend([start_pos.x, start_pos.y, start_pos.z, end_pos.x, end_pos.y, end_pos.z])
        line_colors.extend([0.0, 0.0, 1.0])  # 蓝色
        num_lines += 1

    return line_vertices, line_colors, num_lines


def _generate_contact_force_lines(env, actor_handle, gym, contact_forces, rigid_body_states, scale=0.001):
    """生成接触力可视化线条（从接触点指向力的方向）"""
    line_vertices = []
    line_colors = []
    num_lines = 0

    # 获取接触力张量
    # contact_forces = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).cpu().numpy()
    rigid_body_count = gym.get_actor_rigid_body_count(env, actor_handle)
    # rigid_body_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)) # 获取刚体位置

    # gym.release_net_contact_force_tensor(sim, contact_force_tensor)

    # # 获取角色的所有刚体ID
    # actor_rigid_body_ids = gym.get_actor_rigid_body_ids(env, actor_handle)
    # if not actor_rigid_body_ids:
    #     return line_vertices, line_colors, num_lines
    force_limit = 1e-2  # 接触力阈值，忽略小于此值的接触力
    # 遍历刚体接触力
    for i in range(rigid_body_count): # 52
        body_idx = gym.get_actor_rigid_body_index(env, actor_handle, i, gymapi.DOMAIN_SIM) 
        # 接触力格式：[fx, fy, fz, tx, ty, tz]，取前3个为力的大小

        force = contact_forces[body_idx, :3] # 2x(52+1/4), 3 ?????
        force = [0.0 if np.abs(i)<force_limit else i for i in force] # set zero if the direction is too small
        force_mag = np.linalg.norm(force)
        if force_mag < force_limit:  # 忽略极小接触力
            continue

        # 获取接触点位置（简化：用刚体位置作为接触点，实际应从接触报告获取）
        start_pos0 = rigid_body_states[body_idx,:3]  # 关节起点（刚体位置）
        start_pos = gymapi.Vec3(start_pos0[0], start_pos0[1], start_pos0[2])  # 转为 Vec3 格式


        # 力的方向和终点
        end_pos = gymapi.Vec3(
            start_pos.x + force[0] * scale,
            start_pos.y + force[1] * scale,
            start_pos.z + force[2] * scale
        )

        # 添加线条顶点和颜色（接触力用红色）
        line_vertices.extend([start_pos.x, start_pos.y, start_pos.z, end_pos.x, end_pos.y, end_pos.z])
        line_colors.extend([1.0, 0.0, 0.0])  # 红色
        num_lines += 1

    return line_vertices, line_colors, num_lines




def init_view_force_keyboard_subscriptions(viewer, gym, gymapi):
    # 订阅J键（关节力显示切换）
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_J, "toggle_joint_force")
    # 订阅C键（接触力显示切换
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "toggle_contact_force")

# 键盘事件处理：切换关节力/接触力显示
def handle_key_events(viewer, gym):
    global show_joint_forces, show_contact_forces
    for evt in gym.query_viewer_action_events(viewer):
        # 处理关节力显示切换（J键）
        if evt.action == "toggle_joint_force" and evt.value == 1:  # value=1表示按键按下
            show_joint_forces = not show_joint_forces
            print(f"关节力显示: {'开启' if show_joint_forces else '关闭'}")
        # 处理接触力显示切换（C键）
        elif evt.action == "toggle_contact_force" and evt.value == 1:
            show_contact_forces = not show_contact_forces
            print(f"接触力显示: {'开启' if show_contact_forces else '关闭'}")