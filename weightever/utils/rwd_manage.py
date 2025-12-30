import torch
from weightever.utils import torch_utils
from isaacgym.torch_utils import *
import swanlab
from collections import defaultdict

class State:
    CARRY = 0
    NAVI = 1
    RETRY = 2
    # 字符到数字的映射
    STR_TO_NUM = {
        "CARRY": CARRY,
        "NAVI": NAVI,
        "RETRY": RETRY
    }
    @staticmethod
    def to_num(str_state):
        """字符状态转数字"""
        return State.STR_TO_NUM.get(str_state.upper(), -1)
    
class BaseRewardState:
    def __init__(self):
        # 定义各奖励项的基础权重（可被子类覆盖）
        self.weights = {
            "walk_pos": 0.1,
            "walk_vel": 0.1,
            "walk_face": 0.1,
            "held_hand": 0.4,
            "height": 0.0,
            "carry_pos_far": 0.1,
            "carry_vel": 0.0,
            "carry_pos_near": 0.2,
            "carry_face": 0.2,
            "carry_dir": 0.1,
            "putdown": 0.0
        }
        self.reward_dict = {key: 0.0 for key in self.weights}    

    @torch.jit.script
    def compute_walk_reward(self, root_pos, root_rot, prev_root_pos, box_standing_pos, dt):
        # encourage the agent to walk towards box standing points

        near_threshold = 0.04
        target_speed = 1.0  # target speed in m/s
        pos_err_scale = 2.0
        vel_err_scale = 2.0

        # compute r_walk_pos
        box_standing_points_pos = box_standing_pos[..., 0:2]
        box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
        box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
        box_pos_reward = torch.exp(-pos_err_scale * box_pos_err)

        # compute r_walk_vel

        delta_root_pos = root_pos - prev_root_pos
        root_vel = delta_root_pos / dt.unsqueeze(1)
        box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1)
        box_dir_speed = torch.sum(box_dir * root_vel[..., :2], dim=-1)
        box_vel_err = target_speed - box_dir_speed
        box_vel_err = torch.clamp_min(box_vel_err, 0.0)
        vel_reward = torch.exp(-vel_err_scale * (box_vel_err * box_vel_err))
        speed_mask = box_dir_speed <= 0
        vel_reward[speed_mask] = 0

        # compute r_walk_face

        heading_rot = torch_utils.calc_heading_quat(root_rot)

        facing_dir = torch.zeros_like(root_pos[..., 0:3])
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)

        facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
        facing_reward = torch.clamp_min(facing_err, 0.0)

        # compute r_walk

        near_mask = box_pos_err <= near_threshold
        box_pos_reward[near_mask] = 1.0
        vel_reward[near_mask] = 1.0
        facing_reward[near_mask] = 1.0

        return box_pos_reward, vel_reward, facing_reward

    @torch.jit.script
    def compute_contact_reward(self,hand_positions, box_held_points, root_pos, box_standing_pos, box_pos, tar_pos):
        box_near_threshold = 0.09
        carry_dist_threshold = 0.04
        box_height_threshold = 0.4
        held_pos_err_scale = 5.0
        mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
        hand2box_diff = mean_hand_positions - box_held_points[..., 0:3]
        hands2box_pos_err = torch.sum(hand2box_diff * hand2box_diff, dim=-1)
        hands2box_reward = torch.exp(-held_pos_err_scale * hands2box_pos_err)
        # compute masks when walking to box
        # box_standing_points_pos = box_standing_pos[..., 0:2]
        # box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
        # box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
        # box_near_mask = box_pos_err <= box_near_threshold
        # hands2box_reward[~box_near_mask] = 0.0
        # compute masks when putdown
        box_height = box_held_points[..., 2]
        target_state_diff = tar_pos - box_pos  # xyz
        target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
        near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
        near_and_low_mask = torch.logical_and(
            near_mask, box_height < box_height_threshold)
        hands2box_reward[near_and_low_mask] = 1.0
        return hands2box_reward


    @torch.jit.script
    def compute_height_reward(self, held_point_height):
        target_height = 0.8
        height_err_scale = 10.0
        box_height_diff = target_height - held_point_height
        height_reward = torch.exp(
            -height_err_scale * box_height_diff * box_height_diff)
        return height_reward


    @torch.jit.script
    def compute_carry_reward(self, root_pos, root_rot, box_pos, box_rot, prev_box_pos, target_pos, target_rot, held_point_height, dt_tensor):
        target_speed = 1.0  # target speed in m/s
        carry_dist_threshold = 0.04
        height_threshold = 0.6
        tar_pos_err_far_scale = 0.5
        target_pos_err_near_scale = 10.0
        carry_vel_err_scale = 2.0

        x_axis = torch.zeros_like(root_pos[..., 0:3])
        x_axis[..., 0] = 1.0

        # masks
        box_height = box_pos[..., 2]
        height_mask = box_height < height_threshold

        # compute r_carry_pos
        target_state_diff = target_pos - box_pos  # xyz
        target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
        near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
        target_pos_err_xyz = torch.sum(target_state_diff[..., 0:3] ** 2, dim=-1)
        target_pos_reward_far = torch.exp(-tar_pos_err_far_scale *
                                        target_pos_err_xy)
        target_pos_reward_near = torch.exp(-target_pos_err_near_scale *
                                        target_pos_err_xyz)

        far_and_low_mask = torch.logical_and(~near_mask, height_mask)
        target_pos_reward_far[far_and_low_mask] = 0.0
        target_pos_reward_near[far_and_low_mask] = 0.0
        target_pos_reward_far[near_mask] = 1.0

        # compute_r_carry_face
        tar_dir = target_pos[..., 0:2] - box_pos[..., 0:2]
        tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
        tar_dir_reverse = box_pos[..., 0:2] - target_pos[..., 0:2]
        tar_dir_reverse = torch.nn.functional.normalize(tar_dir_reverse, dim=-1)
        root_heading_rot = torch_utils.calc_heading_quat(root_rot)
        root_facing_dir = quat_rotate(root_heading_rot, x_axis)
        # check whether the marker is behind the agent
        # if target is in front of the agent, then the agent should walk towards the target
        # if target is behind the agent, then the agent should walk backward to the target
        front_mask = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1) > 0
        behind_mask = torch.sum(
            tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1) > 0
        facing_err = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1)
        facing_err[behind_mask] = torch.sum(
            tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1)[behind_mask]
        facing_reward = torch.clamp_min(facing_err, 0.0)
        facing_reward[height_mask] = 0.0 # order here counts
        facing_reward[near_mask] = 1.0

        # compute r_carry_vel
        delta_box_pos = box_pos - prev_box_pos
        box_vel = delta_box_pos / dt_tensor.unsqueeze(1)
        box_tar_dir_speed = torch.sum(
            tar_dir * box_vel[..., 0:2], dim=-1)
        tar_vel_err = target_speed - box_tar_dir_speed
        tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
        tar_vel_reward = torch.exp(-carry_vel_err_scale *
                                (tar_vel_err * tar_vel_err))
        tar_speed_mask = box_tar_dir_speed <= 0
        tar_vel_reward[tar_speed_mask] = 0
        tar_vel_reward[height_mask] = 0.0

        # compute r_carry_dir
        # calculate the facing direction of the box
        box_facing_dir = quat_rotate(box_rot, x_axis)
        tar_facing_dir = quat_rotate(target_rot, x_axis)
        dir_err = torch.sum(
            box_facing_dir[..., 0:2] * tar_facing_dir[..., 0:2], dim=-1)  # xy;higher value indicating better alignment
        dir_reward = torch.clamp_min(dir_err, 0.0)
        dir_reward[~near_mask] = 0.0

        # compute r_putdown
        held_points_height = held_point_height - target_pos[..., 2]
        put_down_height_reward = torch.exp(
            -5.0 * held_points_height * held_points_height)
        put_down_height_reward[~near_mask] = 0
        return target_pos_reward_far, tar_vel_reward, target_pos_reward_near, facing_reward, dir_reward, put_down_height_reward




    @torch.jit.script
    def compute_carry_energy_reward(self, joint_vels, dof_forces, foot_forces):
        # 初始化scale张量（避免循环赋值，更高效）
        scale = torch.ones(joint_vels.shape[-1], device=joint_vels.device)  # 确保与输入在同一设备
        jnt = [0, 1, 6, 7, 8]
        scale[jnt] = 0.5  # 直接批量赋值，替代循环
        
        energy_scale = 1e-5
    
        # 关键修复：用向量化操作替代循环和列表推导
        # 计算每个关节的能量项（元素级乘法+平方）
        energy_terms = (scale * joint_vels * dof_forces).pow(2)  # 形状: (N, D)，N是批量大小，D是关节数
        # 对每个样本的所有关节求和，得到每个样本的总能量
        dof_energy = energy_terms.sum(dim=1)  # 形状: (N,)
        
        # 应用指数衰减（保持原逻辑）
        dof_energy = torch.exp(-energy_scale * dof_energy)  # 形状: (N,)
        
        # # ---------------------- 2. 脚底力垂直性奖励 ----------------------
        # left_force = foot_forces[..., 0:3]  # (env, 3)
        # right_force = foot_forces[..., 6:9]  # (env, 3)
        
        # left_vertical = verticality_reward(left_force)
        # right_vertical = verticality_reward(right_force)
        # vertical_reward = (left_vertical + right_vertical) / 2  # 
        vertical_reward = dof_energy
    
        # # ---------------------- 3. 地面压力奖励 ----------------------
        # left_pressure = pressure_reward(left_force)
        # right_pressure = pressure_reward(right_force)
        # press_reward = (left_pressure + right_pressure) / 2  # 左右脚平均
        press_reward = dof_energy
        
        return dof_energy, vertical_reward, press_reward

    # @torch.jit.script
    def compute_reward(self, root_pos, root_rot , prev_root_pos,
        box_pos, box_rot, prev_box_pos,
        tar_pos, tar_rot, held_point_height, hand_positions,
        box_held_pos, box_standing_pos, dt_tensor, *args,  **kwargs):


        walk_pos, walk_vel, walk_face = self.compute_walk_reward(
            root_pos, root_rot, prev_root_pos, box_standing_pos, dt_tensor)
        held_hand = self.compute_contact_reward(
            hand_positions, box_held_pos, root_pos, box_standing_pos, box_pos, tar_pos)
        height = self.compute_height_reward(held_point_height)
        carry_pos_far, carry_vel, carry_pos_near, carry_face, carry_dir, putdown = self.compute_carry_reward(
            root_pos, root_rot, box_pos, box_rot, prev_box_pos, tar_pos, tar_rot, held_point_height, dt_tensor)
        

        self.reward_dict = {
            "walk_pos": walk_pos * self.weights["walk_pos"],
            "walk_vel": walk_vel * self.weights["walk_vel"],
            "walk_face": walk_face * self.weights["walk_face"],
            "held_hand": held_hand * self.weights["held_hand"],
            "height": height * self.weights["height"],
            "carry_pos_far": carry_pos_far * self.weights["carry_pos_far"],
            "carry_vel": carry_vel * self.weights["carry_vel"],
            "carry_pos_near": carry_pos_near * self.weights["carry_pos_near"],
            "carry_face": carry_face * self.weights["carry_face"],
            "carry_dir": carry_dir * self.weights["carry_dir"],
            "putdown": putdown * self.weights["putdown"]
        }

        return self.reward_dict


class NormalCarryState(BaseRewardState):
    def __init__(self):
        super().__init__()
        # 正常搬运状态的权重（覆盖基类默认值）
        self.weights = {
            "walk_pos": 0.1,
            "walk_vel": 0.1,
            "walk_face": 0.1,
            "held_hand": 0.4,  # 强化手物接触奖励
            "height": 0.1,      # 新增高度奖励权重
            "carry_pos_far": 0.1,
            "carry_vel": 0.05,
            "carry_pos_near": 0.2,
            "carry_face": 0.2,
            "carry_dir": 0.1,
            "putdown": 0.0
        }


class RetryState(BaseRewardState):
    def __init__(self):
        super().__init__()
        # # 重试状态的权重（弱化搬运奖励，强化导航奖励）
        # self.weights = {
        #     "walk_pos": 0.3,    # 强化导航到新站立点的位置奖励
        #     "walk_vel": 0.2,    # 强化移动速度奖励
        #     "walk_face": 0.3,   # 强化朝向新站立点的奖励
        #     "held_hand": 0.1,   # 弱化手物接触奖励（失败后不关注握持）
        #     "height": 0.0,
        #     "carry_pos_far": 0.0,
        #     "carry_vel": 0.0,
        #     "carry_pos_near": 0.0,
        #     "carry_face": 0.0,
        #     "carry_dir": 0.0,
        #     "putdown": 0.0,
        #     "retry_stand_point_reward":0.3,
        #     "retry_facing_reward":0.3,
        # }
        self.weights["retry_stand_point_reward"] = 0.3
        self.weights["retry_facing_reward"] = 0.3
        self.weights["walk_face"] = 0.1

    
        # self.weights = {
        #     "walk_pos": 0.1,
        #     "walk_vel": 0.1,
        #     "walk_face": 0.1,
        #     "held_hand": 0.4,
        #     "height": 0.0,
        #     "carry_pos_far": 0.1,
        #     "carry_vel": 0.0,
        #     "carry_pos_near": 0.2,
        #     "carry_face": 0.2,
        #     "carry_dir": 0.1,
        #     "putdown": 0.0
        # }
    
    def compute_reward(self, root_pos, root_rot , prev_root_pos,
        box_pos, box_rot, prev_box_pos,
        tar_pos, tar_rot, held_point_height, hand_positions,
        box_held_pos, box_standing_pos, dt_tensor,
        foot_positions, bbox_states, box_bps, pre_box_standing_pos):
        # 先调用父类计算基础奖励
        reward_dict = super().compute_reward(root_pos, root_rot , prev_root_pos,
                                            box_pos, box_rot, prev_box_pos,
                                            tar_pos, tar_rot, held_point_height, hand_positions,
                                            box_held_pos, box_standing_pos, dt_tensor)
        # base_reward = sum(self.reward_dict.values())
        # 额外添加前往备选点的重试奖励（假设新增函数）
        # root_pos = args[0]
        # 假设通过kwargs传递备选点信息


        lfus = box_bps[0, 0]
        lfds = box_bps[0, 1]
        lbus = box_bps[0, 2]
        lbds = box_bps[0, 3]

        rfus = box_bps[0, 4]
        rfds = box_bps[0, 5]
        rbus = box_bps[0, 6]
        rbds = box_bps[0, 7]

        curr_lfus = convert_static_point_to_world(lfus, bbox_states[:,:3], bbox_states[:,3:7])
        curr_lfds = convert_static_point_to_world(lfds, bbox_states[:,:3], bbox_states[:,3:7])
        curr_lbus = convert_static_point_to_world(lbus, bbox_states[:,:3], bbox_states[:,3:7])
        curr_lbds = convert_static_point_to_world(lbds, bbox_states[:,:3], bbox_states[:,3:7])
        curr_rfus = convert_static_point_to_world(rfus, bbox_states[:,:3], bbox_states[:,3:7])
        curr_rfds = convert_static_point_to_world(rfds, bbox_states[:,:3], bbox_states[:,3:7])
        curr_rbus = convert_static_point_to_world(rbus, bbox_states[:,:3], bbox_states[:,3:7])
        curr_rbds = convert_static_point_to_world(rbds, bbox_states[:,:3], bbox_states[:,3:7])

        # 获取所有边界框顶点的投影
        bbox_vertices = [
            curr_lfus, curr_lfds, curr_lbus, curr_lbds,
            curr_rfus, curr_rfds, curr_rbus, curr_rbds
        ]

        # 投影所有边界框顶点到xy平面，形状变为 (env, 8, 2)
        projected_bbox = torch.stack([vert[:, :2] for vert in bbox_vertices], dim=1)

        # 投影foot_positions到xy平面，形状为 (env, 2)
        projected_foot = foot_positions[:, :2]

        # 扩展foot_positions形状以匹配边界框顶点，便于广播计算，形状变为 (env, 8, 2)
        expanded_foot = projected_foot.unsqueeze(1).expand(-1, 8, -1)

        # 计算每个环境中foot_positions与8个边界框顶点的欧氏距离
        distances = torch.sqrt(torch.sum((projected_bbox - expanded_foot) **2, dim=2))  # 形状为 (env, 8)

        # 找到每个环境中距离最近的两个顶点的索引
        # argsort返回按距离排序的索引，取前两个
        nearest_indices = torch.argsort(distances, dim=1)[:, :2]  # 形状为 (env, 2)

        # 获取这两个最近顶点的投影位置
        # 为了使用高级索引，我们需要创建环境索引
        env_indices = torch.arange(projected_bbox.size(0)).unsqueeze(1)
        # 最近的两个点的投影位置，形状为 (env, 2, 2)
        nearest_points = projected_bbox[env_indices, nearest_indices]
        side_dir = nearest_points[:, 0]-nearest_points[:, 1]
        side_dir = torch.nn.functional.normalize(side_dir, -1)

        retry_facing_reward, retry_stand_point_reward = self.compute_retry_reward(root_pos, root_rot, side_dir, 
                                           box_standing_pos, pre_box_standing_pos)

        reward_dict["retry_facing_reward"] = retry_facing_reward
        reward_dict["retry_stand_point_reward"] = retry_stand_point_reward

        return reward_dict
    
    def compute_retry_reward(self,root_pos, root_rot, side_dir, box_standing_pos, pre_box_standing_pos):
        standp_k = 0.5
        move_dist_threshold = 0.1

        pre_box_standing_points_pos = pre_box_standing_pos[..., 0:2]
        pre_box_pos_diff = pre_box_standing_points_pos - root_pos[..., 0:2]
        pre_box_pos_err = torch.norm(pre_box_pos_diff, dim=-1)

        box_standing_points_pos = box_standing_pos[..., 0:2]
        box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
        box_pos_err = torch.norm(box_pos_diff, dim=-1)
        box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1)

        near_mask = box_pos_err <= move_dist_threshold  # near_mask

        bl = (pre_box_pos_err/box_pos_err)
        stand_point_reward = bl/(bl+standp_k)
        stand_point_reward[near_mask] = 1




        heading_rot = torch_utils.calc_heading_quat(root_rot)
        facing_dir = torch.zeros_like(root_rot[..., 0:3])
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)

        side_dir_reverse = -side_dir
        behind_mask = torch.sum(
            side_dir_reverse * facing_dir[..., 0:2], dim=-1) > 0
        side_facing_err = torch.sum(side_dir * facing_dir[..., 0:2], dim=-1)
        side_facing_err[behind_mask] = torch.sum(
            side_dir_reverse * facing_dir[..., 0:2], dim=-1)[behind_mask]
        side_facing_reward = torch.clamp_min(side_facing_err, 0.0)
        box_facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
        box_facing_reward = torch.clamp_min(box_facing_err, 0.0)
        facing_reward = torch.max(side_facing_reward,box_facing_reward)




        return facing_reward, stand_point_reward
        # facing_reward[height_mask] = 0.0 # order here counts
        # facing_reward[near_mask] = 1.0

class NaviState(BaseRewardState):
    def __init__(self):
        super().__init__()
        # # 重试状态的权重（弱化搬运奖励，强化导航奖励）
        # self.weights = {
        #     "walk_pos": 0.3,    # 强化导航到新站立点的位置奖励
        #     "walk_vel": 0.2,    # 强化移动速度奖励
        #     "walk_face": 0.3,   # 强化朝向新站立点的奖励
        #     "held_hand": 0.1,   # 弱化手物接触奖励（失败后不关注握持）
        #     "height": 0.0,
        #     "carry_pos_far": 0.0,
        #     "carry_vel": 0.0,
        #     "carry_pos_near": 0.0,
        #     "carry_face": 0.0,
        #     "carry_dir": 0.0,
        #     "putdown": 0.0,
        #     "retry_stand_point_reward":0.3,
        #     "retry_facing_reward":0.3,
        # }
        self.weights["navi_vel_dir_rwd"] = 0.3
        self.weights["navi_keep_vel_rwd"] = 0.1
        self.weights["navi_body_dis_rwd"] = 0.3
        self.weights["navi_facing_reward"] = 0.3
        self.weights["back_to_obstacle_penalty"] = 0.3

        

        
        # self.weights = {
        #     "walk_pos": 0.1,
        #     "walk_vel": 0.1,
        #     "walk_face": 0.1,
        #     "held_hand": 0.4,
        #     "height": 0.0,
        #     "carry_pos_far": 0.1,
        #     "carry_vel": 0.0,
        #     "carry_pos_near": 0.2,
        #     "carry_face": 0.2,
        #     "carry_dir": 0.1,
        #     "putdown": 0.0
        # }
    
    def compute_reward(self,box_pos, vbox_pos, root_states, prev_root_pos, box_standing_pos, min_dis_mat, dt_tensor):
        # 先调用父类计算基础奖励
        # reward_dict = super().compute_reward(root_pos, root_rot , prev_root_pos,
        #                                     box_pos, box_rot, prev_box_pos,
        #                                     tar_pos, tar_rot, held_point_height, hand_positions,
        #                                     box_held_pos, box_standing_pos, dt_tensor)



                                                                                                            
        navi_vel_dir_rwd, navi_keep_vel_rwd, navi_body_dis_rwd, navi_facing_reward, back_to_obstacle_penalty = self.compute_navi_reward(box_pos, vbox_pos,root_states, prev_root_pos, box_standing_pos, min_dis_mat, dt_tensor)
        # dof_energy,_,_ = self.compute_carry_energy_reward()
        self.reward_dict["navi_vel_dir_rwd"] = navi_vel_dir_rwd*self.weights["navi_vel_dir_rwd"]
        self.reward_dict["navi_keep_vel_rwd"] = navi_keep_vel_rwd*self.weights["navi_keep_vel_rwd"]
        self.reward_dict["navi_body_dis_rwd"] = navi_body_dis_rwd*self.weights["navi_body_dis_rwd"]
        self.reward_dict["navi_facing_reward"] = navi_facing_reward*self.weights["navi_facing_reward"]
        self.reward_dict["back_to_obstacle_penalty"] = back_to_obstacle_penalty*self.weights["back_to_obstacle_penalty"]





        return self.reward_dict
                            
    def compute_navi_reward(self, box_pos, vbox_pos, root_states, prev_root_pos, box_standing_pos, min_dis_mat, dt):
        body_dis_scale = 10
        tar_dis_scale = 5
        keep_vel_rwd_scale = 10

        root_pos = root_states[...,:3]
        tar_dis_ = box_standing_pos-root_pos
        tar_dis_[...,2] = 0.0
        tar_vel = torch.nn.functional.normalize(tar_dis_,dim=-1)
        tar_dis = torch.norm(tar_dis_,dim=-1) 
        

        delta_root_pos = root_pos - prev_root_pos
        root_vel = delta_root_pos / dt.unsqueeze(1)
        root_vel_scale = torch.norm(root_vel, dim=-1)
        root_vel_dir = torch.nn.functional.normalize(root_vel,dim=-1)

        # near_tar_standp_mask
        near_tar_standp_mask = tar_dis <= 0.5

        # vel dir rwd
        vel_dir_rwd =  torch.sum(tar_vel * root_vel_dir, dim=-1) #should be (env,) 
        vel_dir_rwd[near_tar_standp_mask] = 1


        # keep vel rwd
        expected_vel_ = 1-torch.exp(-tar_dis_scale * (tar_dis**2))
        keep_vel_rwd = torch.exp(-keep_vel_rwd_scale*(root_vel_scale-expected_vel_)**2)
        keep_vel_rwd[near_tar_standp_mask] = 1

        # body_dis at least 
        body_dis = torch.mean(torch.norm(min_dis_mat.view(-1,15,3),dim=-1),dim=-1) # (envs, 45) 0-inf
        body_dis_rwd = 1-torch.exp(-body_dis_scale*(body_dis**2)) 
        almost_enough_mask = body_dis_rwd>=0.8 # body_dis>0.4m
        body_dis_rwd[almost_enough_mask] = 1.0

        body_dis_min = torch.min(torch.norm(min_dis_mat.view(-1,15,3),dim=-1),dim=-1).values
        almost_coll_mask = body_dis_min<0.1
        body_dis_rwd[almost_coll_mask] = -10

        tar_dir = vbox_pos[..., 0:2]-root_pos[..., 0:2]
        tar_dir = torch.nn.functional.normalize(tar_dir,dim=-1)
        root_rot = root_states[...,3:7]
        x_axis = torch.zeros_like(root_pos[..., 0:3])
        x_axis[..., 0] = 1.0
        root_heading_rot = torch_utils.calc_heading_quat(root_rot)
        root_facing_dir = quat_rotate(root_heading_rot, x_axis)

        facing_err = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1)
        navi_facing_reward = torch.clamp_min(facing_err, 0.0)
        # far_mask = tar_dis >= 1
        # facing_reward[far_mask] = 0.0 # order here counts
        # facing_reward[near_mask] = 1.0
        # box_pos_local = quat_rotate(
        #         torch_utils.calc_heading_quat_inv(root_rot), box_pos - root_pos)[..., 0:2]
        barrier_pos_local = quat_rotate(
                torch_utils.calc_heading_quat_inv(root_rot), torch.mean(min_dis_mat.view(-1,15,3),dim=1))[..., 0:2]
        
        # box_dir = torch.nn.functional.normalize(box_pos_local, dim=-1, p=2)# 若障碍物在角色后方（x轴负方向），且距离较近，给予惩罚
        # behind_obstacle_mask = (box_dir[..., 0] < -0.5) & (body_dis_min < 1.5)

        barrier_dir = torch.nn.functional.normalize(barrier_pos_local, dim=-1, p=2)
        behind_obstacle_mask = (barrier_dir[..., 0] < -0.5) & (body_dis < 1.5)

        back_to_obstacle_penalty = -0.5 * behind_obstacle_mask.float()# 3. 组合朝向奖励（目标导向+避障朝向）



        return vel_dir_rwd, keep_vel_rwd, body_dis_rwd, navi_facing_reward, back_to_obstacle_penalty
        # facing_reward[height_mask] = 0.0 # order here counts
        # facing_reward[near_mask] = 1.0


# class RewardStateMachine:
#     def __init__(self):
#         self.current_state = "CARRY"  # 初始状态：正常搬运
#         self.states = {
#             "CARRY": NormalCarryState(),
#             "RETRY": RetryState()
#             # "PUT_DOWN": PutDownState()  # 可扩展其他状态
#         }
    
#     def set_state(self, state_name):
#         if state_name in self.states:
#             self.current_state = state_name
#         else:
#             raise ValueError(f"未知状态: {state_name}")
    
#     def compute_total_reward(self, *args, **kwargs):
#         # 委托给当前状态的奖励计算逻辑
#         return self.states[self.current_state].compute_reward(*args, **kwargs) # include all and spilit_dict
    

class MultiEnvRewardStateMachine():
    def __init__(self, env_num, init_state_name, device):
        self.init_state_name = init_state_name
        self.device =device
        self.env_num = env_num
        self.current_states = torch.full(
            (env_num,), State.to_num(init_state_name), device=self.device)  # 每个环境的状态数组
        
        self.states = {
            State.CARRY: NormalCarryState(),
            State.NAVI: NaviState(),
            State.RETRY: RetryState()
        }
        # 为每个环境维护独立的重试计数
        self.retry_counts = torch.zeros(env_num, dtype=torch.int32, device=self.device)
        self.max_retries = 3  # 最大重试次数
    
        history_length = 10
        # 历史箱子位置 (env_num, history_length, 3)
        self.box_pos_history = torch.zeros((env_num, history_length, 3), device=self.device)
        # 历史箱子高度 (env_num, history_length)
        self.box_height_history = torch.zeros((env_num, history_length), device=self.device)
        self.hand_touched_history = torch.zeros((env_num, history_length), device=self.device)
        self.his_failure_mask = torch.zeros((env_num, history_length), device=self.device)
        self.valid_failure_mask = torch.zeros((env_num, 3*history_length), device=self.device)
        self.navi_reached_history = torch.zeros((env_num, history_length), device=self.device)
        # reward


        # log
        self.is_log=False
        
    def reset(self, env_ids):
        self.current_states[env_ids] = State.to_num(self.init_state_name)
        self.retry_counts[env_ids] *= 0

    def set_state(self, env_indices, state_name):
        """set state for specified env"""
        if State.to_num(state_name) not in self.states:
            raise ValueError(f"unk name: {state_name}")
        self.current_states[env_indices] = State.to_num(state_name)
    
    def compute_total_reward(self,box_pos, vbox_pos, root_states, prev_root_pos, 
                                        box_standing_pos, min_dis_mat, dt_tensor):
        """批量计算所有环境的奖励"""
        all_rewards = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)
        # all_reward_dicts = [{}]*self.env_num # 存储每个环境的奖励详情
        # unique_states, state_indices = torch.unique(self.current_states, return_inverse=True)
        # state_groups = defaultdict(list)
        # for env_idx, state in enumerate(self.current_states):
        #     state_groups[state.item()].append(env_idx)
        unique_states = torch.unique(self.current_states)
        # hardcode!! init reward_dict with all possible keys
        all_reward_dicts = {k: torch.zeros(self.env_num, device=self.device) for k in self.states[2].weights.keys()}
        # all_reward_dicts.update({"retry_facing_reward": torch.zeros(self.env_num, device=self.device),
        #                         "retry_stand_point_reward": torch.zeros(self.env_num, device=self.device)})

        for state in unique_states:
            # 批量获取该状态的所有环境索引（张量操作）
            env_mask = self.current_states == state
            env_indices = torch.where(env_mask)[0]
            if len(env_indices) == 0:
                continue
            state_class = self.states[state.item()]
            batch_reward = state_class.compute_reward(box_pos, vbox_pos,
                                        root_states, prev_root_pos, 
                                        box_standing_pos, min_dis_mat, dt_tensor)
            for k, v in batch_reward.items():
                all_reward_dicts[k][env_mask] = v

        # batch_reward_dict = self._merge_reward_dicts(all_reward_dicts)
        return all_reward_dicts
    
    # def _merge_reward_dicts(self, reward_dicts):
    #     """将每个环境的奖励字典合并为批量字典"""
    #     if not reward_dicts:
    #         return {}
        
    #     batch_dict = {}
    #     first_dict = reward_dicts[0]
    #     for key in first_dict.keys():
    #         # 假设所有环境的奖励字典键一致
    #         batch_dict[key] = torch.tensor(
    #             [d[key] for d in reward_dicts], 
    #             dtype=torch.float32,
    #             device=first_dict[key].device if isinstance(first_dict[key], torch.Tensor) else None
    #         )
    #     return batch_dict
    
    def update_history(self, box_pos, vbox_height, hand_contact, failure_mask, valid_failure, navi_reached, env_ids=None):
        """更新箱子位置和高度的历史记录"""
        # 左移历史记录，添加新数据 # env, len, dim
        if env_ids!=None:  # this is either reset or init : should override last record instead of roll
            self.box_pos_history[env_ids]*=0
            self.box_height_history[env_ids]*=0
            self.hand_touched_history[env_ids]*=0
            self.box_pos_history[env_ids, -1] = box_pos # check nect time here should be next step!!!
            self.box_height_history[env_ids, -1] = vbox_height
            self.hand_touched_history[env_ids, -1] = hand_contact
            self.his_failure_mask[env_ids]*= 0 # detect failure record
            self.valid_failure_mask[env_ids]*= 0 # detect failure and change stand point record
            self.navi_reached_history[env_ids]*=0
        else:
            self.box_pos_history = torch.roll(self.box_pos_history, -1, dims=1)
            self.box_height_history = torch.roll(self.box_height_history, -1, dims=1)
            self.hand_touched_history = torch.roll(self.hand_touched_history, -1, dims=1)
            self.his_failure_mask = torch.roll(self.his_failure_mask, -1, dims=1)
            self.valid_failure_mask = torch.roll(self.valid_failure_mask, -1, dims=1)
            self.navi_reached_history = torch.roll(self.navi_reached_history, -1, dims=1)


            self.box_pos_history[:, -1] = box_pos
            self.box_height_history[:, -1] = vbox_height
            self.hand_touched_history[:, -1] = hand_contact
            self.his_failure_mask[:,-1] = failure_mask
            self.valid_failure_mask[:,-1] = valid_failure

            self.navi_reached_history[:,-1] = 0
            self.navi_reached_history[torch.where(navi_reached)[0],-1] = 1



    def is_lift_failure(self, root_pos, vbox_pos, vbox_height, vtar_box_pos, dt):
        """
        检测是否满足搬运失败条件（返回形状为(env_num,)的布尔张量）
        """
        device = root_pos.device
        env_num = self.env_num
        failure_mask = torch.zeros(env_num, dtype=torch.bool, device=device)
        
        carry_dist_threshold = 0.8
        character_dist_threshold = 1.0 # TODO: bigger obj with bigger dist
        box_vel_mag_threshold = 0.02 # TODO: need to be properly set
        
        # 1. 角色是否靠近箱子（距离<0.5m）
        root_to_box_dist = torch.sqrt(torch.sum((root_pos[:,:2] - vbox_pos[:,:2])** 2, dim=1))
        near_box_mask = root_to_box_dist < character_dist_threshold
        
        # 2. 箱子是否远离目标（距离>0.2m）
        box_to_tar_dist = torch.sqrt(torch.sum((vbox_pos[:,:2] - vtar_box_pos[:,:2])** 2, dim=1))
        far_from_tar_mask = box_to_tar_dist > carry_dist_threshold
        
        # 3. 箱子是否停滞（历史平均速度<0.05m/s）
        box_vel_history = (self.box_pos_history[:, -1] - self.box_pos_history[:, -5]) / (dt * 4)
        # box_vel_history = (self.box_pos_history[:, -1] - self.box_pos_history[:, 0]) / (dt * self.box_pos_history.shape[-2])

        box_vel_mag = torch.sqrt(torch.sum(box_vel_history**2, dim=1))
        stagnant_mask = box_vel_mag < box_vel_mag_threshold
        # whether hand haved touched: # delayed mechanisim! 
        hand_touched_mask = torch.sum(self.hand_touched_history[:,:2],dim=-1) == 2
        
        # 4. 箱子高度是否低或掉落（当前高度<0.3m 或 历史高度差<-0.1m）
        low_height_mask = vbox_height < 0.35 #TODO: need to be reset # hardcode
        current_height_threshold = vbox_height + 0.1  # 形状: (env_num,)
        current_height_threshold = current_height_threshold.unsqueeze(1) # (env_num, 1)
        # 历史中是否存在高于阈值的记录 (env_num, history_length) → (env_num,)
        height_drop_mask = torch.any(
            self.box_height_history > current_height_threshold, 
            dim=1  # 沿历史时间维度检查
        )
        if (self.is_log):
            swanlab.log({"failcond: near and far_tar": torch.mean(torch.logical_and(near_box_mask, far_from_tar_mask).float()),
                        "failcond: stagnant and touched": torch.mean(torch.logical_and(stagnant_mask, hand_touched_mask).float()),
                        "failcond: picked and fall": torch.mean(torch.logical_and(height_drop_mask, low_height_mask).float()),                                  
                                                                 })

        # 综合失败条件：靠近箱子 + 远离目标 + (停滞且touched 或 高度掉落)
        failure_mask = torch.logical_and(
            torch.logical_and(near_box_mask, far_from_tar_mask),
            torch.logical_or(torch.logical_and(stagnant_mask, hand_touched_mask), 
                             torch.logical_and(height_drop_mask, low_height_mask)
                             ))
        # print("stagnant_mask",stagnant_mask[0],"    hand_touched_mask", hand_touched_mask[0])
        # print("height_drop_mask",height_drop_mask[0],"    low_height_mask", low_height_mask[0])

        return failure_mask
    
    def is_putdown(self, root_pos, vbox_height, failure_envs):
        # TODO: regulate the bound!!!
        human_stand_mask = root_pos[:,2]>0.6
        box_down_mask = vbox_height<0.5 # hardcode
        putdown_mask = torch.logical_and(human_stand_mask, box_down_mask)
        putdown_envids = torch.where(putdown_mask)[0]
        # keep_mask = torch.isin(failure_envs, putdown_envids, invert=False)
        # # assert len(keep_mask)+len(endnavi) == len(failure_envs)
        # putdown_envids = failure_envs[keep_mask]
        # failure_envs = failure_envs[~keep_mask]

        # return env_id
        return putdown_envids
    
    def is_navi_reached(self,root_pos, box_standing_points):
        navi_envs = self.get_envs("NAVI")
        if len(navi_envs) == 0: # long? int?
            return navi_envs # should be [] 
        # cond 1: 
        # root_pos near box standing pos in xy
        distance_threshold = 0.5  # 示例：距离小于0.5视为到达
    
        # 1. 提取navi_envs对应的根节点XY坐标
        # 假设root_pos形状为[num_envs, 3]，其中前两列是X、Y坐标
        root_xy = root_pos[navi_envs, :2]  # 形状: [len(navi_envs), 2]
        
        # 2. 提取对应盒子站点的XY坐标
        # 假设box_standing_points形状为[num_envs, 2]或[num_envs, 3]（取前两列XY）
        box_xy = box_standing_points[navi_envs, :2]  # 形状: [len(navi_envs), 2]
        
        # 3. 计算XY平面上的欧氏距离（避免开方，用平方距离比较更高效）
        xy_diff = root_xy - box_xy  # 坐标差
        squared_distances = torch.sum(xy_diff **2, dim=1)  # 平方和（即距离的平方）
        
        # 4. 筛选距离小于等于阈值的环境（比较平方距离避免开方运算）
        reached_mask = squared_distances <= (distance_threshold** 2)

    

        
        # 5. 从navi_envs中提取满足条件的环境索引
        reached_envs = navi_envs[reached_mask]

        return reached_envs
    
    def get_envs(self, state_name):
        env_mask = self.current_states == State.to_num(state_name)
        env_indices = torch.where(env_mask)[0].to(self.device)
        return env_indices


    def get_current_standby_points(self, env_indices):
        """获取当前环境的备选站立点和抓握点"""
        idx = ((self.retry_counts[env_indices]) % 4).long()
        stand_offset = self.standby_stand_points[env_indices, idx,:]
        held_offset = self.standby_held_points[env_indices, idx, :]
        # TODO: replace with stand_offset,held_offset = pred() # pred from wight_pred class
        return stand_offset, held_offset
    
    def init_offset(self,s,h):
        self.standby_stand_points = s
        self.standby_held_points = h

    def log(self, is_log= False):
        self.is_log = is_log


@torch.jit.script
def convert_static_point_to_world(point_pos, central_pos, central_rot):
    point_states = torch.zeros_like(central_pos[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    return target_point_staets

# @torch.jit.script
def verticality_reward(foot_force, eps: float = 1e-6, vertical_k: float = 5.0):
    fx, fy, fz = foot_force[..., 0], foot_force[..., 1], foot_force[..., 2]
    horizontal_mag = torch.sqrt(fx**2 + fy**2 + eps)  # 水平力大小（加eps避免除0）
    vertical_mag = torch.abs(fz) + eps  # 垂直力大小
    ratio = horizontal_mag / vertical_mag  # 期望趋近于0
    return torch.exp(-vertical_k * ratio)  # 期望趋近于1

# @torch.jit.script
# ---------------------- 3. 地面压力奖励（举物时压力最大化） ----------------------
def pressure_reward(foot_force, max_pressure: float = 100.0):  # 新增float类型注解
    fz = foot_force[..., 2]  # 垂直方向力（up为正）
    positive_fz = torch.clamp(fz, min=0.0)  # 只考虑向下的压力（忽略向上的力）
    normalized = positive_fz / max_pressure  # 归一化到[0,1]（超过max_pressure则饱和）
    return torch.tanh(normalized)  # 奖励在[0,1)，expect-->1
