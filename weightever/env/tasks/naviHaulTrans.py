import os
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

# import weightever.env.tasks.humanoid_amp_task as humanoid_amp_task
import weightever.env.tasks.humanoid_amp_naviHaul as humanoid_amp_naviHaul
from weightever.utils import torch_utils
import swanlab
from weightever.utils import vbox
import trimesh
from weightever.utils import rwd_manage
from weightever.utils import bbox_contact
from weightever.utils import weight_pred


from weightever.utils.cloudp import generate_box_pointcloud, compute_sdf, PointCloudVisualizer
from weightever.utils import view_force
import xml.etree.ElementTree as ET

class naviHaulTrans(humanoid_amp_naviHaul.HumanoidAMPnaviHaul):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.force_duration_counter = torch.zeros(self.num_envs).to(self.device)
        self.scheduled_forces = self._sample_force_params(self.num_envs)
        self.applied_forces = torch.zeros(self.num_envs).to(self.device)
        self.enable_random_force = cfg['args'].enable_random_force
    def pre_physics_step(self, actions):
        if self.enable_random_force:
            self._apply_random_forces()
        # 更新当前的μ向量（存储在self.current_mu[i]）
        # TODO for Z
        # self.current_mu[i, 10:13] = force_vec  # 假设μ的10-12维是力大小*方向
        # self.current_mu[i, 13:16] = force_pos  # 13-15维是力位置

        super().pre_physics_step(actions)
        return



    def _apply_random_forces(self):
        """
        根据预设的force schedule或随机触发，施加力到箱子上。
        模拟：投放（向下力）、推（横向力）等干扰。
        """
        # 预先准备力和位置的张量
        # force_tensor = gymapi.Tensor(self.sim, gymapi.DATA_TYPE_FLOAT32, (self.num_envs, 3))
        # pos_tensor = gymapi.Tensor(self.sim, gymapi.DATA_TYPE_FLOAT32, (self.num_envs, 3))
        # 采样力参数（从μ的力相关部分）
        force_mag = self.scheduled_forces['magnitude'] # N
        force_dir = self.scheduled_forces['direction']  # [x,y,z]单位向量
        force_pos = self.scheduled_forces['position']   # 相对箱子中心
        # force_vec = force_dir*force_mag.unsqueeze(-1)
        # 计算全局坐标的施力点（需要箱子当前pose）
        box_pose = self._box_states[:, :7]  # [pos(3), quat(4)]
        box_pos = box_pose[:,:3]
        # 简化：假设force_pos是局部坐标，转全局
        # global_force_pos = box_pos + force_pos  # 可以用quaternion旋转，这里简化
        global_force_pos = box_pos + \
                quat_rotate(self._box_rot, force_pos)
        # 填充数据
        for i in range(self.num_envs):
            self._should_trigger_force(i)
            # if self._should_trigger_force(i):

            #     force_tensor.set_item(i*3,     force_vec[i,0])
            #     force_tensor.set_item(i*3 + 1, force_vec[i,1])
            #     force_tensor.set_item(i*3 + 2, force_vec[i,2])

            #     pos_tensor.set_item(i*3,     global_force_pos[i,0])
            #     pos_tensor.set_item(i*3 + 1, global_force_pos[i,1])
            #     pos_tensor.set_item(i*3 + 2, global_force_pos[i,2])
            # else:
            #     force_tensor.set_item(i*3,     0.0)
            #     force_tensor.set_item(i*3 + 1, 0.0)
            #     force_tensor.set_item(i*3 + 2, 0.0)

            #     pos_tensor.set_item(i*3,     0.0)
            #     pos_tensor.set_item(i*3 + 1, 0.0)
            #     pos_tensor.set_item(i*3 + 2, 0.0)
        force_tensor = force_dir*force_mag.unsqueeze(-1)
        force_tensor *= self.applied_forces.unsqueeze(-1)
        global_force_pos *= self.applied_forces.unsqueeze(-1)
        mid = (force_tensor.repeat(1, self.num_bodies+1)).reshape(self.num_envs,self.num_bodies+1,3)
        mid[:,:-1,:]=0
        force_tensor = mid.reshape(-1,3)
        mid = (global_force_pos.repeat(1, self.num_bodies+1)).reshape(self.num_envs,self.num_bodies+1,3)
        mid[:,:-1,:]=0
        global_force_pos = mid.reshape(-1,3)

        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(force_tensor),
            # None,
            gymtorch.unwrap_tensor(global_force_pos),
            gymapi.ENV_SPACE
        )
                # self.gym.apply_rigid_body_force_at_pos_tensors(
                #     self.sim,
                #     self.envs[i],
                #     self._box_handles[i],  # 箱子的actor handle
                #     box_body_idx,    # rigid body index（通常箱子是单body，索引0）
                #     gymapi.Vec3(*force_vec.tolist()),
                #     gymapi.Vec3(*global_force_pos.tolist()),
                #     gymapi.ENV_SPACE  # 或LOCAL_SPACE，取决于坐标系
                # )
                
            #     # 记录：本step施加了力（用于观测或日志）
            #     self.applied_forces[i] = 1
            # else:
            #     self.applied_forces[i] = 0  # 无力



    def _sample_force_params(self, num_envs):
        """
        采样力参数（μ的一部分），存储到scheduled_forces。
        在reset_idx时调用，准备好待触发的力配置。
        """
        forces = {}
        
        # 力大小：0-200N（可调）
        forces['magnitude'] = torch.rand(num_envs, device=self.device) * 200.0 + 300
        
        # 力方向：随机单位向量（或固定向下/横向）
        # 示例：向下为主（模拟投放），加小噪声
        force_dir = torch.tensor([[0.0, -0.3, -1.0]], device=self.device).repeat(num_envs, 1)
        force_dir += torch.randn(num_envs, 3, device=self.device) * 0.1  # 小噪声
        force_dir = force_dir / torch.norm(force_dir, dim=1, keepdim=True)  # 归一化
        forces['direction'] = force_dir
        
        # 力应用位置：相对箱子中心（e.g., 上部=[0,0,0.2]） # TODO: other force pos
        forces['position'] = torch.tensor([[0.0, 0.0, -0.2]], device=self.device).repeat(num_envs, 1)
        
        return forces



    def _should_trigger_force(self, env_id):
        trigger_prop = 0.005
        if self.force_duration_counter[env_id] > 0:
            self.force_duration_counter[env_id] -= 1
            self.applied_forces[env_id] = 1
            return True  # 继续施加
        elif torch.rand(1).item() < trigger_prop:
            # 触发新力，设置持续时间
            self.force_duration_counter[env_id] = 90  # 30秒 # TODO: apply shorter pulse!
            self.applied_forces[env_id] = 1
            return True
        self.applied_forces[env_id] = 0
        return False
    
    def _draw_task(self):
        super()._draw_task()
        self._draw_extra_force()

    def _draw_extra_force(self):
        cols = np.array([[1.0, 0.5, 1.0]], dtype=np.float32)
        ratio=10
        # 将数据提交给仿真绘制
        for i, env_ptr in enumerate(self.envs):
            if self.applied_forces[i]>0:
                force_mag = self.scheduled_forces['magnitude'][i]  # N
                force_dir = self.scheduled_forces['direction'][i]  # [x,y,z]单位向量
                force_start = self.scheduled_forces['position'][i] + self._box_pos[i]   # 相对箱子中心
                force_end = force_start+force_dir*force_mag/ratio
                force = torch.cat([force_start, force_end], dim=-1).cpu().numpy()
                self.gym.add_lines(self.viewer, env_ptr,
                                1, force, cols)


    