import os
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from weightever.env.tasks.base_task import BaseTask

import weightever.env.tasks.humanoid_amp_task as humanoid_amp_task
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

class HumanoidAMPnaviHaul(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_envs = cfg["env"]["numEnvs"]
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.reward_fsm = rwd_manage.MultiEnvRewardStateMachine(self.num_envs, "CARRY", device_type)
        self.reward_fsm.log(cfg['args'].swanlab_name != "")
        self.weight_pred = weight_pred.WeightPred()

        self._box_dist_min = 0.5
        self._box_dist_max = 10
        self._target_dist_min = 1.5
        self._target_dist_max = 10

        # scaling object size
        self._box_min_scale = cfg['args'].box_min_size_scale #0.8
        self._box_max_scale = cfg['args'].box_max_size_scale #1.0
        self.scaling_factor = self._box_min_scale + \
            (self._box_max_scale - self._box_min_scale) * \
            torch.rand(self.num_envs)

        # scaling object weight
        self._box_min_weight_scale = cfg['args'].box_min_weight_scale
        self._box_max_weight_scale = cfg['args'].box_max_weight_scale #0.8
        self.scaling_factor_weight = self._box_min_weight_scale + \
            (self._box_max_weight_scale - self._box_min_weight_scale) * \
            torch.rand(self.num_envs)

        self._default_box_width_size = cfg['args'].box_width #0.5
        self._default_box_length_size = cfg['args'].box_length
        self._default_box_height_size = cfg['args'].box_height

        self.obs_add_noise = False
        self.noise_level = 0.0


        self._width_box_size = torch.zeros(self.num_envs).to(device)
        self._length_box_size = torch.zeros(self.num_envs).to(device)
        self._height_box_size = torch.zeros(self.num_envs).to(device)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.spacing = cfg["env"]['envSpacing']
        self.reset_time = 0

        self.log_success = False

        if cfg['env']['eval_mode']:
            # for calculate success rate and distance error and execution time
            self.log_success = True
            self._distance_to_target = [
                [] for _ in range(self.num_envs)]
            self.log_success_rate = []
            self.log_success_precision = []

        if cfg['env']['save_motions']:
            self.save_motion_for_blender = False
            self._save_all_state = True
            self.record_frame_number = self.cfg['env']['episodeLength']

            self.output_dict = {}
            self.output_dict['trans'] = np.zeros(
                [self.num_envs, 10*self.record_frame_number, 15, 3])
            self.output_dict['rot'] = np.zeros(
                [self.num_envs, 10*self.record_frame_number, 15, 4])
            self.output_dict['obj_pos'] = np.zeros(
                [self.num_envs, 10*self.record_frame_number, 3])
            self.output_dict['obj_rot'] = np.zeros(
                [self.num_envs, 10*self.record_frame_number, 4])
            self.navi_flag = np.zeros(self.num_envs)
            if self._save_all_state:
                self.output_dict['root_pos'] = np.zeros(
                    [self.num_envs, 10*self.record_frame_number, 3])
                self.output_dict['root_rot'] = np.zeros(
                    [self.num_envs, 10*self.record_frame_number, 4])
                self.output_dict['dof_pos'] = np.zeros(
                    [self.num_envs, 10*self.record_frame_number, 28])
        self.record_step = 0
        # after scaled; real len and wid 
        width_half_size = self._width_box_size / 2.0
        length_half_size = self._length_box_size / 2.0
        height_half_size = self._height_box_size / 2.0

        lfus = torch.stack(
            [-length_half_size, width_half_size, height_half_size], dim=1)
        lfds = torch.stack(
            [-length_half_size, width_half_size, -height_half_size], dim=1)
        lbus = torch.stack(
            [-length_half_size, -width_half_size, height_half_size], dim=1)
        lbds = torch.stack(
            [-length_half_size, -width_half_size, -height_half_size], dim=1)
        rfus = torch.stack(
            [length_half_size, width_half_size, height_half_size], dim=1)
        rfds = torch.stack(
            [length_half_size, width_half_size, -height_half_size], dim=1)
        rbus = torch.stack(
            [length_half_size, -width_half_size, height_half_size], dim=1)
        rbds = torch.stack(
            [length_half_size, -width_half_size, -height_half_size], dim=1)
        self.box_bps = torch.stack(
            [lfus, lfds, lbus, lbds, rfus, rfds, rbus, rbds], dim=0).transpose(0,1)




        # (x, y0, z0) # lenght_x_dir; width_y_dir

        stand_points_left = torch.stack(
            [-length_half_size - 0.35, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        stand_points_right = torch.stack(
            [length_half_size + 0.35, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        stand_points_up = torch.stack(
            [torch.zeros(self.num_envs).to(device), width_half_size+ 0.35,torch.zeros(self.num_envs).to(device)], dim=1)
        stand_points_down = torch.stack(
            [torch.zeros(self.num_envs).to(device), -width_half_size-0.35 ,torch.zeros(self.num_envs).to(device)], dim=1)
        
        self.vbox_s = width_half_size*2
        held_points_left = torch.stack(
            [-length_half_size + 1*self.vbox_s/5, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_right = torch.stack(
            [length_half_size - 1*self.vbox_s/5, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_up = torch.stack(
            [torch.zeros(self.num_envs).to(device), width_half_size- 1*self.vbox_s/5 ,torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_down = torch.stack(
            [torch.zeros(self.num_envs).to(device), -width_half_size + 1*self.vbox_s/5 ,torch.zeros(self.num_envs).to(device)], dim=1)
        
        self.stand_points_offset = torch.cat((stand_points_left, stand_points_right, stand_points_up, stand_points_down),dim=-1).reshape(self.num_envs,-1,3)
        self.held_points_offset = torch.cat((held_points_left, held_points_right, held_points_up, held_points_down),dim=-1).reshape(self.num_envs,-1,3)
        self.topKidx = torch.ones((self.num_envs,4),device=self.device,dtype=torch.float)


        # init hardcode! 
        self.vbox_bps = vbox.compute_initial_b_part_bbox(self.held_points_offset[:,0,:], self.vbox_s, self.vbox_s, self.vbox_s)

        # self.stand_held_points_offset = torch.stack(
        #     [stand_points_left, stand_points_right, held_points_left, held_points_right], dim=0)

        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_vbox_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_gp_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)

        lift_body_names = cfg["env"]["liftBodyNames"]
        self._lift_body_ids = self._build_lift_body_ids_tensor(lift_body_names)

        self._build_box_tensors()
        # self._build_platform_tensors()
        self._build_target_state_tensors()
        # self.obs_buf = {"navi_obs":
        #                 torch.zeros((self.num_envs, self.cfg['env'].numNaviObs), 
        #                             device=self.device, dtype=torch.float),
        #                 "carry_obs":
        #                 torch.zeros((self.num_envs, self.cfg['env'].numCarryObs), 
        #                             device=self.device, dtype=torch.float),}
        self.num_obs = self.cfg['env']['numObservations'] + max(self.cfg['env']['numNaviObs'],self.cfg['env']['numCarryObs']) 
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        
        self.valid_failure = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.failure_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.navi_reached_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.navi_env = torch.tensor([], dtype=torch.long, device=self.device)
        self.failure_envs = torch.tensor([], dtype=torch.long, device=self.device) 

        self.show_joint_forces = False
        self.show_contact_forces = True
        self.show_bbox = False
        self.save_images = self.cfg['env']['saveImages']

    def _build_lift_body_ids_tensor(self, lift_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in lift_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._box_handles = []
        if self.cfg["args"].asset_root == "Box":
            self._load_box_asset()
        elif self.cfg["args"].asset_root == "MRB_Box":
            self._load_target_asset_MRB()
        else:
            self._load_target_asset_SRB()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_cloudp(self):
        """获取所有环境中立方体的点云"""
        pointclouds = []
        
        for env_id in range(self.num_envs):
            box_length = self._default_box_length_size* self.scaling_factor[env_id]
            box_width = self._default_box_width_size* self.scaling_factor[env_id]
            box_height = self._default_box_height_size* self.scaling_factor[env_id]
            # 生成点云
            pointcloud = generate_box_pointcloud(
                box_length, box_width, box_height, 
                count=1024, seed=2024
            )
            pointclouds.append(pointcloud)
    
        # 可以根据需要将所有点云组合成一个张量
        return torch.stack(pointclouds).to(self.device) # nunm_env,1024,3
    
    def _load_box_asset(self):
        self.objname = []
        length_box_size = self._default_box_length_size
        width_box_size = self._default_box_width_size
        height_box_size = self._default_box_height_size

        self._box_asset = []
        self.asset_density = torch.zeros(self.num_envs).to(self.device)
        self.pointscloud = self._load_cloudp() # check: should be env,1024,3

        for env_id in range(self.num_envs):
            scaling_factor_l = self.scaling_factor[env_id]
            scaling_factor_w = self.scaling_factor[env_id]
            scaling_factor_h = self.scaling_factor[env_id]
            scaling_factor_weight = self.scaling_factor_weight[env_id]

            box_length = scaling_factor_l * length_box_size
            box_width = scaling_factor_w * width_box_size
            box_height = scaling_factor_h * height_box_size
            self.objname.append(f"Box_{box_length}_{box_width}_{box_height}_{env_id}")

            asset_options = gymapi.AssetOptions()

            asset_options.density = scaling_factor_weight * self.cfg['args'].box_density / \
                (scaling_factor_l * scaling_factor_w * scaling_factor_h)

            self.asset_density[env_id] = asset_options.density

            self._box_asset.append(self.gym.create_box(
                self.sim, box_length, box_width, box_height, asset_options))

        return

    def _load_target_asset_MRB(self):
        asset_root = "Box_test5"
        asset_root = "/mnt/cwever202/vData/Split/Cube_t3"
        # asset_root = "/mnt/cwever202/vData/exp1/exp1_cube45"
        asset_root = "/mnt/cwever202/vData/exp6/exp6_long35_grad"


        self.obj_info_input = []
        obj_info_input_key = ["com_offset","I_com","I_principal","I_rot_quat_xyzw","total_mass"]

        self.objpath = []
        self._target_asset = []
        points_num = []
        self.object_points = []
        self.aabb=[]
        object_names = ["boxes"] #'d3largebox'
        self.objname = []

        params_path = os.path.join(asset_root, "all_objects_params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"参数文件不存在: {params_path}")
        with open(params_path, "r", encoding="utf-8") as f:
            obj_info = json.load(f)
        s = np.random.randint(0,len(obj_info))
        
        for i, env_id in enumerate(range(self.num_envs)): # check file here # only display the integral obj
            # object_name = object_names[i%len(object_names)]
            object_name = obj_info[(s+i)%len(obj_info)]['object_id']
            # 假设 obj_info 是包含多个字典的列表，每个字典的值 p 是列表
            # obj_info_input_key 是需要保留的 key 列表
            obj_info_input = [
                item  # 最终添加的元素（可能是列表中的元素，或单个float）
                for k, p in obj_info[(s + i) % len(obj_info)].items()  # 遍历key-value
                if k in obj_info_input_key  # 过滤key
                # 内层循环：若p是列表则遍历元素，否则直接遍历单个元素（p本身）
                for item in (p if isinstance(p, (list, tuple)) else [p])
            ]
            self.obj_info_input.append(obj_info_input)
            self.objname.append(object_name)
            asset_file_root = os.path.join(asset_root, obj_info[(s+i)%len(obj_info)]['mass_distribution'])
            self.asset_file = object_name + ".urdf"
            # obj_file = asset_root + "/"+ object_name + '/' + object_name + '.obj'
            self.objpath.append(os.path.join(asset_file_root,self.asset_file))

        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = self.cfg['env']['env_rand'][env_id]['object']["angular_damping"]
            asset_options.linear_damping = self.cfg['env']['env_rand'][env_id]['object']["linear_damping"]

            asset_options.fix_base_link = False  # 让物体可以自由移动
            # asset_options.flip_visual_attachments = False
            # asset_options.use_mesh_materials = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            # asset_options.vhacd_enabled = True
            # max_convex_hulls = 64
            # asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            # asset_options.vhacd_params.max_num_vertices_per_ch = 64
            # asset_options.vhacd_params.resolution = 300000


            self._target_asset.append(self.gym.load_asset(self.sim, asset_file_root, self.asset_file, asset_options))

            """            
            mesh_obj = trimesh.load(obj_file, force='mesh')
            aabb = mesh_obj.bounding_box
            aabb = aabb.bounds
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            while object_points.shape[0] < 1024:
            object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            """

            aabb, object_points = get_full_box_info_from_urdf(asset_file_root+'/'+self.asset_file, sample_points=1024)
            

            self.aabb.append(np.array(aabb))
            # self.object_points.append(to_torch(object_points))
            self.object_points.append(torch.tensor(object_points,dtype=torch.float).to(self.device))
#
        view_cloud_point = False
        if view_cloud_point:
            visualizer = PointCloudVisualizer()
            visualizer.set_data(object_points.cpu().numpy())
            visualizer.visualize(title="point cloud", point_size=10)
            # self.num_obj_bodies.append(self.gym.get_asset_rigid_body_count(self._target_asset[-1]))
            # self.num_obj_shapes.append(self.gym.get_asset_rigid_shape_count(self._target_asset[-1]))

        self.pointscloud = torch.stack(self.object_points, dim=0)
        self.obj_info_input = torch.tensor(self.obj_info_input).to(self.device)
        return
    def _load_target_asset_SRB(self): # smplx
        # asset_root = self.cfg["env"]["asset"]["assetRoot"]+"/objects/"
        asset_root = self.cfg["args"].asset_root
        self._target_asset = []
        points_num = []
        self.object_points = []
        self.aabb=[]
        # for i, object_name in enumerate(self.object_name):
        allobjects = ["xiaopapa","tangyi","longsofa","halflongsofa"]
        allobjects = ["demo"]
        self.objname= []
        CLOUD_SUFFIX = "_cloudpoints.npz"
        for idx, env_id in enumerate(range(self.num_envs)):
            objname = allobjects[idx%len(allobjects)]
            self.asset_file = objname +".urdf"
            obj_file = asset_root + "objects/"+objname+"/"+objname+".obj"
            cloud_file = os.path.join(os.path.dirname(obj_file), f"{objname}{CLOUD_SUFFIX}")
            self.objname.append(objname)

            max_convex_hulls = 64
            # density = self.cfg['args'].box_density
        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            # asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000

            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, self.asset_file, asset_options))

            mesh_obj = trimesh.load(obj_file, force='mesh')
            aabb = mesh_obj.bounding_box
            self.aabb.append(aabb.bounds)

            # obj_verts = mesh_obj.vertices
            # center = np.mean(obj_verts, 0)
            # object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            # object_points = to_torch(object_points - center)
            # while object_points.shape[0] < 1024:
            #     object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            # self.object_points.append(to_torch(object_points))

            # 检查点云缓存文件是否存在
            if os.path.exists(cloud_file):
                # 读取已保存的点云
                try:
                    data = np.load(cloud_file)
                    object_points = to_torch(data['points'])
                    print(f"Loaded cached cloudpoints for {objname} from {cloud_file}")
                except (IOError, KeyError) as e:
                    print(f"Failed to load cloudpoints from {cloud_file}, regenerating. Error: {e}")
                    # 加载失败时重新生成
                    need_regenerate = True
            else:
                obj_verts = mesh_obj.vertices
                center = np.mean(obj_verts, 0)
                object_points, object_faces = trimesh.sample.sample_surface_even(
                    mesh_obj, count=1024, seed=2024
                )
                # 中心化处理
                object_points = to_torch(object_points - center)
                # 确保点数量为1024
                while object_points.shape[0] < 1024:
                    object_points = torch.cat([
                        object_points, 
                        object_points[:1024 - object_points.shape[0]]
                    ], dim=0)
            
                # 保存点云到缓存文件
                try:
                    np.savez(cloud_file, points=object_points.cpu().numpy())
                    print(f"Saved cloudpoints for {objname} to {cloud_file}")
                except IOError as e:
                    print(f"Failed to save cloudpoints to {cloud_file}. Error: {e}")
            self.object_points.append(to_torch(object_points))






        # self.num_obj_bodies.append(self.gym.get_asset_rigid_body_count(self._target_asset[-1]))
        # self.num_obj_shapes.append(self.gym.get_asset_rigid_shape_count(self._target_asset[-1]))

        self.pointscloud = torch.stack(self.object_points, dim=0)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self.cfg["args"].asset_root=="Box":
            self._build_box(env_id, env_ptr)
        elif self.cfg["args"].asset_root=="MRB_Box":
            self._bulid_asset_MRB(env_id, env_ptr)
        else:
            self._bulid_asset_SRB(env_id, env_ptr)

        # self._create_ground_plane()
        if self.cfg['args'].random_env!="None":
            tile_thickness = 0.1 # 111
            tile_size_x = 2*self.cfg["env"]['envSpacing'] * 0.98
            tile_size_y = 2*self.cfg["env"]['envSpacing'] * 0.98
            gasset_options =  gymapi.AssetOptions()
            gasset_options.fix_base_link = True
            ground_asset = self.gym.create_box(self.sim, tile_size_x, tile_size_y, tile_thickness,gasset_options)
            self.ground_actors=[]

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, - tile_thickness * 0.5)  # 顶面在 z=0
            # pose.r = gymapi.Quat(0.0, 0.2588, 0.0, 0.9659) # friction check
            ground_actor = self.gym.create_actor(env_ptr, ground_asset, pose, f"ground_{env_id}", env_id, 0)
            
            MU_MIN, MU_MAX = 0.0, 0.5  # 按你的阶段配置替换（例如阶段1: 0.8–1.2；阶段3: 0.1–2.0）
            set_ground_color_for_env(self.gym, env_ptr, ground_actor,
                                    mu_value=self.cfg['env']['env_rand'][env_id]['ground']["friction"],
                                    mu_min=MU_MIN, mu_max=MU_MAX)
            # 3.2) 设置该地砖的接触摩擦/恢复系数
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, ground_actor)
            # 从你的参数库里取对应 env 的 ground 配置
            for sp in shape_props:
                # 许多版本只有一个 friction 字段；plane 的 static/dynamic 是特殊接口
                # 这里用 static 作为 shape 的摩擦基值（或取 dynamic，二者差别通常不大）
                sp.friction = self.cfg['env']['env_rand'][env_id]['ground']["friction"]
                sp.rolling_friction = self.cfg['env']['env_rand'][env_id]['ground']["rolling_friction"] # fixed
                sp.torsion_friction = self.cfg['env']['env_rand'][env_id]['ground']["torsion_friction"] # fixed
                sp.restitution = self.cfg['env']['env_rand'][env_id]['ground']["restitution"]
                sp.friction = 0.8
                sp.rolling_friction = 0
                sp.torsion_friction = 0
                sp.restitution = 0.2
                print(sp.rolling_friction, sp.torsion_friction, sp.restitution,sp.friction)

        #         # 可选：组合模式（不同 API 版本可能暴露 combine mode）
        #         # sp.frictionCombineMode = gymapi.COMBINE_MULTIPLY / AVERAGE / MIN / MAX
        #     self.gym.set_actor_rigid_shape_properties(env_ptr, ground_actor, shape_props)
        #     rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, ground_actor)
        #     for p in rb_props:
        #         p.mass = 1
        #         # p.flags |= gymapi.RIGID_BODY_DISABLE_GRAVITY
        #     self.gym.set_actor_rigid_body_properties(env_ptr, ground_actor, rb_props, True)


 
            self.ground_actors.append(ground_actor)

        return

    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 3.0

        scaling_factor_l = self.scaling_factor[env_id]
        scaling_factor_w = self.scaling_factor[env_id]
        scaling_factor_h = self.scaling_factor[env_id]

        self._width_box_size[env_id] = scaling_factor_w * \
            self._default_box_width_size
        self._length_box_size[env_id] = scaling_factor_l * \
            self._default_box_length_size
        self._height_box_size[env_id] = scaling_factor_h * \
            self._default_box_height_size

        box_handle = self.gym.create_actor(
            env_ptr, self._box_asset[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        props = self.gym.get_actor_dof_properties(env_ptr, box_handle)
        props['friction'].fill(5.0)
        self.gym.set_actor_dof_properties(env_ptr, box_handle, props)
        self._box_handles.append(box_handle)
        return
    def _bulid_asset_SRB(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        self.object_name = ["myasset"]*len(self._target_asset) # hardcode!
        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id], default_pose, self.object_name[env_id], col_group, col_filter, segmentation_id)
        self.gym.set_actor_scale(env_ptr, target_handle, self.scaling_factor[env_id])

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.3
            props[p_idx].friction = 0.8
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.8
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        rigid_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle) # hardcode here
        assert self.gym.get_actor_rigid_body_count(env_ptr, target_handle) == 1
        rigid_body_props[0].mass = 40
        # check if you need recompute inertia!!!
        success = self.gym.set_actor_rigid_body_properties(
            env_ptr, 
            target_handle, 
            rigid_body_props, 
            recomputeInertia=True  # 质量变化时建议开启，自动更新惯性
        )


        self._box_handles.append(target_handle)

        # self._target_asset[env_id] = 
        # set bbox len,wid,hei
        xyz = self.aabb[env_id][1]-self.aabb[env_id][0]
        self._length_box_size[env_id] = xyz[0]
        self._width_box_size[env_id] = xyz[1]
        self._height_box_size[env_id] = xyz[2]

        # self._width_box_size[env_id] = scaling_factor_w * xyz[0]
        # self._length_box_size[env_id] = scaling_factor_l * xyz[1]
        # self._height_box_size[env_id] = scaling_factor_h * xyz[2]


    def _bulid_asset_MRB(self, env_id, env_ptr):
        col_group = env_id
        # col_group = 0
        col_filter = 0
        segmentation_id = 0
        object_name = f"MRB{env_id}" # hardcode!
        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(0.0, 0.0, 2.2)  # 放置在地面上方
        default_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id], default_pose, object_name, col_group, col_filter, segmentation_id)
        
        self.gym.set_actor_scale(env_ptr, target_handle, self.scaling_factor[env_id])


        # set shape properties of the target actor       
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = self.cfg['env']['env_rand'][env_id]['object']["restitution"] # 0.3
            props[p_idx].friction = self.cfg['env']['env_rand'][env_id]['object']["friction"] # 0.8
            props[p_idx].rolling_friction = self.cfg['env']['env_rand'][env_id]['object']["rolling_friction"] # 0.01
            props[p_idx].torsion_friction = self.cfg['env']['env_rand'][env_id]['object']["torsion_friction"] # 0.8
            props[p_idx].restitution = 0.2
            props[p_idx].friction = 0.5
            props[p_idx].rolling_friction = 0.0
            props[p_idx].torsion_friction = 0.0

        print("retitution",props[0].restitution,"friction",props[0].friction,"rolling_friction",props[0].rolling_friction,"torsion_friction",props[0].torsion_friction)
        success = self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props )
        if success:
            print("Successfully updated rigid body properties[physic]")
        

        # # set color properties of the target actor
        # color set(correspond to weight)
        
        rigid_handle_num = self.gym.get_actor_rigid_body_count(env_ptr, target_handle) #hardcode here
        target_masses = [0.1 for i in range(rigid_handle_num)] #hardcode here!
        assert rigid_handle_num == len(target_masses), "The number of rigid bodies should match the number of target masses"
        body_names = self.gym.get_actor_rigid_body_names(env_ptr, target_handle) # zero actor handle in an env # hardcode here
        
        rigid_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle) # hardcode here
        for i in range(rigid_handle_num):
            body_name = body_names[i]
            # if i == 0: 
            #     assert "base" in body_name#"The first rigid body should be the base link"
            #     continue
            mass_coff = max(1 - target_masses[i] / max(target_masses), 0.001) # normalize the mass to sum to 1
            r = (0.0 + 200.0 * mass_coff ) / 255.0
            g = mass_coff  * 200.0/255.0
            b = mass_coff * 200.0/255.0

            # b = i/rigid_handle_num  * 200.0/255.0
            # self.gym.set_rigid_body_color(env_ptr, target_handle, i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(r, g, b))
            
            rigid_body_props[i].mass = target_masses[i]
            print(f"Set {body_name} color and mass({rigid_body_props[i].mass})")

        # check if you need recompute inertia!!!
        # success = self.gym.set_actor_rigid_body_properties(
        #     env_ptr, 
        #     target_handle, 
        #     rigid_body_props, 
        #     recomputeInertia=True  # 质量变化时建议开启，自动更新惯性
        # )
        # if success:
        #     print("Successfully updated rigid body properties[color]")


        # self._target_handles.append(target_handle)
        self._box_handles.append(target_handle)

        # bbox and scale bbox
        xyz = self.aabb[env_id][1]-self.aabb[env_id][0]
        self._length_box_size[env_id] = xyz[0]*self.scaling_factor[env_id]
        self._width_box_size[env_id] = xyz[1]*self.scaling_factor[env_id]
        self._height_box_size[env_id] = xyz[2]*self.scaling_factor[env_id]

        # point cloud and scale point cloud
        # there may be bug here
        assert max(self.pointscloud[0,:,0])+min(self.pointscloud[0,:,0]) < 0.01
        assert max(self.pointscloud[0,:,1])+min(self.pointscloud[0,:,1]) < 0.01
        assert max(self.pointscloud[0,:,2])+min(self.pointscloud[0,:,2]) < 0.01
        self.pointscloud[env_id] *= self.scaling_factor[env_id]

        return
    
    def _build_target_state_tensors(self):
        self._target_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self._target_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self._const_target_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self._const_target_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self.tar_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.tar_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        return
    # def _build_asset_tensors(self):
    #     num_actors = self.get_num_actors_per_env()

    #     self._box_states = self._root_states.view(
    #         self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
    #     self.box_standing_points = torch.zeros(
    #         self.num_envs, 3).to(self.device)
    #     self.box_held_points = torch.zeros(
    #         self.num_envs, 3).to(self.device)
        
    #     self.ground_points_offset = torch.zeros(
    #         self.num_envs, 2, 3).to(self.device) # actually 2 points of the bps
    #     self.tar_ground_points = torch.zeros(
    #         self.num_envs, 3).to(self.device)
    #     self.box_ground_points = torch.zeros(
    #         self.num_envs, 3).to(self.device)
        
    #     self._init_box_height = torch.zeros(
    #         self.num_envs).to(self.device)
    #     self._box_actor_ids = to_torch(
    #         num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
    #     self._box_pos = self._box_states[..., :3]
    #     bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
    #     contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
    #         self.sim)
    #     contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
    #     self._box_contact_forces = contact_force_tensor.view(
    #         self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :] # all 64 parts of the box

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._box_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self.box_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        
        self.ground_points_offset = torch.zeros(
            self.num_envs, 2, 3).to(self.device) # actually 2 points of the bps
        self.tar_ground_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_ground_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        
        self._init_box_height = torch.zeros(
            self.num_envs).to(self.device)
        self._box_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        self._box_pos = self._box_states[..., :3]
        self._box_rot = self._box_states[..., 3:7]

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.contact_force_tensor = gymtorch.wrap_tensor(self.contact_force_tensor)
        self._box_contact_forces = self.contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

    def _reset_actors(self, env_ids, randomize=True):
        super()._reset_actors(env_ids)

        if self.headless:
            self._reset_box(env_ids, randomize=randomize)
            self._reset_target(env_ids, randomize=randomize)
        else:
            self._reset_box(env_ids, randomize=randomize)
            self._reset_target(env_ids, randomize=randomize)
        # self._reset_platform(env_ids, randomize=True)
        return

    def _reset_box(self, env_ids, randomize=True):

        n = len(env_ids)
        # randomize = False
        if randomize:
            rand_dist = (self._box_dist_max - self._box_dist_min) * torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device) + self._box_dist_min
            random_numbers = torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device)
            random_numbers_rot = torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device)

            rand_theta = 2 * np.pi * random_numbers

            self._box_states[env_ids, 0] = rand_dist * \
                torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
            self._box_states[env_ids, 1] = rand_dist * \
                torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
            self._box_states[env_ids, 2] = self._height_box_size[env_ids] / 2.0 + 0.00000

            rand_rot_theta = 2 * np.pi * random_numbers_rot
            axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
            rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
            self._box_states[env_ids, 3:7] = rand_rot
            self._box_states[env_ids, 7:] = 0.0

        else:
            self._box_states[env_ids,
                             0] = self._humanoid_root_states[env_ids, 0] + 1.0
            self._box_states[env_ids,
                             1] = self._humanoid_root_states[env_ids, 1] + 1.0
            self._box_states[env_ids, 2] = self._height_box_size[env_ids] / 2.0 + 0.00000
            
            self._box_states[env_ids, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
            self._box_states[env_ids, 7:] = 0.0
            random_numbers_rot = torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device)

            rand_rot_theta = 2 * np.pi * random_numbers_rot
            rand_rot_theta *= 0.0
            axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
            rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
            self._box_states[env_ids, 3:7] = rand_rot
            self._box_states[env_ids, 7:] = 0.0
        return

    def _reset_target(self, env_ids, randomize=True):
        n = len(env_ids)
        randomize = False
        if randomize:
            rand_dist = (self._target_dist_max - self._target_dist_min) * torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device) + self._target_dist_min
            random_numbers = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)
            random_numbers_rot = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)

            rand_theta = 2 * np.pi * random_numbers
            self._target_pos[env_ids, 0] = rand_dist * \
                torch.cos(rand_theta) + self._box_states[env_ids, 0]
            self._target_pos[env_ids, 1] = rand_dist * \
                torch.sin(rand_theta) + self._box_states[env_ids, 1]
            self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0 + 0.00000
            
            rand_rot_theta = 2 * np.pi * random_numbers_rot
            axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
            rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
            self._target_rot[env_ids] = rand_rot
            # check whether it changes with origin
            self._const_target_rot[env_ids] = self._target_rot[env_ids]
            self._const_target_pos[env_ids] = self._target_pos[env_ids]
        else:
            self._target_pos[env_ids,
                             0] = self._box_states[env_ids, 0] + 0.0
            self._target_pos[env_ids,
                             1] = self._box_states[env_ids, 1] + 3.0
            self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0 + 0.00000
            self._target_rot[env_ids] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
        
            self._const_target_rot[env_ids] = self._target_rot[env_ids]
            self._const_target_pos[env_ids] = self._target_pos[env_ids]

        return
    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        box_env_ids_int32 = self._box_actor_ids[env_ids]

        env_ids_int32 = torch.cat((env_ids_int32,box_env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self._root_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # dof_pos = self._dof_state[..., :, 0]
        # dof_pos = dof_pos.contiguous()
        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                               gymtorch.unwrap_tensor(dof_pos),
        #                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0


    def _reset_tar2putdown(self, failure_envs):
        self._target_pos[failure_envs] = self._box_pos[failure_envs]
        self._target_pos[failure_envs,2] = self._init_box_height[failure_envs]
        # check _const not change; rot batch right
        self._target_rot[failure_envs] = extract_z_rotation_torch_batch(self._box_rot[failure_envs])

    def _reset_putdown2tar(self, boxdown_envids):
        self._target_pos[boxdown_envids] = self._const_target_pos[boxdown_envids]
        # check _const not change; rot batch right
        self._target_rot[boxdown_envids] = self._const_target_rot[boxdown_envids]
        

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        self._prev_vbox_pos[:] = self.box_held_points[:]
        self._prev_gp_pos[:] = self.box_ground_points[:]
        return


    def update_standing_and_held_points(self,root_states,  box_states, tar_pos, tar_rot, env_ids=None, extra_task = None):
        """
            if env specified, update offset!
        """
        if env_ids is None:
            env_ids = torch.arange(len(box_states), device=box_states.device)

            box_pos = box_states[env_ids, 0:3]
            box_rot = box_states[env_ids, 3:7]
            tar_pos = tar_pos[env_ids]
            tar_rot = tar_rot[env_ids]

            #TODO: should update or parse in stand point and held point offset judgement here


            # hardcode: set standing points to the left
            # topKidx = 0
            # self.stand_points_offset should be (envs,topK,3)
            # (env,3) = (env,3)+(env,3)
            self.box_standing_points[env_ids] = box_pos + \
                quat_rotate(box_rot, self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            self.box_standing_points[..., 2] = 0.0
            self.box_held_points[env_ids] = box_pos + \
                quat_rotate(box_rot, self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            # hardcode: set standing points to the left
            self.tar_standing_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            self.tar_held_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])


            # check this!
            # update ground_point of the box
            stand_offset = self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:]

            judge = _compute_gpoint_idx(stand_offset, self.box_bps[env_ids])
            row_indices = env_ids.unsqueeze(1).repeat(1, 2)  # 形状：(env, 2)
            self.ground_points_offset[env_ids] = self.box_bps[row_indices, judge] # num_env, 3
            # all

            gp = torch.mean(self.ground_points_offset[env_ids],dim=-2)
            
            self.box_ground_points[env_ids] = box_pos + \
                quat_rotate(box_rot, gp)
            # hardcode: set standing points to the left
            self.tar_ground_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, gp)
            



        else:
            box_pos = box_states[env_ids, 0:3]
            box_rot = box_states[env_ids, 3:7]
            root_pos = root_states[env_ids,0:3]
            root_pos[...,2] = 0.0 # check whether root_states changes
            tar_pos = tar_pos[env_ids]
            tar_rot = tar_rot[env_ids]


            #TODO: should update or parse in stand point and held point offset judgement here
            # there may be bug here!! 
            num_candidates = self.stand_points_offset.shape[1]
            # check: transpose???
            standp = box_pos.unsqueeze(1).repeat(1, num_candidates, 1).reshape(-1,3) + quat_rotate(
                    box_rot.unsqueeze(1).repeat(1, num_candidates, 1).reshape(-1,4),  
                    self.stand_points_offset[env_ids].reshape(-1,3))  
            standp = standp.reshape(-1,num_candidates,3) # env, num_candidate, xyz  
            dist = torch.norm(
                    root_pos.unsqueeze(1)[..., :2] - standp[..., :2],  # 只比较XY坐标
                    dim=-1
                )
            # idx 形状：[num_envs]，每个元素是该环境的最佳候选点索引（0~num_candidates-1）
            # select the nearest point when train
            assert sum(self.topKidx[env_ids,0] == 1.0)==len(env_ids)
            self.topKidx[env_ids,1] = torch.argmin(dist, dim=-1).float() # should be (num_env, 1)
            self.topKidx[env_ids,2] *= 0.0



            if "base train"=="no base train": # random selct 2 side 
                group_idx = torch.randint(low=0, high=4, size=(len(env_ids),), device=self.device).to(torch.float)
                self.topKidx[env_ids,1] = group_idx/2
                self.topKidx[env_ids,3] = 0
                self.topKidx[env_ids,2] = 0 
            if "determine"=="determine": # selct 1 side 
                # group_idx = torch.randint(low=0, high=0, size=(len(env_ids),), device=self.device).to(torch.float)
                self.topKidx[env_ids,1] = 0
                self.topKidx[env_ids,3] = 0
                self.topKidx[env_ids,2] = 0 

            #fisrt time: always the nearest point
            self.box_standing_points[env_ids] = box_pos + \
                quat_rotate(box_rot, self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            self.box_standing_points[env_ids, 2] = 0.0
            self.box_held_points[env_ids] = box_pos + \
                quat_rotate(box_rot, self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            # hardcode: set standing points to the left
            self.tar_standing_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])
            self.tar_held_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:])


            # check!!!
            # the corresponding vbox bbox
            held_offset = self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:]
            self.vbox_bps[env_ids] = vbox.compute_initial_b_part_bbox(held_offset, self.vbox_s[env_ids], self.vbox_s[env_ids], self.vbox_s[env_ids])

            
            # update ground_point of the box
            stand_offset = self.stand_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.stand_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:]
            judge = _compute_gpoint_idx(stand_offset, self.box_bps[env_ids])
            row_indices = env_ids.unsqueeze(1).repeat(1, 2)  # 形状：(env, 2)
            self.ground_points_offset[env_ids] = self.box_bps[row_indices, judge] # num_env, 3

            gp = torch.mean(self.ground_points_offset[env_ids],dim=-2)
            self.box_ground_points[env_ids] = box_pos + \
                quat_rotate(box_rot, gp)
            # hardcode: set standing points to the left
            self.tar_ground_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, gp)
            
            # record the height of the box
            self._init_box_height[env_ids] = self.box_held_points[env_ids,2]

        if extra_task == "vbox":
            held_offset = self.held_points_offset[env_ids, self.topKidx[env_ids,1].to(torch.long),:]+self.topKidx[env_ids,2].reshape(-1,1)*self.held_points_offset[env_ids, self.topKidx[env_ids,3].to(torch.long),:]
            self.vbox_bps[env_ids] = vbox.compute_initial_b_part_bbox(held_offset, self.vbox_s[env_ids], self.vbox_s[env_ids], self.vbox_s[env_ids])


        return
    
    # def update_standing_and_held_points(self,root_states, box_states, tar_pos, tar_rot, env_ids=None):
    #     if env_ids is None:
    #         box_pos = box_states[..., 0:3]
    #         box_rot = box_states[..., 3:7]
    #         root_pos[...] = root_states[...,0:3]
    #         root_pos[...,2] = 0.0
    #         env_indices = torch.arange(len(box_pos), device=box_pos.device)


    #         # num_candidates = self.stand_points_offset.shape[1]
    #         # standp = box_pos.unsqueeze(0).repeat(num_candidates, 1, 1).transpose(0,1).reshape(-1,3) + quat_rotate(
    #         #         box_rot.unsqueeze(0).repeat(num_candidates, 1, 1).transpose(0,1).reshape(-1,4),  
    #         #         self.stand_points_offset.reshape(-1,3))  
    #         # standp = standp.reshape(-1,num_candidates,3).transpose(0,1)  # num_candidate, env, xyz  
    #         # dist = torch.norm(
    #         #         root_pos.unsqueeze(0)[..., :2] - standp[..., :2],  # 只比较XY坐标
    #         #         dim=-1
    #         #     )

    #         # 3. 对每个环境，找到距离最近的候选点索引
    #         # idx 形状：[num_envs]，每个元素是该环境的最佳候选点索引（0~num_candidates-1）
    #         # select the nearest point when train
    #         # topKidx = torch.argmin(dist, dim=0)  # 在候选点维度（dim=0）上取最小值索引
    #         topKidx = self.topKidx
    #         # self.box_standing_points = standp[topKidx, env_indices]  # 形状：[num_envs, 3]
    #         # self.box_standing_points[..., 2] = 0.0
    #         assert sum(sum((self.held_points_offset[env_indices, topKidx,:]*self.stand_points_offset[env_indices,topKidx,:])<0))==0

    #         self.box_standing_points[:] = box_pos + \
    #             quat_rotate(box_rot, self.stand_points_offset[env_indices,topKidx,:])
    #         self.box_standing_points[..., 2] = 0.0
    #         self.box_held_points = box_pos + \
    #             quat_rotate(box_rot, self.held_points_offset[env_indices, topKidx,:])
    #         # hardcode: set standing points to the left
    #         self.tar_standing_points = tar_pos + \
    #             quat_rotate(tar_rot, self.stand_points_offset[env_indices, topKidx,:])
    #         self.tar_held_points = tar_pos + \
    #             quat_rotate(tar_rot, self.held_points_offset[env_indices, topKidx,:])


    #         # update ground_point of the box
    #         judge = _compute_gpoint_idx(self.stand_points_offset[env_indices,topKidx,:], self.box_bps[env_indices])
    #         row_indices = env_indices.unsqueeze(1).repeat(1, 2)  # 形状：(env, 2)
    #         self.ground_points_offset[env_indices] = self.box_bps[row_indices, judge] # num_env, 3
    #         # all

    #         gp = torch.mean(self.ground_points_offset[env_indices],dim=-2)
            
    #         self.box_ground_points[env_indices] = box_pos + \
    #             quat_rotate(box_rot, gp)
    #         # hardcode: set standing points to the left
    #         self.tar_ground_points[env_indices] = tar_pos + \
    #             quat_rotate(tar_rot, gp)


    # #         topKidx = 0
    # #         self.box_standing_points[:] = box_pos + \
    # #             quat_rotate(box_rot, self.stand_points_offset[:,topKidx,:])
    # #         self.box_standing_points[..., 2] = 0.0
    # #         self.box_held_points[:] = box_pos + \
    # #             quat_rotate(box_rot, self.held_points_offset[:,topKidx,:])

    # #         self.tar_standing_points[:] = tar_pos + \
    # #             quat_rotate(tar_rot, self.stand_points_offset[:,topKidx,:])
    # #         self.tar_held_points[:] = tar_pos + \
    # #             quat_rotate(tar_rot, self.held_points_offset[:,topKidx,:])

    #     else: # init!!
    #         box_pos = box_states[env_ids, 0:3]
    #         box_rot = box_states[env_ids, 3:7]
    #         tar_pos = tar_pos[env_ids]
    #         tar_rot = tar_rot[env_ids]
    #         root_pos = root_states[env_ids,0:3]
    #         root_pos[...,2] = 0.0

    #         # there may be bug here!! 
    #         num_candidates = self.stand_points_offset.shape[1]

    #         standp = box_pos.unsqueeze(0).repeat(num_candidates, 1, 1).transpose(0,1).reshape(-1,3) + quat_rotate(
    #                 box_rot.unsqueeze(0).repeat(num_candidates, 1, 1).transpose(0,1).reshape(-1,4),  
    #                 self.stand_points_offset[env_ids].reshape(-1,3))  
    #         standp = standp.reshape(-1,num_candidates,3).transpose(0,1)  # num_candidate, env, xyz  
    #         dist = torch.norm(
    #                 root_pos.unsqueeze(0)[..., :2] - standp[..., :2],  # 只比较XY坐标
    #                 dim=-1
    #             )

    #         # 3. 对每个环境，找到距离最近的候选点索引
    #         # idx 形状：[num_envs]，每个元素是该环境的最佳候选点索引（0~num_candidates-1）
    #         # select the nearest point when train
    #         self.topKidx[env_ids] = torch.argmin(dist, dim=0)  # 在候选点维度（dim=0）上取最小值索引
    #         topKidx = self.topKidx[env_ids]
    #         # env_indices = torch.arange(len(env_ids), device=dist.device)
    #         # self.box_standing_points[env_ids] = standp[topKidx, env_indices]  # 形状：[num_envs, 3]
    #         # self.box_standing_points[env_ids, 2] = 0.0

    #         self.box_standing_points[env_ids] = box_pos + \
    #             quat_rotate(box_rot, self.stand_points_offset[env_ids,topKidx,:])
    #         self.box_standing_points[env_ids, 2] = 0.0

    #         assert sum(sum((self.held_points_offset[env_ids, topKidx,:]*self.stand_points_offset[env_ids,topKidx,:])<0))==0
    #         self.box_held_points[env_ids] = box_pos + \
    #             quat_rotate(box_rot, self.held_points_offset[env_ids, topKidx,:])
    #         # hardcode: set standing points to the left
    #         self.tar_standing_points[env_ids] = tar_pos + \
    #             quat_rotate(tar_rot, self.stand_points_offset[env_ids, topKidx,:])
    #         self.tar_held_points[env_ids] = tar_pos + \
    #             quat_rotate(tar_rot, self.held_points_offset[env_ids, topKidx,:])
    #         # the corresponding vbox bbox
    #         self.vbox_bps[env_ids] = vbox.compute_initial_b_part_bbox(self.held_points_offset[env_ids, topKidx,:], self.vbox_s[env_ids], self.vbox_s[env_ids], self.vbox_s[env_ids])

            
    #         # update ground_point of the box
    #         judge = _compute_gpoint_idx(self.stand_points_offset[env_ids,topKidx,:], self.box_bps[env_ids])
    #         row_indices = env_ids.unsqueeze(1).repeat(1, 2)  # 形状：(env, 2)
    #         self.ground_points_offset[env_ids] = self.box_bps[row_indices, judge] # num_env, 3

    #         gp = torch.mean(self.ground_points_offset[env_ids],dim=-2)
    #         self.box_ground_points[env_ids] = box_pos + \
    #             quat_rotate(box_rot, gp)
    #         # hardcode: set standing points to the left
    #         self.tar_ground_points[env_ids] = tar_pos + \
    #             quat_rotate(tar_rot, gp)
            
    #         # record the height of the box
    #         self._init_box_height[env_ids] = self.box_held_points[env_ids,2].clone()
    #     return
    
    # def update_standing_and_held_points(self, box_states, tar_pos, tar_rot, env_ids=None):
    #     if env_ids is None:
            # box_pos = box_states[..., 0:3]
            # box_rot = box_states[..., 3:7]


            # topKidx = 0
            # self.box_standing_points[:] = box_pos + \
            #     quat_rotate(box_rot, self.stand_points_offset[:,topKidx,:])
            # self.box_standing_points[..., 2] = 0.0
            # self.box_held_points[:] = box_pos + \
            #     quat_rotate(box_rot, self.held_points_offset[:,topKidx,:])

            # self.tar_standing_points[:] = tar_pos + \
            #     quat_rotate(tar_rot, self.stand_points_offset[:,topKidx,:])
            # self.tar_held_points[:] = tar_pos + \
            #     quat_rotate(tar_rot, self.held_points_offset[:,topKidx,:])
    #     else:
    #         box_pos = box_states[env_ids, 0:3]
    #         box_rot = box_states[env_ids, 3:7]
    #         tar_pos = tar_pos[env_ids]
    #         tar_rot = tar_rot[env_ids]
    #         # there may be bug here!! 

    #         topKidx = 0 
    #         self.box_standing_points[env_ids] = box_pos + \
    #             quat_rotate(box_rot, self.stand_points_offset[env_ids, topKidx,:])
    #         self.box_standing_points[env_ids, 2] = 0.0
    #         self.box_held_points[env_ids] = box_pos + \
    #             quat_rotate(box_rot, self.held_points_offset[env_ids, topKidx,:])
    #         # hardcode: set standing points to the left
    #         self.tar_standing_points[env_ids] = tar_pos + \
    #             quat_rotate(tar_rot, self.stand_points_offset[env_ids, topKidx,:])
    #         self.tar_held_points[env_ids] = tar_pos + \
    #             quat_rotate(tar_rot, self.held_points_offset[env_ids, topKidx,:])
    #     return
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            # assert (env_ids==None or len(env_ids.shape)==1)

            carry_obs, navi_obs = self._compute_task_obs(env_ids)
            # 处理carry_obs：拼接humanoid_obs并填充到非导航环境（仅当非空时）
            # if carry_obs.numel() > 0:  # 检查carry_obs非空（元素数>0）
            carry_obs = torch.cat([humanoid_obs, carry_obs], dim=-1)
            pad_size = self.obs_buf.shape[-1] - carry_obs.shape[-1]
            if pad_size > 0:
                carry_obs = torch.nn.functional.pad(
                    carry_obs, 
                    pad=(0, pad_size),  
                    mode='constant', 
                    value=0.0
                )
            # 仅填充非导航环境（用掩码限制范围，避免覆盖navi环境）
            obs = carry_obs  # ~self.navi_env是carry环境的掩码
    
            # 处理navi_obs：拼接humanoid_obs并填充到导航环境（仅当非空时）
            # if navi_obs.numel() > 0:  # 检查navi_obs非空
            navi_obs = torch.cat([humanoid_obs, navi_obs], dim=-1)
            # 导航环境的obs维度应与目标一致，若有需要可加pad（可选）
            # pad_size = self.obs_buf.shape[-1] - navi_obs.shape[1]
            # if pad_size > 0:
            #     navi_obs = torch.nn.functional.pad(navi_obs, pad=(0, pad_size), value=0.0)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            local_navi_env = self.navi_env
            obs[local_navi_env] = navi_obs[local_navi_env] 
            self.obs_buf[:] = obs
            # self.obs_buf["navi_obs"][:] = torch.clamp(navi_obs, -self.clip_obs, self.clip_obs).to(self.rl_device)

        else:
            local_navi_mask = torch.isin(env_ids, self.navi_env)
            local_env = torch.arange(len(env_ids), device=self.device)
            local_navi_env = local_env[local_navi_mask]
            obs[local_navi_env] = navi_obs[local_navi_env] 
            self.obs_buf[env_ids] = obs
            # self.obs_buf["navi_obs"][env_ids] = torch.clamp(navi_obs, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return
    
    def _compute_task_obs(self, env_ids=None):
        # device = torch.device(
        #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if env_ids is None:

            self.update_standing_and_held_points(self._humanoid_root_states,
                    self._box_states, self._target_pos, self._target_rot)
            # self.update_standing_and_held_points(self._box_states, self._target_pos, self._target_rot)
            box_states = self._box_states
            # # hardcode here
            vbox_states = box_states.clone()
            # check: shape of it 
            vbox_states[...,:3] = self.box_held_points
            vtar_pos = self.tar_held_points

            tar_pos = self._target_pos
            tar_rot = self._target_rot
            
            root_states = self._humanoid_root_states
            rigid_bodies = self._rigid_body_pos[...,:self.num_bodies]
            box_cloudp = self.pointscloud
            box_bps = self.box_bps
            boxsize = box_bps[:,4,:]-box_bps[:,3,:]
            assert (box_bps[:,3,:]<0).any() and (box_bps[:,4,:]>0).any()
            vbox_bps = self.vbox_bps

            bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
            rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
            rhand_pos_dist = rigid_body_state_reshaped[:,5,:3] # hand: idx5 8 
            lhand_pos_dist = rigid_body_state_reshaped[:,8,:3] # hand: idx5 8 
            # check whether this is vbox dist!!
            rhand_pos_dist, lhand_pos_dist = bbox_contact.get_hand_to_bbox_distances(
                                            box_states=box_states,        # (env, 7)
                                            vbox_bps=box_bps,              # (env, 8, 3)
                                            rhand_pos=rhand_pos_dist,          # (env, 3)
                                            lhand_pos=lhand_pos_dist           # (env, 3)
                                            )
            # 判断距离是否小于0.05
            rhand_contact = rhand_pos_dist < 0.05    # (env,) 布尔张量
            lhand_contact = lhand_pos_dist < 0.05    # (env,) 布尔张量
            # 合并左右接触状态：任意一只手接触则为True
            hand_contact = torch.logical_or(rhand_contact, lhand_contact).float() # (env,)

            # vtar_pos = self.tar_held_points
            # vtar_rot = self._target_rot
            box_standing_points = self.box_standing_points
            self.check_and_switch_states(root_states, box_states, vbox_states, vtar_pos, hand_contact, box_standing_points,self.dt)

            # Note: Update Standing points and held points only can be after the box_states is updated
            box_standing_points = self.box_standing_points
            box_held_points = self.box_held_points
            tar_standing_points = self.tar_standing_points
            tar_held_points = self.tar_held_points
            # density = self.asset_density
        else:

            self.update_standing_and_held_points(self._humanoid_root_states,
                self._box_states, self._target_pos, self._target_rot, env_ids)
            # self.update_standing_and_held_points(
            # self._box_states, self._target_pos, self._target_rot, env_ids)
            
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]

            rigid_bodies = self._rigid_body_pos[env_ids,:self.num_bodies]
            box_cloudp = self.pointscloud[env_ids]

            # # hardcode here
            vbox_states = self._box_states[env_ids]
            vbox_states[..., :3] = self.box_held_points[env_ids]


            box_bps = self.box_bps[env_ids, :,:]
            boxsize = box_bps[:,4,:]-box_bps[:,3,:]
            assert (box_bps[:,3,:]<0).any() and (box_bps[:,4,:]>0).any()
            vbox_bps = self.vbox_bps[env_ids,:, :]
            tar_pos = self._target_pos[env_ids]
            tar_rot = self._target_rot[env_ids]
            vtar_pos = self.tar_held_points[env_ids]
            # vtar_rot = self._target_rot[env_ids]

            bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
            rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
            rhand_pos_dist = rigid_body_state_reshaped[env_ids,5,:3] # hand: idx5 8 
            lhand_pos_dist = rigid_body_state_reshaped[env_ids,8,:3] # hand: idx5 8 
            rhand_pos_dist, lhand_pos_dist = bbox_contact.get_hand_to_bbox_distances(
                                            box_states=box_states,        # (env, 7)
                                            vbox_bps=box_bps,              # (env, 8, 3)
                                            rhand_pos=rhand_pos_dist,          # (env, 3)
                                            lhand_pos=lhand_pos_dist           # (env, 3)
                                            )
            # 判断距离是否小于0.05
            rhand_contact = rhand_pos_dist < 0.05    # (env,) 布尔张量
            lhand_contact = lhand_pos_dist < 0.05    # (env,) 布尔张量
            # 合并左右接触状态：任意一只手接触则为True
            hand_contact = torch.logical_or(rhand_contact, lhand_contact).float() # (env_ids_num,)




            box_standing_points = self.box_standing_points[env_ids]

            self.check_and_switch_states(root_states, box_states, vbox_states, vtar_pos, hand_contact, box_standing_points, self.dt, env_ids = env_ids)
            
            box_standing_points = self.box_standing_points[env_ids]
            box_held_points = self.box_held_points[env_ids]
            tar_standing_points = self.tar_standing_points[env_ids]
            tar_held_points = self.tar_held_points[env_ids]
            # density = self.asset_density[env_ids]
        # chaek: box_standing_points should be on the correct fir of the objA
        obs = compute_carrybox_observations(
            root_states, box_states, tar_pos, tar_rot, vbox_bps, box_standing_points, tar_standing_points, box_held_points, tar_held_points, boxsize
        )
        navi_obs = compute_navi_observations(
            root_states, box_states, box_bps.transpose(0,1), rigid_bodies, box_cloudp, box_standing_points
        )

        if self.obs_add_noise:
            # add normal noise to the observation
            num_env = obs.shape[0]
            # add noise on first 39 dimensions
            noise = torch.normal(mean=0.0, std=self.noise_level, size=(
                num_env, 39), device=obs.device, dtype=obs.dtype)
            obs[:, :39] += noise
        if self.cfg['env']['save_motions']:
            if self.record_step < 10*self.record_frame_number:
                self.output_dict['trans'][:,self.record_step] = self._rigid_body_pos[:].cpu(
                ).numpy()
                self.output_dict['rot'][:,self.record_step] = self._rigid_body_rot[:].cpu(
                ).numpy()
                self.output_dict['obj_pos'][:,self.record_step] = self._box_states[:, 0:3].cpu(
                ).numpy()
                self.output_dict['obj_rot'][:,self.record_step] = self._box_states[:, 3:7].cpu(
                ).numpy()
                self.navi_flag[self.failure_envs.cpu()] += 1
            if self.record_step == 15*self.record_frame_number:
                import shutil
                for i in range(self.num_envs):
                    os.makedirs(f"output/motion_sequence/awr_unevan_{self.navi_flag[i]}_{self.objname[i]}",exist_ok=True)
                    tosave = {k:v[i] for k,v in self.output_dict.items()}
                    np.save(f"output/motion_sequence/awr_unevan_{self.navi_flag[i]}_{self.objname[i]}/Motion.npy",
                        tosave)
                    src_filename = os.path.basename(self.objpath[i])
                    # 定义目标文件路径（移动到目标目录，保留原文件名）
                    dst_file_path = os.path.join(f"output/motion_sequence/awr_unevan_{self.navi_flag[i]}_{self.objname[i]}", src_filename)
                    
                    # 移动文件（若目标目录已存在同名文件，可选择覆盖或重命名）
                    # 方案1：直接移动，若目标文件存在则覆盖
                    shutil.copy2(self.objpath[i], dst_file_path)
                    
        self.record_step += 1
        if self.log_success:
            distance_to_target = torch.norm(box_states[..., 0:3] - tar_pos, dim=-1)
            self._distance_to_target = np.array(distance_to_target.cpu().numpy())
            success_env_mask = self._distance_to_target< 0.2
            success_env_id = np.where(success_env_mask)
            # success_step = [np.where(self._distance_to_target[i] < 0.2)[0][0] for i in success_env_id]
            # print("Success rate: ", success_env_mask.sum() / self.num_envs)
            # print("Success step: ", np.mean(success_step) / 30)
            if self._distance_to_target[success_env_mask].size > 0:
                mean_error = self._distance_to_target[success_env_mask].mean()
            else:
                mean_error = 100
            # print("mean distance: ", mean_error)
            self.log_success_rate.append(success_env_mask.sum() / self.num_envs)
            self.log_success_precision.append(mean_error)
            print("Max Success rate: ", max(self.log_success_rate))
            print("Max Success precision: ", min(self.log_success_precision))

            print(self.record_step)
        return obs, navi_obs

    def get_task_obs_size(self):
        obs_size = 0
        # task_obs will be implemented later
        # if (self._enable_task_obs):
        #     carry_obs_size = 75
        #     navi_obs_size = 84
        # the original was 75, add 1 dimension for the density of the box
        return obs_size
    
    def check_and_switch_states(self, root_states, box_states, vbox_states, vtar_pos, hand_contact, box_standing_points, dt, env_ids = None):
        """检查每个环境是否需要切换状态"""
        root_pos = root_states[:,:3] # check
        vbox_pos = vbox_states[:,:3] # check
        box_pos = box_states[:,:3] # check
        vbox_height = vbox_states[:,2] # check
        env_num = self.reward_fsm.env_num
        # 更新历史状态
        self.reward_fsm.update_history(box_pos, vbox_height, hand_contact, self.failure_mask, self.valid_failure, self.navi_reached_mask, env_ids)
        if env_ids == None: # means not init nor no reset env(because of terminal or reset signal)
            self.failure_mask[:] = self.reward_fsm.is_lift_failure(root_pos, vbox_pos, vbox_height, vtar_pos, dt)
            # history_all_one = (torch.sum(self.reward_fsm.his_failure_mask, dim=-1) == self.reward_fsm.his_failure_mask.shape[-1])  # (env,)
            history_all_one = (torch.sum(self.reward_fsm.his_failure_mask, dim=-1) > 1)  # (env,) # at least 3 times detected 
            history_all_zero = (torch.sum(self.reward_fsm.valid_failure_mask, dim=-1) == 0)  # (env,)
            self.valid_failure[:] = torch.logical_and(torch.logical_and(self.failure_mask , history_all_one), history_all_zero)
            # if self.cfg['args'].awr_play:
            #     self.valid_failure.fill_(False)
            if torch.any(self.valid_failure):
                failure_envs = torch.where(self.valid_failure)[0]
                self.weight_pred.prepare_topK(failure_envs)  # just update in weighte_pred
                # check ! self.failure_envs actually the fail_flag
                # to reset the just fail env # make sure real tar saved
                self._reset_tar2putdown(failure_envs)
                combined = torch.cat([self.failure_envs, failure_envs])
                self.failure_envs = torch.unique(combined, sorted=False)
            boxdown_envids = self.reward_fsm.is_putdown(root_pos, vbox_height, self.failure_envs)
            remove_mask = torch.isin(self.failure_envs, boxdown_envids, invert=False)
            if sum(remove_mask) > 0:
                # should do this!!!!
                # self.failure_envs.remove(boxdown_envids)
                # boxdown_envids = boxdown_envids.to(dtype=torch.long, device=self.failure_envs.device)

                # 生成掩码：标记 self.failure_envs 中不在 boxdown_envids 中的元素（True 表示保留）
                # invert=True 表示取反：不在 boxdown_envids 中的元素为 True
                # 筛选保留的元素，更新 self.failure_envs
                # check if none or empty!!
                boxdown_envids = self.failure_envs[remove_mask]
                self._reset_putdown2tar(boxdown_envids)
                self.topKidx[boxdown_envids] = self.weight_pred.access_topK(boxdown_envids, "base train")
                # self.topKidx[boxdown_envids] = self.weight_pred.get_new_pos(boxdown_envids)
                self.failure_envs = self.failure_envs[~remove_mask]

                # self.box_standing_points, self.box_held_points = new_pos(box_state, stand_offset, held_offset)
                # self.tar_standing_points, self.tar_held_points, self.tar_ground_points = new_tar_pos(const_tar_state, new_offset)
                # OR!! you need to check !
                self.update_standing_and_held_points(self._humanoid_root_states.clone(), self._box_states, self._target_pos, self._target_rot, extra_task="vbox")
                
                # reset or change to NAVI state
                self.reward_fsm.set_state(boxdown_envids, "NAVI")
                self.reward_fsm.retry_counts[boxdown_envids] += 1 # record reset times
                combined = torch.cat([self.navi_env, boxdown_envids])
                self.navi_env = torch.unique(combined, sorted=False)


                # 更新备选站立点
                # 更新备选 stand/held offset # not fail case: keep and remain # (env, topK>=1, 3)--->(env, 3)
                #TODO: use the auto model to find the best K points to go
                # self.stand_points_offset[:,0,:], self.held_points_offset[:,0,:] = self.reward_fsm.get_current_standby_points(
                #     torch.arange(env_num, device=root_pos.device)
                # )
            navi_reached_ids = self.reward_fsm.is_navi_reached(root_pos, box_standing_points)
            self.navi_reached_mask.fill_(False)
            self.navi_reached_mask[navi_reached_ids] = True
            endnavi = (torch.sum(self.reward_fsm.navi_reached_history, dim=1) > self.reward_fsm.his_failure_mask.shape[-1]/2)  # (env,)
            endnavi = torch.where(endnavi)[0]
            if len(endnavi) > 0: #check!!!
                self.reward_fsm.set_state(endnavi, "CARRY")
                keep_mask = ~torch.isin(self.navi_env, endnavi, invert=False)
                # assert len(keep_mask)+len(endnavi) == len(self.navi_env)
                self.navi_env = self.navi_env[keep_mask]
                self.navi_reached_mask[endnavi].fill_(False)

        else: # reset to new or init
            self.valid_failure[env_ids].fill_(False)
            self.failure_mask[env_ids].fill_(False) # reset to normal mask
            self.reward_fsm.reset(env_ids)
            # ???
            # self.failure_envs?
            # combined = torch.cat([self.navi_env, env_ids])
            # self.navi_env = torch.unique(combined, sorted=False)
            keep_mask = ~torch.isin(self.navi_env, env_ids, invert=False)
            self.navi_env = self.navi_env[keep_mask]

            # self.stand_points_offset[env_ids,0,:], self.held_points_offset[env_ids,0,:] =  \
            #         self.reward_fsm.get_current_standby_points(env_ids)
        if self.cfg['args'].swanlab_name != "":
            swanlab.log({"fail case avg": torch.mean(self.failure_mask.float())})

    def _compute_reset(self):
        box_pos = self._box_states[..., 0:3]
        prev_box_pos = self._prev_box_pos
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, self._rigid_body_pos, self._box_contact_forces,
            self._lift_body_ids, self.max_episode_length,
            self._enable_early_termination, self._termination_heights,
            box_pos, prev_box_pos, dt_tensor, hand_positions
        )
        return

    def _compute_reward(self, actions):
        walk_pos_reward_w = 0.1
        walk_vel_reward_w = 0.1
        walk_face_reward_w = 0.1
        held_hand_reward_w = 0.4
        held_height_reward_w = 0.0
        carry_gp_pos_reward_w = 0.1
        carry_box_reward_velocity_w = 0.1
        carry_box_reward_pos_far_w = 0.1
        carry_box_reward_pos_near_w = 0.2
        carry_box_face_reward_w = 0.2
        carry_box_dir_reward_w = 0.1
        putdown_reward_w = 0.0
        # human_reset_reward_w = 0.5
        cenergy_reward_w = 0.05
        venergy_reward_w = 0.01
        press_reward_w = 0.01

        box_pos = self._box_states[..., 0:3]  # Box position
        box_height = box_pos[..., 2]
        vbox_pos = self.box_held_points
        box_rot = self._box_states[..., 3:7]  # Box rotation
        vbox_rot = self._box_states[..., 3:7]  # Box rotation
        prev_box_pos = self._prev_box_pos
        prev_vbox_pos = self._prev_vbox_pos
        box_standing_pos = self.box_standing_points
        box_held_pos = self.box_held_points
        held_point_height = box_held_pos[..., 2]
        _init_box_height = self._init_box_height
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        root_pos = self._humanoid_root_states[..., 0:3]  # 3d state
        root_rot = self._humanoid_root_states[..., 3:7]  # 4d state
        prev_root_pos = self._prev_root_pos
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]
        # box_states = self._box_states[..., 0:3]
        tar_pos = self._target_pos
        vtar_pos = self.tar_held_points
        tar_rot = self._target_rot
        vtar_rot = self._target_rot

        dof_forces = self.dof_force_tensor # env,28
        joint_vels= self._dof_vel # env,28
        foot_forces = self.vec_sensor_tensor # left/right foot 6+6 (env,12)


        target_gp_pos = self.tar_ground_points
        gp_pos = self.box_ground_points
        prev_gp_pos = self._prev_gp_pos

        walk_pos_reward, walk_vel_reward, walk_face_reward = compute_walk_reward(
            root_pos, root_rot, prev_root_pos, box_standing_pos, dt_tensor)
        held_hand_reward = compute_contact_reward(
            hand_positions, box_held_pos, root_pos, box_standing_pos, vbox_pos, vtar_pos)
        height_reward = compute_height_reward(held_point_height, _init_box_height)

        carry_gp_pos_reward,  carry_box_reward_velocity, \
        carry_box_reward_pos_far, carry_box_reward_pos_near, carry_box_face_reward, \
            carry_box_dir_reward, put_down_height_reward= compute_carry_reward(
                target_gp_pos, prev_gp_pos, gp_pos,
                root_pos, root_rot, vbox_pos, vbox_rot, prev_vbox_pos, vtar_pos, vtar_rot, held_point_height, dt_tensor, _init_box_height)
        dof_energy, vertical_reward, press_reward = compute_carry_energy_reward(joint_vels, dof_forces, foot_forces)

        self.rew_buf[:] = walk_pos_reward_w * walk_pos_reward + \
            walk_vel_reward_w * walk_vel_reward + \
            walk_face_reward_w * walk_face_reward + \
            held_hand_reward_w * held_hand_reward + \
            held_height_reward_w * height_reward + \
            carry_gp_pos_reward_w * carry_gp_pos_reward + \
            carry_box_reward_velocity_w * carry_box_reward_velocity + \
            carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
            carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
            carry_box_face_reward_w * carry_box_face_reward + \
            carry_box_dir_reward_w * carry_box_dir_reward + \
            putdown_reward_w * put_down_height_reward + \
            cenergy_reward_w * dof_energy + \
            venergy_reward_w * vertical_reward + \
            press_reward_w * press_reward
            # human_reset_reward_w * human_reset_reward + \
        
        
        if self.cfg["args"].swanlab_name!="":
            swanlab.log({
                "carry_box_reward_pos_near_w * carry_box_reward_pos_near": torch.mean(carry_box_reward_pos_near_w * carry_box_reward_pos_near),
                "putdown_reward_w * put_down_height_reward": torch.mean(putdown_reward_w * put_down_height_reward),
                "walk_pos_reward_w * walk_pos_reward": torch.mean(walk_pos_reward_w * walk_pos_reward),
                "venergy_reward_w * vertical_reward": torch.mean(venergy_reward_w * vertical_reward),
                "press_reward_w * press_reward": torch.mean(press_reward_w * press_reward),
                "cenergy_reward_w*dof_energy": torch.mean(cenergy_reward_w * dof_energy),
                "walk_vel_reward_w * walk_vel_reward": torch.mean(walk_vel_reward_w * walk_vel_reward),
                "walk_face_reward_w * walk_face_reward": torch.mean(walk_face_reward_w * walk_face_reward),
                "held_hand_reward_w * held_hand_reward": torch.mean(held_hand_reward_w * held_hand_reward),
                "held_height_reward_w * height_reward": torch.mean(held_height_reward_w * height_reward),
                "carry_box_reward_pos_far_w * carry_box_reward_pos_far": torch.mean(carry_box_reward_pos_far_w * carry_box_reward_pos_far),
                "carry_box_reward_velocity_w * carry_box_reward_velocity": torch.mean(carry_box_reward_velocity_w * carry_box_reward_velocity),
                "carry_box_face_reward_w * carry_box_face_reward": torch.mean(carry_box_face_reward_w * carry_box_face_reward),
                "carry_box_dir_reward_w * carry_box_dir_reward": torch.mean(carry_box_dir_reward_w * carry_box_dir_reward),
                # "human_reset_reward_w * human_reset_reward": torch.mean(human_reset_reward_w * human_reset_reward),
                })
        target_state_diff = tar_pos - box_pos
        err =  torch.norm(target_state_diff[..., 0:2], dim=-1)
        success = err <= 0.1  # near_mask
        self.extras['success'] = success.float()
        self.extras['err'] = err.float()
        self.extras['task_awr_rwd'] = dof_energy
        # walk_reward = walk_pos_reward_w * walk_pos_reward + \
        #     walk_vel_reward_w * walk_vel_reward + \
        #     walk_face_reward_w * walk_face_reward
        # contact_reward = held_hand_reward_w * held_hand_reward
        # carry_reward = carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
        #     carry_box_reward_velocity_w * carry_box_reward_velocity + \
        #     carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
        #     carry_box_face_reward_w * carry_box_face_reward + \
        #     carry_box_dir_reward_w * carry_box_dir_reward + \
        #     putdown_reward_w * put_down_height_reward

        # box_half_height = self._height_box_size / 2.0
        # height_diff = compute_box_raise_height(box_half_height, box_height)
        return
    def _evaluate_metrics(self):
        box_pos = self._box_states[..., 0:3]
        tar_pos = self._const_target_pos[..., 0:3]
        distance_to_target = torch.norm(box_pos - tar_pos, dim=-1)
        self._distance_to_target = np.array(distance_to_target.cpu().numpy())
        success_env_mask = (self._distance_to_target< 0.2)
        success_env_id = np.where(success_env_mask)
        mean_error = 100
        if self._distance_to_target[success_env_mask].size > 0:
            mean_error = self._distance_to_target[success_env_mask].mean()
        metrics = {
            "distance_to_target": self._distance_to_target,
            "success_mask": success_env_mask,
            "success_precision": mean_error
        }
        return metrics
    
    def _update_task(self):
        # if self._enable_task_update:
        #     change_steps = torch.randint(low=self._task_finish_steps_min, high=self._task_finish_steps_max, size=(
        #         self.num_envs,), device=self.device, dtype=torch.int64)
        #     reset_task_mask = self._task_finish_steps > change_steps
        #     rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        #     if len(rest_env_ids) > 0:
        #         self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        # self._reset_target(env_ids, randomize=True)
        # self._task_finish_steps[env_ids] = 0
        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        # if (self._enable_task_obs):
        #     task_obs_size = self.get_task_obs_size()
        #     obs_size += task_obs_size
        return obs_size
    


    
    def _draw_task(self):
        cols = np.array([[1.0, 0.1, 0.2]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        tar_pos = self._target_pos
        tar_rot = self._target_rot

        box_bps = self.box_bps.transpose(0,1)
        lfus = box_bps[0]
        lfds = box_bps[1]
        lbus = box_bps[2]
        lbds = box_bps[3]

        rfus = box_bps[4]
        rfds = box_bps[5]
        rbus = box_bps[6]
        rbds = box_bps[7]

        tar_lfus = convert_static_point_to_world(lfus, tar_pos, tar_rot)
        tar_lfds = convert_static_point_to_world(lfds, tar_pos, tar_rot)
        tar_lbus = convert_static_point_to_world(lbus, tar_pos, tar_rot)
        tar_lbds = convert_static_point_to_world(lbds, tar_pos, tar_rot)
        tar_rfus = convert_static_point_to_world(rfus, tar_pos, tar_rot)
        tar_rfds = convert_static_point_to_world(rfds, tar_pos, tar_rot)
        tar_rbus = convert_static_point_to_world(rbus, tar_pos, tar_rot)
        tar_rbds = convert_static_point_to_world(rbds, tar_pos, tar_rot)

        verts1 = torch.cat([tar_lfus, tar_lfds], dim=-1).cpu().numpy()
        verts2 = torch.cat([tar_lbus, tar_lbds], dim=-1).cpu().numpy()
        verts3 = torch.cat([tar_rfus, tar_rfds], dim=-1).cpu().numpy()
        verts4 = torch.cat([tar_rbus, tar_rbds], dim=-1).cpu().numpy()
        verts5 = torch.cat([tar_lfus, tar_lbus], dim=-1).cpu().numpy()
        verts6 = torch.cat([tar_lbus, tar_rbus], dim=-1).cpu().numpy()
        verts7 = torch.cat([tar_rbus, tar_rfus], dim=-1).cpu().numpy()
        verts8 = torch.cat([tar_rfus, tar_lfus], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
        if self.show_bbox:
            self._draw_curr_vbox()
            self._draw_curr_box()
            self._draw_tar_vbox()
            self._draw_vbox_standp()
            self._draw_vbox_heldp()
        if self.show_contact_forces:
            self._draw_force()
        return
    def _draw_force(self):
        # 1. draw force #############
        all_lines = view_force.generate_multi_env_lines(
            envs=self.envs,
            # actor_handles=self.humanoid_handles,
            actor_handles=self._box_handles,

            gym=self.gym,
            show_joint=self.show_joint_forces,
            show_contact=self.show_contact_forces,
            contact_forces = self.contact_force_tensor.clone().cpu().numpy(), 
            # dof_forces_tensor=self.dof_force_tensor.clone().reshape(-1).cpu().numpy(),  
            dof_forces_tensor=None,  
            rigid_body_states= self._rigid_body_state.clone().cpu().numpy(),
            scale=0.05
        )

        # 2. 为每个环境添加线条到viewer
        for env_idx, (verts, cols, num) in enumerate(all_lines):
            if num > 0:  # 只添加有线条的环境
                self.gym.add_lines(
                    self.viewer,
                    self.envs[env_idx],  # 指定当前环境
                    num,
                    verts,
                    cols
            )

        # ##############################
        # self.gym.draw_viewer(self.viewer, self.sim, True)

    def _draw_curr_box(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        box_pos = self._box_pos
        box_rot = self._box_rot

        box_bps = self.box_bps.transpose(0,1)
        lfus = box_bps[0]
        lfds = box_bps[1]
        lbus = box_bps[2]
        lbds = box_bps[3]

        rfus = box_bps[4]
        rfds = box_bps[5]
        rbus = box_bps[6]
        rbds = box_bps[7]

        tar_lfus = convert_static_point_to_world(lfus, box_pos, box_rot)
        tar_lfds = convert_static_point_to_world(lfds, box_pos, box_rot)
        tar_lbus = convert_static_point_to_world(lbus, box_pos, box_rot)
        tar_lbds = convert_static_point_to_world(lbds, box_pos, box_rot)
        tar_rfus = convert_static_point_to_world(rfus, box_pos, box_rot)
        tar_rfds = convert_static_point_to_world(rfds, box_pos, box_rot)
        tar_rbus = convert_static_point_to_world(rbus, box_pos, box_rot)
        tar_rbds = convert_static_point_to_world(rbds, box_pos, box_rot)

        verts1 = torch.cat([tar_lfus, tar_lfds], dim=-1).cpu().numpy()
        verts2 = torch.cat([tar_lbus, tar_lbds], dim=-1).cpu().numpy()
        verts3 = torch.cat([tar_rfus, tar_rfds], dim=-1).cpu().numpy()
        verts4 = torch.cat([tar_rbus, tar_rbds], dim=-1).cpu().numpy()
        verts5 = torch.cat([tar_lfus, tar_lbus], dim=-1).cpu().numpy()
        verts6 = torch.cat([tar_lbus, tar_rbus], dim=-1).cpu().numpy()
        verts7 = torch.cat([tar_rbus, tar_rfus], dim=-1).cpu().numpy()
        verts8 = torch.cat([tar_rfus, tar_lfus], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
    def _draw_tar_vbox(self):
        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        initial_b_vertices = self.vbox_bps
        num_envs = initial_b_vertices.shape[0]
        # root_pos = self._root_states[:,0:3]
        # root_rot = self._root_states[:,3:7]
        # box_pos = self._box_states[:,0:3]
        tar_pos = self._target_pos
        tar_rot = self._target_rot
        # 1. 扩展旋转四元数以匹配顶点数量
        # 从 [num_envs, 4] 扩展为 [num_envs, 8, 4]，再展平为 [num_envs*8, 4]
        box_rot_extended = tar_rot.unsqueeze(1).repeat(1, 8, 1).view(-1, 4)
        
        # 2. 展平顶点坐标以便批量旋转 [num_envs*8, 3]
        vertices_flat = initial_b_vertices.view(-1, 3)
        
        # 3. 应用旋转（绕A的原点旋转）
        rotated_vertices = quat_rotate(box_rot_extended, vertices_flat)  # [num_envs*8, 3]
        
        # 4. 恢复形状并应用位移（A的位置偏移）
        rotated_vertices = rotated_vertices  # [num_envs*8, 3]
        world_vertices = rotated_vertices + tar_pos.unsqueeze(1).repeat(1, 8, 1).view(-1, 3)  # 加上A的位移
        world_vertices = world_vertices.view(num_envs,-1,3)

        verts1 = torch.cat([world_vertices[:,0,:], world_vertices[:,1,:]], dim=-1).cpu().numpy()
        verts2 = torch.cat([world_vertices[:,2,:], world_vertices[:,3,:]], dim=-1).cpu().numpy()
        verts3 = torch.cat([world_vertices[:,4,:], world_vertices[:,5,:]], dim=-1).cpu().numpy()
        verts4 = torch.cat([world_vertices[:,6,:], world_vertices[:,7,:]], dim=-1).cpu().numpy()
        verts5 = torch.cat([world_vertices[:,0,:], world_vertices[:,2,:]], dim=-1).cpu().numpy()
        verts6 = torch.cat([world_vertices[:,2,:], world_vertices[:,6,:]], dim=-1).cpu().numpy()
        verts7 = torch.cat([world_vertices[:,6,:], world_vertices[:,4,:]], dim=-1).cpu().numpy()
        verts8 = torch.cat([world_vertices[:,4,:], world_vertices[:,0,:]], dim=-1).cpu().numpy()
        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
    def _draw_vbox_standp(self):
        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        curr_cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tar_cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        curr_start = self.box_standing_points.cpu().numpy()
        curr_end = curr_start+cols
        vert_curr = np.concatenate((curr_start,curr_end),axis=-1)
        tar_start = self.tar_standing_points.cpu().numpy()
        tar_end = tar_start+cols
        vert_tar = np.concatenate((tar_start,tar_end),axis=-1)

        for i, env_ptr in enumerate(self.envs):
            curr_verts = vert_curr[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, curr_cols)
            tar_verts = vert_tar[i]
            tar_verts = tar_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               tar_verts.shape[0], tar_verts, tar_cols)
    def _draw_vbox_heldp(self):
        cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        curr_cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        tar_cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        curr_start = self.box_held_points.cpu().numpy()
        curr_end = curr_start+cols
        vert_curr = np.concatenate((curr_start,curr_end),axis=-1)
        tar_start = self.tar_held_points.cpu().numpy()
        tar_end = tar_start+cols
        vert_tar = np.concatenate((tar_start,tar_end),axis=-1)

        for i, env_ptr in enumerate(self.envs):
            curr_verts = vert_curr[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, curr_cols)
            tar_verts = vert_tar[i]
            tar_verts = tar_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               tar_verts.shape[0], tar_verts, tar_cols)
    def _draw_curr_vbox(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        initial_b_vertices = self.vbox_bps
        num_envs = initial_b_vertices.shape[0]
        root_pos = self._root_states[:,0:3]
        root_rot = self._root_states[:,3:7]
        box_pos = self._box_states[:,0:3]
        box_rot = self._box_states[:,3:7]
        # 1. 扩展旋转四元数以匹配顶点数量
        # 从 [num_envs, 4] 扩展为 [num_envs, 8, 4]，再展平为 [num_envs*8, 4]
        box_rot_extended = box_rot.unsqueeze(1).repeat(1, 8, 1).view(-1, 4) # env,p
        
        # 2. 展平顶点坐标以便批量旋转 [num_envs*8, 3]
        vertices_flat = initial_b_vertices.view(-1, 3)
        
        # 3. 应用旋转（绕A的原点旋转）
        rotated_vertices = quat_rotate(box_rot_extended, vertices_flat)  # [num_envs*8, 3]
        
        # 4. 恢复形状并应用位移（A的位置偏移）
        rotated_vertices = rotated_vertices  # [num_envs*8, 3]
        world_vertices = rotated_vertices + box_pos.unsqueeze(1).repeat(1, 8, 1).view(-1, 3)  # 加上A的位移
        world_vertices = world_vertices.view(num_envs,-1,3)

        verts1 = torch.cat([world_vertices[:,0,:], world_vertices[:,1,:]], dim=-1).cpu().numpy()
        verts2 = torch.cat([world_vertices[:,2,:], world_vertices[:,3,:]], dim=-1).cpu().numpy()
        verts3 = torch.cat([world_vertices[:,4,:], world_vertices[:,5,:]], dim=-1).cpu().numpy()
        verts4 = torch.cat([world_vertices[:,6,:], world_vertices[:,7,:]], dim=-1).cpu().numpy()
        verts5 = torch.cat([world_vertices[:,0,:], world_vertices[:,2,:]], dim=-1).cpu().numpy()
        verts6 = torch.cat([world_vertices[:,2,:], world_vertices[:,6,:]], dim=-1).cpu().numpy()
        verts7 = torch.cat([world_vertices[:,6,:], world_vertices[:,4,:]], dim=-1).cpu().numpy()
        verts8 = torch.cat([world_vertices[:,4,:], world_vertices[:,0,:]], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)


#####################################################################
### =========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_static_point_to_local_observation(point_pos, root_states, central_pos, central_rot):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    point_states = torch.zeros_like(root_states[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_point_pos = quat_rotate(heading_rot, target_point_staets - root_pos)
    return local_point_pos


@torch.jit.script
def convert_static_point_to_world(point_pos, central_pos, central_rot):
    point_states = torch.zeros_like(central_pos[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    return target_point_staets


# @torch.jit.script   
def compute_carrybox_observations(root_states, box_states, tar_pos, tar_rot, vbox_bps, box_standing_points, tar_standing_points, box_held_points, tar_held_points, boxsize):
    num_envs = len(root_states)
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)  # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]
    vbox_pos = box_held_points
    vbox_rot = box_states[:, 3:7]
    vbox_vel = box_states[:, 7:10]
    vbox_ang_vel = box_states[:, 10:13]

    local_vbox_pos = vbox_pos - root_pos
    local_vbox_pos = quat_rotate(heading_rot, local_vbox_pos)

    box_standing_points_xy = box_standing_points[:, 0:3].clone()
    box_standing_points_xy[:, 2] = 0.0
    local_box_standing_points_pos = box_standing_points_xy - root_pos
    local_box_standing_points_pos = quat_rotate(
        heading_rot, local_box_standing_points_pos)

    local_vbox_rot = quat_mul(heading_rot, vbox_rot)
    local_vbox_rot_obs = torch_utils.quat_to_tan_norm(local_vbox_rot)

    local_vbox_vel = quat_rotate(heading_rot, vbox_vel)
    local_vbox_ang_vel = quat_rotate(heading_rot, vbox_ang_vel)


    vtar_pos = tar_held_points
    vtar_rot = tar_rot
    local_vtar_pos = vtar_pos - root_pos
    local_vtar_pos_obs = quat_rotate(heading_rot, local_vtar_pos)
    local_vtar_rot = quat_mul(heading_rot, vtar_rot)
    local_vtar_rot_obs = torch_utils.quat_to_tan_norm(local_vtar_rot)

    tar_standing_points_xy = tar_standing_points[:, 0:3]
    tar_standing_points_xy[:, 2] = 0.0
    local_tar_standing_points_pos = tar_standing_points_xy - root_pos
    local_tar_standing_points_pos = quat_rotate(
        heading_rot, local_tar_standing_points_pos)



    box_local_vertices = vbox.compute_transformed_b_part_bbox(vbox_bps, box_pos, box_rot, root_states) # num_env,8,3
    tar_local_vertices = vbox.compute_transformed_b_part_bbox(vbox_bps, tar_pos, tar_rot, root_states)
    
    # lfus = box_bps[0]
    # lfds = box_bps[1]
    # lbus = box_bps[2]
    # lbds = box_bps[3]

    # rfus = box_bps[4]
    # rfds = box_bps[5]
    # rbus = box_bps[6]
    # rbds = box_bps[7]

    # box_local_lfus_pos = convert_static_point_to_local_observation(
    #     lfus, root_states, box_pos, box_rot)
    # box_local_lfds_pos = convert_static_point_to_local_observation(
    #     lfds, root_states, box_pos, box_rot)
    # box_local_lbus_pos = convert_static_point_to_local_observation(
    #     lbus, root_states, box_pos, box_rot)
    # box_local_lbds_pos = convert_static_point_to_local_observation(
    #     lbds, root_states, box_pos, box_rot)

    # box_local_rfus_pos = convert_static_point_to_local_observation(
    #     rfus, root_states, box_pos, box_rot)
    # box_local_rfds_pos = convert_static_point_to_local_observation(
    #     rfds, root_states, box_pos, box_rot)
    # box_local_rbus_pos = convert_static_point_to_local_observation(
    #     rbus, root_states, box_pos, box_rot)
    # box_local_rbds_pos = convert_static_point_to_local_observation(
    #     rbds, root_states, box_pos, box_rot)

    # add bps for tar

    # tar_local_lfus_pos = convert_static_point_to_local_observation(
    #     lfus, root_states, tar_pos, tar_rot)
    # tar_local_lfds_pos = convert_static_point_to_local_observation(
    #     lfds, root_states, tar_pos, tar_rot)
    # tar_local_lbus_pos = convert_static_point_to_local_observation(
    #     lbus, root_states, tar_pos, tar_rot)
    # tar_local_lbds_pos = convert_static_point_to_local_observation(
    #     lbds, root_states, tar_pos, tar_rot)

    # tar_local_rfus_pos = convert_static_point_to_local_observation(
    #     rfus, root_states, tar_pos, tar_rot)
    # tar_local_rfds_pos = convert_static_point_to_local_observation(
    #     rfds, root_states, tar_pos, tar_rot)
    # tar_local_rbus_pos = convert_static_point_to_local_observation(
    #     rbus, root_states, tar_pos, tar_rot)
    # tar_local_rbds_pos = convert_static_point_to_local_observation(
    #     rbds, root_states, tar_pos, tar_rot)
    obs = boxsize
    obs = torch.cat([local_vbox_pos, local_vbox_rot_obs,
                    local_vbox_vel, local_vbox_ang_vel,obs], dim=-1)
    # obs = torch.cat([box_local_lfus_pos, box_local_lfds_pos, box_local_lbus_pos, box_local_lbds_pos,
    #                 box_local_rfus_pos, box_local_rfds_pos, box_local_rbus_pos, box_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([box_local_vertices.reshape(num_envs,-1), obs], dim=-1)
    obs = torch.cat([local_box_standing_points_pos, obs], dim=-1)
    obs = torch.cat([local_vtar_pos_obs, local_vtar_rot_obs, obs], dim=-1)
    # obs = torch.cat([tar_local_lfus_pos, tar_local_lfds_pos, tar_local_lbus_pos, tar_local_lbds_pos,
    #                 tar_local_rfus_pos, tar_local_rfds_pos, tar_local_rbus_pos, tar_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([tar_local_vertices.reshape(num_envs,-1), obs], dim=-1)


    return obs

# @torch.jit.script   
def compute_navi_observations(root_states, box_states, box_bps,rigid_bodies, box_cloudp,box_standing_points):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)  # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    # box_ang_vel = box_states[:, 10:13]

    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    # box_standing_points_xy = box_standing_points[:, 0:3]
    # box_standing_points_xy[:, 2] = 0.0
    # local_box_standing_points_pos = box_standing_points_xy - root_pos
    # local_box_standing_points_pos = quat_rotate(
    #     heading_rot, local_box_standing_points_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    local_box_vel = quat_rotate(heading_rot, box_vel)
    # local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    # local_tar_pos = tar_pos - root_pos
    # local_tar_pos_obs = quat_rotate(heading_rot, local_tar_pos)
    # local_tar_rot = quat_mul(heading_rot, tar_rot)
    # local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    box_standing_points_xy = box_standing_points[:, 0:3]
    box_standing_points_xy[:, 2] = 0.0
    local_box_standing_points_pos = box_standing_points_xy - root_pos
    local_box_standing_points_pos = quat_rotate(
        heading_rot, local_box_standing_points_pos)

    lfus = box_bps[0]
    lfds = box_bps[1]
    lbus = box_bps[2]
    lbds = box_bps[3]

    rfus = box_bps[4]
    rfds = box_bps[5]
    rbus = box_bps[6]
    rbds = box_bps[7]

    box_local_lfus_pos = convert_static_point_to_local_observation(
        lfus, root_states, box_pos, box_rot)
    box_local_lfds_pos = convert_static_point_to_local_observation(
        lfds, root_states, box_pos, box_rot)
    box_local_lbus_pos = convert_static_point_to_local_observation(
        lbus, root_states, box_pos, box_rot)
    box_local_lbds_pos = convert_static_point_to_local_observation(
        lbds, root_states, box_pos, box_rot)

    box_local_rfus_pos = convert_static_point_to_local_observation(
        rfus, root_states, box_pos, box_rot)
    box_local_rfds_pos = convert_static_point_to_local_observation(
        rfds, root_states, box_pos, box_rot)
    box_local_rbus_pos = convert_static_point_to_local_observation(
        rbus, root_states, box_pos, box_rot)
    box_local_rbds_pos = convert_static_point_to_local_observation(
        rbds, root_states, box_pos, box_rot)



    #### cloud point ####
    # 计算物体点云在全局坐标系中的位置
    # 1. 扩展旋转向量以匹配点云形状
    # box_cloudp形状: (num_envs, 1024, 3)
    # check!!! set special points to verify
    box_rot_extend = box_rot.unsqueeze(1).repeat(1, box_cloudp.shape[1], 1)  # (num_envs, 1024, 4)
    box_rot_extend = box_rot_extend.view(-1, 4)  # (num_envs*1024, 4)
    
    # 2. 展平点云以便旋转
    box_cloudp_flat = box_cloudp.view(-1, 3)  # (num_envs*1024, 3)
    
    # 3. 应用旋转并恢复形状
    rotated_points = quat_rotate(box_rot_extend, box_cloudp_flat)  # (num_envs*1024, 3)
    rotated_points = rotated_points.view(box_cloudp.shape[0], box_cloudp.shape[1], 3)  # (num_envs, 1024, 3)
    
    # 4. 应用平移，将点云转换到全局坐标系
    global_box_cloud = rotated_points + box_pos.unsqueeze(1)  # (num_envs, 1024, 3)
    # p2-->p1 # change direction by -
    min_dis_mat = -compute_sdf(rigid_bodies,global_box_cloud).reshape(box_cloudp.shape[0], -1)  # (num_envs, 15, 3)

    obs = torch.cat([local_box_pos, local_box_rot_obs, local_box_vel], dim=-1)
    obs = torch.cat([box_local_lfus_pos, box_local_lfds_pos, box_local_lbus_pos, box_local_lbds_pos,
                    box_local_rfus_pos, box_local_rfds_pos, box_local_rbus_pos, box_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([local_box_standing_points_pos, obs], dim=-1) # env, 3
    obs = torch.cat([min_dis_mat, obs], dim=-1) # env,45



    return obs
# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids,
                           rigid_body_pos, box_contact_forces, lift_body_ids,
                           max_episode_length, enable_early_termination,
                           termination_heights,
                           box_pos, prev_box_pos, dt_tensor, hand_positions):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor,Tensor,Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0
    box_vel_threshold = 1.0
    box_height_threshold = 0.35

    terminated = torch.zeros_like(reset_buf)

    # Early termination logic based on contact forces and body positions
    if enable_early_termination:
        # Mask the contact forces of the lifting body parts so they're not considered
        fall_masked_contact_buf = contact_buf.clone()
        fall_masked_contact_buf[:, contact_body_ids, :] = 0

        # Check if any body parts are making contact with a force above a minimal threshold
        # to determine if a fall contact has occurred.
        fall_contact = torch.any(
            torch.abs(fall_masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # Check if the body height of any body parts is below a certain threshold
        # to determine if a fall due to height has occurred.
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        # Do not consider lifting body parts for the height check
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # Combine the conditions to determine if the humanoid has fallen
        has_fallen = torch.logical_and(fall_contact, fall_height)

        # check if the humanoid is kicking the box
        # box_has_contact_horizontal = torch.any(
        #     torch.abs(box_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        # check if the humanoid is kicking the box
        box_height = box_pos[..., 2]
        delta_box_pos = box_pos - prev_box_pos
        box_vel = delta_box_pos / dt_tensor
        box_vel_xy = box_vel[..., 0:2]
        box_vel_xy_norm = torch.norm(box_vel_xy, dim=-1)
        box_has_velocity_horizontal = box_vel_xy_norm > box_vel_threshold
        box_low = box_height < box_height_threshold
        mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
        hand_high = mean_hand_positions[..., 2] > 0.5

        box_kicked = torch.logical_and(box_has_velocity_horizontal, box_low)
        box_kicked_with_hands_high = torch.logical_and(box_kicked, hand_high)

        # has_failed = has_fallen
        # if forbid the agents to kick box,the agents may not know what to do.
        # has_failed = torch.logical_or(has_fallen, box_kicked_with_hands_high)
        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(
            has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_walk_reward(root_pos, root_rot, prev_root_pos, box_standing_pos, dt):
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
    root_vel = delta_root_pos / dt
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
def compute_contact_reward(hand_positions, box_held_points, root_pos, box_standing_pos, vbox_pos, vtar_pos):
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
    target_state_diff = vtar_pos - vbox_pos  # xyz
    target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
    near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
    near_and_low_mask = torch.logical_and(
        near_mask, box_height < box_height_threshold)
    hands2box_reward[near_and_low_mask] = 1.0
    return hands2box_reward


@torch.jit.script
def compute_height_reward(held_point_height, _init_box_height):
    least_height = _init_box_height + 0.15

    target_height = 0.8
    height_err_scale = 5.0
    box_height_diff = target_height - held_point_height
    height_reward = torch.exp(
        -height_err_scale * box_height_diff * box_height_diff)
    toolow_mask = held_point_height < least_height
    height_reward[toolow_mask] = height_reward[toolow_mask]*height_reward[toolow_mask]
    return height_reward


# @torch.jit.script
def compute_carry_reward(target_gp_pos, prev_gp_pos, gp_pos, root_pos, root_rot, vbox_pos, vbox_rot, prev_vbox_pos, vtarget_pos, vtarget_rot, held_point_height, dt_tensor, _init_box_height):
    target_speed = 1.0  # target speed in m/s
    carry_dist_threshold = 0.2
    height_threshold = _init_box_height + 0.1 # hardcode：0.2
    vtar_pos_err_far_scale = 0.5
    vtarget_pos_err_near_scale = 10.0
    gp_pos_err_far_scale = 10.0
    carry_vel_err_scale = 2.0
    tar_dis_scale = 5
    human_reset_reward_scale = 10.0

    x_axis = torch.zeros_like(root_pos[..., 0:3])
    x_axis[..., 0] = 1.0

    # masks
    vbox_height = vbox_pos[..., 2]
    height_mask = vbox_height < height_threshold

    # compute r_carry_pos
    gp_state_diff = target_gp_pos - gp_pos  # xyz
    gp_pos_err_xy = torch.norm(gp_state_diff[..., 0:2], dim=-1)
    gp_pos_reward = torch.exp(-gp_pos_err_far_scale *
                                      gp_pos_err_xy)
    gp_near_mask = gp_pos_err_xy <= carry_dist_threshold  # near_mask
    far_and_low_mask = torch.logical_and(~gp_near_mask, height_mask)
    gp_pos_reward[gp_near_mask] = 1.0
    gp_pos_reward[far_and_low_mask] = 0.0

    # compute r_carry_pos
    vtarget_state_diff = vtarget_pos - vbox_pos  # xyz
    vtarget_pos_err_xy = torch.norm(vtarget_state_diff[..., 0:2], dim=-1)
    near_mask = vtarget_pos_err_xy <= carry_dist_threshold  # near_mask
    vtarget_pos_err_xyz = torch.norm(vtarget_state_diff[..., 0:3], dim=-1)
    vtarget_pos_reward_far = torch.exp(-vtar_pos_err_far_scale *
                                      vtarget_pos_err_xy)
    vtarget_pos_reward_near = torch.exp(-vtarget_pos_err_near_scale *
                                       (vtarget_pos_err_xyz**2))

    vtarget_pos_reward_far[~gp_near_mask] = 0.0
    vtarget_pos_reward_near[~gp_near_mask] = 0.0
    vtarget_pos_reward_far[near_mask] = 1.0
    # 
    # boxok_mask = target_pos_reward_near >= 0.55
    # box_height_mask = torch.sum(target_state_diff[...,2]** 2,dim=-1)<=0.02
    # boxok_mask = torch.logical_and(boxok_mask,box_height_mask)
    # human_reset_xy = torch.sum((root_pos-box_pos)[...,:2]** 2, dim=-1)
    # human_reset_reward = 1-torch.exp(-human_reset_reward_scale *
    #                                   human_reset_xy)
    # ok_mask = human_reset_xy >= 0.25
    # ok_mask = torch.logical_and(boxok_mask,ok_mask)
    
    # human_reset_reward[~boxok_mask] = 0.0

    # human_reset_reward[ok_mask] = 1.0



    # compute_r_carry_face
    tar_dir = target_gp_pos[..., 0:2] - gp_pos[..., 0:2]
    tar_dist = torch.norm(tar_dir,dim=-1)
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_reverse = gp_pos[..., 0:2] - target_gp_pos[..., 0:2]
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
    facing_reward[gp_near_mask] = 1.0

    # compute r_carry_vel
    delta_gp_pos =  gp_pos - prev_gp_pos
    gp_vel = delta_gp_pos / dt_tensor
    gp_tar_dir_speed = torch.sum(
        tar_dir * gp_vel[..., 0:2], dim=-1)
    target_speed = 1-torch.exp(-tar_dis_scale * (tar_dist**2))
    gp_vel_err = target_speed - gp_tar_dir_speed
    gp_vel_reward = torch.exp(-carry_vel_err_scale *
                               (gp_vel_err * gp_vel_err))

    tar_speed_mask = gp_tar_dir_speed <= 0
    gp_vel_reward[tar_speed_mask] = 0
    gp_vel_reward[height_mask] = 0.0
    gp_vel_reward[gp_near_mask] = 1.0

    
    # delta_vbox_pos = vbox_pos - prev_vbox_pos
    # vbox_vel = delta_vbox_pos / dt_tensor
    # vbox_tar_dir_speed = torch.sum(
    #     tar_dir * vbox_vel[..., 0:2], dim=-1)
    # # target_speed = target_speed # speed slowed when near
    # target_speed = 1-torch.exp(-tar_dis_scale * (tar_dist**2))
    # vtar_vel_err = target_speed - vbox_tar_dir_speed
    # # vtar_vel_err = torch.clamp_min(vtar_vel_err, 0.0)
    # vtar_vel_reward = torch.exp(-carry_vel_err_scale *
    #                            (vtar_vel_err * vtar_vel_err))
    # tar_speed_mask = vbox_tar_dir_speed <= 0
    # vtar_vel_reward[tar_speed_mask] = 0
    # vtar_vel_reward[height_mask] = 0.0
    # vtar_vel_reward[near_mask] = 1.0

    # compute r_carry_dir
    # calculate the facing direction of the box
    vbox_facing_dir = quat_rotate(vbox_rot, x_axis)
    vtar_facing_dir = quat_rotate(vtarget_rot, x_axis)
    dir_err = torch.sum(
        vbox_facing_dir[..., 0:2] * vtar_facing_dir[..., 0:2], dim=-1)  # xy;higher value indicating better alignment
    dir_reward = torch.clamp_min(dir_err, 0.0)
    dir_reward[~near_mask] = 0.0

    # compute r_putdown
    held_points_height = held_point_height - vtarget_pos[..., 2]
    put_down_height_reward = torch.exp(
        -5.0 * held_points_height * held_points_height)
    put_down_height_reward[~near_mask] = 0


    return gp_pos_reward, gp_vel_reward, vtarget_pos_reward_far,  vtarget_pos_reward_near, facing_reward, dir_reward, put_down_height_reward

@torch.jit.script
def verticality_reward(foot_force, eps: float = 1e-6, vertical_k: float = 5.0):
    fx, fy, fz = foot_force[..., 0], foot_force[..., 1], foot_force[..., 2]
    horizontal_mag = torch.sqrt(fx**2 + fy**2 + eps)  # 水平力大小（加eps避免除0）
    vertical_mag = torch.abs(fz) + eps  # 垂直力大小
    ratio = horizontal_mag / vertical_mag  # 期望趋近于0
    return torch.exp(-vertical_k * ratio)  # 期望趋近于1

@torch.jit.script
# ---------------------- 3. 地面压力奖励（举物时压力最大化） ----------------------
def pressure_reward(foot_force, max_pressure: float = 100.0):  # 新增float类型注解
    fz = foot_force[..., 2]  # 垂直方向力（up为正）
    positive_fz = torch.clamp(fz, min=0.0)  # 只考虑向下的压力（忽略向上的力）
    normalized = positive_fz / max_pressure  # 归一化到[0,1]（超过max_pressure则饱和）
    return torch.tanh(normalized)  # 奖励在[0,1)，expect-->1

@torch.jit.script
def verticality_reward(foot_force, eps: float = 1e-6, vertical_k: float = 5.0):
    fx, fy, fz = foot_force[..., 0], foot_force[..., 1], foot_force[..., 2]
    horizontal_mag = torch.sqrt(fx**2 + fy**2 + eps)  # 水平力大小（加eps避免除0）
    vertical_mag = torch.abs(fz) + eps  # 垂直力大小
    ratio = horizontal_mag / vertical_mag  # 期望趋近于0
    return torch.exp(-vertical_k * ratio)  # 期望趋近于1

@torch.jit.script
def compute_carry_energy_reward(joint_vels, dof_forces, foot_forces):
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
    
    # ---------------------- 2. 脚底力垂直性奖励 ----------------------
    left_force = foot_forces[..., 0:3]  # (env, 3)
    right_force = foot_forces[..., 6:9]  # (env, 3)
    
    left_vertical = verticality_reward(left_force)
    right_vertical = verticality_reward(right_force)
    vertical_reward = (left_vertical + right_vertical) / 2  # 左右脚平均
    
    # ---------------------- 3. 地面压力奖励 ----------------------
    left_pressure = pressure_reward(left_force)
    right_pressure = pressure_reward(right_force)
    press_reward = (left_pressure + right_pressure) / 2  # 左右脚平均
    
    return dof_energy, vertical_reward, press_reward


@torch.jit.script
def compute_task_finish(box_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - box_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask


@torch.jit.script
def compute_box_raise_height(box_half_size, box_height):
    height_diff = box_height - box_half_size
    return height_diff

# @torch.jit.script
def _compute_gpoint_idx(standp_offset, box_bps ):
    gpoint = (standp_offset.unsqueeze(-2).repeat(1,8,1)**2)>box_bps**2
    s = torch.sum(gpoint,dim=-2)==8 # 每个环境中，哪些位置是True(表示站点在轴的方向)
    if (len(s[torch.sum(s,dim=-1)>1])>0): # hardcode here!!!!
        s[torch.sum(s,dim=-1)>1] = torch.tensor([1,0,0], device=s.device, dtype=s.dtype)
    s = standp_offset*s # 每个环境中，保留位置是True的符号
    # 表示触地点在轴的方向(x or y)
    s = -s # 符号取反为触地点符号
    s[:,-1] = -1 #触地点在z轴的负方向
    s = s.unsqueeze(1).repeat(1,8,1)
    # env,8,3
    judge = box_bps*s>0 # 希望两个符号均与之相同，还有一个是0； 
    judge = torch.sum(judge, dim=-1)==2 # env,8(2 True  in 8)
    judge = judge.nonzero().reshape(len(judge),2,2)[...,1] # env,2【the bbox idx】

    return judge

# @torch.jit.script
def extract_z_rotation_torch_batch(batch_rot):
    """
    从批量torch.Tensor格式的四元数中提取仅绕Z轴的旋转（xyzw格式输入）
    
    参数：
        batch_rot: torch.Tensor，形状为(n,4)，格式为[x, y, z, w]
        
    返回：
        batch_z_rot: torch.Tensor，形状为(n,4)，格式为[x, y, z, w]（仅z和w有值，x=y=0）
    """
    device = batch_rot.device
    n = batch_rot.shape[0]
    
    # 1. 解析四元数分量（xyzw格式）
    x = batch_rot[:, 0]  # (n,)
    y = batch_rot[:, 1]  # (n,)
    z = batch_rot[:, 2]  # (n,)
    w = batch_rot[:, 3]  # (n,)
    
    # 2. 批量计算四元数转旋转矩阵（n,3,3）
    xx = x * x  # (n,)
    yy = y * y  # (n,)
    zz = z * z  # (n,)
    xy = x * y  # (n,)
    xz = x * z  # (n,)
    yz = y * z  # (n,)
    xw = x * w  # (n,)
    yw = y * w  # (n,)
    zw = z * w  # (n,)
    
    # 构造旋转矩阵的每个元素（向量化计算）
    r00 = 1 - 2 * (yy + zz)  # (n,)
    r01 = 2 * (xy - zw)     # (n,)
    r02 = 2 * (xz + yw)     # (n,)
    r10 = 2 * (xy + zw)     # (n,)
    r11 = 1 - 2 * (xx + zz)  # (n,)
    r12 = 2 * (yz - xw)     # (n,)
    r20 = 2 * (xz - yw)     # (n,)
    r21 = 2 * (yz + xw)     # (n,)
    r22 = 1 - 2 * (xx + yy)  # (n,)
    
    # 组合成旋转矩阵 (n,3,3)
    rot_matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)  # 形状: (n,3,3)
    
    # 3. 提取绕Z轴的旋转角度θ（仅保留XY平面旋转）
    cos_theta = rot_matrix[:, 0, 0]  # (n,)
    sin_theta = rot_matrix[:, 1, 0]  # (n,)
    theta = torch.atan2(sin_theta, cos_theta)  # (n,) 弧度值
    
    # 4. 构造纯Z轴旋转的四元数（xyzw格式）
    half_theta = theta / 2.0  # (n,)
    z_quat_w = torch.cos(half_theta)  # (n,) w分量
    z_quat_z = torch.sin(half_theta)  # (n,) z分量（绕Z轴旋转时x=y=0）
    
    # 5. 构造批量四元数 (n,4)，格式为[x,y,z,w]
    batch_z_rot = torch.zeros(n, 4, device=device)
    batch_z_rot[:, 2] = z_quat_z  # z分量
    batch_z_rot[:, 3] = z_quat_w  # w分量
    
    # 6. 归一化（处理数值误差）
    norms = torch.norm(batch_z_rot, dim=1, keepdim=True)  # (n,1)
    batch_z_rot = batch_z_rot / norms  # 形状: (n,4)
    
    return batch_z_rot

# @torch.jit.script
def extract_z_rotation_torch(box_rot):
    """
    从torch.Tensor格式的四元数中提取仅绕Z轴的旋转（xyzw格式输入）
    
    参数：
        box_rot: torch.Tensor，形状为(4,)，格式为[x, y, z, w]
        
    返回：
        z_rot: torch.Tensor，形状为(4,)，格式为[x, y, z, w]（仅z和w有值，x=y=0）
    """
    device = box_rot.device
    # 1. 解析四元数分量（xyzw格式）
    x, y, z, w = box_rot.unbind(dim=0)  # 从张量中拆分出x,y,z,w
    
    # 2. 四元数转旋转矩阵（3x3）
    # 公式参考：https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    
    # 构造旋转矩阵（3x3）
    rot_matrix = torch.tensor([
        [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
        [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
        [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
    ])
    
    # 3. 提取绕Z轴的旋转角度θ（仅保留XY平面旋转）
    cos_theta = rot_matrix[0, 0]  # R[0,0] = cosθ
    sin_theta = rot_matrix[1, 0]  # R[1,0] = sinθ
    theta = torch.atan2(sin_theta, cos_theta)  # 计算角度（弧度）
    
    # 4. 构造纯Z轴旋转的四元数（标准wxyz格式：[w, x, y, z]）
    half_theta = theta / 2.0
    z_quat_w = torch.cos(half_theta).to(device)    # w分量
    z_quat_z = torch.sin(half_theta).to(device)    # z分量（绕Z轴旋转时x=y=0）
    
    # 5. 转换回xyzw格式，并归一化（处理数值误差）
    z_rot = torch.stack([
        torch.tensor(0.0, device=device),  # x=0
        torch.tensor(0.0, device=device),  # y=0
        z_quat_z,                                  # z
        z_quat_w                                   # w
    ])
    z_rot = z_rot / torch.norm(z_rot)  # 归一化确保单位四元数
    
    return z_rot


def get_full_box_info_from_urdf(urdf_path, sample_points=1024, seed=2024):
    """
    从URDF文件中读取整体大盒子的信息并生成AABB和点云
    :param urdf_path: URDF文件路径
    :param sample_points: 采样点云数量
    :return: (aabb, points)
             aabb: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
             points: (N, 3)  numpy 数组
    """
    # 解析URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 找到 collision -> geometry -> box
    collision = root.find(".//collision/geometry/box")
    if collision is None:
        raise ValueError("URDF中找不到整体collision的box定义")

    # 读取尺寸
    size_str = collision.get("size")
    sx, sy, sz = map(float, size_str.split())

    # 计算AABB（大盒子以原点为中心）
    min_x, max_x = -sx/2, sx/2
    min_y, max_y = -sy/2, sy/2
    min_z, max_z = -sz/2, sz/2
    aabb = [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    # 创建虚拟box mesh
    box_mesh = trimesh.creation.box(extents=[sx, sy, sz])

    # 均匀采样表面点
    points, _ = trimesh.sample.sample_surface_even(box_mesh, count=sample_points, seed=seed)
    # object_points = to_torch(object_points - center)
    while points.shape[0] < 1024:
        points = torch.cat([points, points[:1024 - points.shape[0]]], dim=0)


    return aabb, points


def friction_to_gray_color(mu, mu_min, mu_max, gray_min=0.2, gray_max=0.9):
    """
    mu: 当前地面摩擦（用你写入 rigid shape 的那个值）
    mu_min, mu_max: 该阶段课程里预期的摩擦范围，用来映射颜色
    gray_min: 最暗的灰度（0=黑，1=白）。建议>=0.2避免太黑看不清
    gray_max: 最亮的灰度
    返回: gymapi.Vec3(r,g,b)，越大越亮；我们会按“摩擦越大越暗”取反
    """
    if mu_max <= mu_min:
        t = 0.0
    else:
        t = (mu - mu_min) / (mu_max - mu_min)
    t = float(np.clip(t, 0.0, 1.0))  # 0=最小摩擦，1=最大摩擦
    v = gray_max - t * (gray_max - gray_min)  # 反向: 摩擦越大颜色越暗
    return gymapi.Vec3(v, v, v)
def set_ground_color_for_env(gym, env_ptr, ground_actor, mu_value, mu_min, mu_max):
    color = friction_to_gray_color(mu_value, mu_min, mu_max,
                                   gray_min=0.2, gray_max=0.9)
    # 获取该 actor 的刚体名称并设置颜色（通常只有一个刚体）
    rb_names = gym.get_actor_rigid_body_names(env_ptr, ground_actor)  # list[str]
    for name in rb_names:
        rb_handle = gym.find_actor_rigid_body_handle(env_ptr, ground_actor, name) # 15+1+1
        # 只改可视网格
        gym.set_rigid_body_color(env_ptr, ground_actor, 0,
                                 gymapi.MESH_VISUAL, color)
