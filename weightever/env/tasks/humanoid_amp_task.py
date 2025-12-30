import torch
import swanlab
import weightever.env.tasks.humanoid_amp as humanoid_amp
from isaacgym import gymapi
import os

class HumanoidAMPTask(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return

    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
            dataname = self.objname[0]
            env_ids = 0
            frame_id = self.progress_buf[env_ids]
            # self.save_images = True
            if self.save_images or self.trigger_save_images:

                # if self.what2do == 'play':
                #     frame_id = t
                # else:

                rgb_filename = "output/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("output/data/images/" + dataname +"/cam", exist_ok=True)
                print("[Viewer]Saving image to: ", "output/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id))
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
                self.trigger_save_images = False
            if (len(self.camera_handles) > 0 and self.trigger_save_images):
                rgb_filename2 = "output/data/images/" + dataname + "/cam/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0],self.camera_handles[0], gymapi.IMAGE_COLOR, rgb_filename2)
                print("[Camera]Saving image to: ", "output/data/images/" + dataname + "/cam/rgb_env%d_frame%05d.png" % (env_ids, frame_id))
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
            # if self.cfg['args'].swanlab_name != "":
            #     swanlab.log({"local_tar_standing_pos_x": task_obs[0,45]})
            #     swanlab.log({"local_tar_standing_pos_y": task_obs[0,46]})
            #     swanlab.log({"local_tar_standing_pos_z": task_obs[0,47]})


        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return