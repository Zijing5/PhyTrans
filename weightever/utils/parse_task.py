from weightever.env.tasks.humanoid import Humanoid
from weightever.env.tasks.humanoid_amp import HumanoidAMP
from weightever.env.tasks.vec_task_wrappers import VecTaskPythonWrapper


from isaacgym import rlgpu

import json
import numpy as np

from weightever.env.tasks.naviHaulTrans import naviHaulTrans


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [naviHaulTrans]")

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    # device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg["clip_observations"] = cfg_train.get("clip_observations", np.inf)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=args.device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
