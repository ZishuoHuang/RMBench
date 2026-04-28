import argparse
import shutil
import importlib
import json
import os
import sys

import yaml
from sapien.render import clear_cache

sys.path.append("./")
from envs import *


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_cfg(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    emb = args.get("embodiment")

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        fp = emb_types[name]["file_path"]
        if fp is None:
            raise RuntimeError("missing embodiment files")
        return fp

    if len(emb) == 1:
        args["left_robot_file"] = emb_file(emb[0])
        args["right_robot_file"] = emb_file(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = emb_file(emb[0])
        args["right_robot_file"] = emb_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["task_config"] = task_config
    return args


def ensure_seed_file(save_path, seed_value, episode_id):
    os.makedirs(save_path, exist_ok=True)
    seed_path = os.path.join(save_path, "seed.txt")
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            vals = [int(x) for x in txt.split()] if txt else []
    else:
        vals = []

    if len(vals) <= episode_id:
        vals.extend([0] * (episode_id + 1 - len(vals)))
    vals[episode_id] = int(seed_value)

    with open(seed_path, "w", encoding="utf-8") as f:
        for v in vals:
            f.write(f"{int(v)} ")


def main():
    parser = argparse.ArgumentParser(description="Replay one saved trajectory PKL into a new HDF5 with target camera settings")
    parser.add_argument("task_name", type=str)
    parser.add_argument("source_task_config", type=str, help="config owning _traj_data/seed.txt")
    parser.add_argument("target_task_config", type=str, help="config defining camera/output settings")
    parser.add_argument("--source-episode", type=int, required=True)
    parser.add_argument("--target-episode", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    target_episode = args.source_episode if args.target_episode is None else args.target_episode

    task = class_decorator(args.task_name)
    cfg = load_cfg(args.task_name, args.target_task_config)

    source_save = os.path.join("data", args.task_name, args.source_task_config)
    target_save = os.path.join("data", args.task_name, args.target_task_config)

    seed_file = os.path.join(source_save, "seed.txt")
    if not os.path.isfile(seed_file):
        raise FileNotFoundError(f"source seed file missing: {seed_file}")

    with open(seed_file, "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]

    if args.source_episode < 0 or args.source_episode >= len(seeds):
        raise RuntimeError(f"source episode out of range: {args.source_episode}, available 0..{len(seeds)-1}")

    seed = int(seeds[args.source_episode])

    cfg["need_plan"] = False
    cfg["save_data"] = True
    cfg["collect_data"] = True
    cfg["render_freq"] = 0

    os.makedirs(target_save, exist_ok=True)
    ensure_seed_file(target_save, seed, target_episode)

    print(f"[ReplayH5] source={args.source_task_config}:episode{args.source_episode}, seed={seed}")
    print(f"[ReplayH5] target={args.target_task_config}:episode{target_episode}")

    cfg["save_path"] = target_save
    task.setup_demo(now_ep_num=target_episode, seed=seed, **cfg)

    src_traj = os.path.join(source_save, "_traj_data", f"episode{args.source_episode}.pkl")
    dst_traj_dir = os.path.join(target_save, "_traj_data")
    dst_traj = os.path.join(dst_traj_dir, f"episode{args.source_episode}.pkl")
    os.makedirs(dst_traj_dir, exist_ok=True)
    if not os.path.isfile(src_traj):
        raise FileNotFoundError(f"source traj missing: {src_traj}")
    shutil.copy2(src_traj, dst_traj)

    traj = task.load_tran_data(args.source_episode)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    info_file = os.path.join(target_save, "scene_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            info_db = json.load(f)
    else:
        info_db = {}

    info = task.play_once()
    info_db[f"episode_{target_episode}"] = info
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info_db, f, ensure_ascii=False, indent=2)

    task.close_env(clear_cache=True)
    clear_cache()
    task.merge_pkl_to_hdf5_video()
    task.remove_data_cache()

    h5_path = os.path.join(target_save, "data", f"episode{target_episode}.hdf5")
    print(f"[ReplayH5] done: {h5_path}")


if __name__ == "__main__":
    main()
