import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(
    task_name=None,
    task_config=None,
    episode_start=None,
    episode_end=None,
    scene_info_input=None,
    override_use_seed=None,
    override_collect_data=None,
    override_episode_num=None,
):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    if override_use_seed is not None:
        args["use_seed"] = bool(override_use_seed)
    if override_collect_data is not None:
        args["collect_data"] = bool(override_collect_data)
    if override_episode_num is not None:
        args["episode_num"] = int(override_episode_num)

    # Prefer external camera candidates produced by interactive picker.
    camera_cfg = args.setdefault("camera", {})
    if bool(camera_cfg.get("record_single_camera", False)):
        record_camera_name = str(camera_cfg.get("record_camera_name", "world_camera1"))
        auto_candidates_file = os.path.join(
            "task_config",
            "camera_candidates",
            f"{task_config}_{record_camera_name}.yml",
        )
        if os.path.exists(auto_candidates_file):
            camera_cfg["random_camera_candidates_file"] = auto_candidates_file
            print(f"[INFO] Use camera candidates file: {auto_candidates_file}")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["scene_info_input"] = scene_info_input
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args, episode_start=episode_start, episode_end=episode_end)


def run(TASK_ENV, args, episode_start=None, episode_end=None):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    scene_info_db = None

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        seed_file = os.path.join(args["save_path"], "seed.txt")
        if not os.path.exists(seed_file):
            raise RuntimeError(
                f"seed file not found: {seed_file}. "
                "Please generate seeds first (set use_seed=false or use --override-use-seed false)."
            )
        with open(seed_file, "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

        if len(seed_list) == 0:
            raise RuntimeError(
                f"seed file is empty: {seed_file}. "
                "Please generate seeds first (set use_seed=false or use --override-use-seed false)."
            )

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        scene_info_input = args.get("scene_info_input")
        if scene_info_input:
            if os.path.isfile(scene_info_input):
                with open(scene_info_input, "r", encoding="utf-8") as file:
                    scene_info_db = json.load(file)
                print(f"Use scene_info input: {scene_info_input}")
            else:
                print(f"[WARN] scene_info input not found, ignore: {scene_info_input}")

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        if episode_start is None and episode_end is None:
            start_idx = 0
            end_idx = args["episode_num"]
        else:
            start_idx = 0 if episode_start is None else max(0, int(episode_start))
            end_idx = args["episode_num"] if episode_end is None else min(args["episode_num"], int(episode_end))
            if end_idx <= start_idx:
                print(f"Skip replay range [{start_idx}, {end_idx})")
                return

        if len(seed_list) <= start_idx:
            raise RuntimeError(
                f"Not enough seeds for replay range [{start_idx}, {end_idx}). "
                f"seed_count={len(seed_list)}. "
                "Generate more seeds first (set use_seed=false or use --override-use-seed false)."
            )

        if len(seed_list) < end_idx:
            print(
                f"[WARN] seed count ({len(seed_list)}) is smaller than requested end ({end_idx}). "
                f"Replay range will be truncated to [{start_idx}, {len(seed_list)})."
            )
            end_idx = len(seed_list)

        # Use per-range scene_info file in parallel replay to avoid write races.
        if episode_start is None and episode_end is None:
            info_file_path = os.path.join(args["save_path"], "scene_info.json")
        else:
            info_file_path = os.path.join(args["save_path"], f"scene_info_{start_idx}_{end_idx}.json")

        if not os.path.exists(info_file_path):
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump({}, file, ensure_ascii=False)

        failed_episodes = []

        for episode_idx in range(start_idx, end_idx):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")
            if exist_hdf5(episode_idx):
                print(f"Skip existing hdf5 for episode {episode_idx}")
                continue
            try:
                if scene_info_db is not None:
                    args["scene_info_episode"] = scene_info_db.get(f"episode_{episode_idx}")
                else:
                    args["scene_info_episode"] = None

                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                TASK_ENV.merge_pkl_to_hdf5_video()
                TASK_ENV.remove_data_cache()

                if not TASK_ENV.check_success():
                    print(f"Collect Error at episode {episode_idx}, skip")
                    failed_episodes.append(episode_idx)
            except Exception as e:
                import traceback
                print(" -------------")
                print(f"collect data episode {episode_idx} fail")
                print("Error: ", e)
                traceback.print_exc()
                print(" -------------")
                failed_episodes.append(episode_idx)
                try:
                    TASK_ENV.close_env(clear_cache=True)
                except Exception:
                    pass
                time.sleep(0.3)
                continue

        if failed_episodes:
            raise RuntimeError(
                f"Collection failed for episodes: {failed_episodes}. "
                "Please check logs above for the first exception."
            )

        if episode_start is None and episode_end is None:
            command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
            os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        raise ValueError(f"Invalid bool value: {v}")

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--scene-info-input", type=str, default=None)
    parser.add_argument("--override-use-seed", type=str, default=None)
    parser.add_argument("--override-collect-data", type=str, default=None)
    parser.add_argument("--override-episode-num", type=int, default=None)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config
    episode_start = parser.episode_start
    episode_end = parser.episode_end
    scene_info_input = parser.scene_info_input
    override_use_seed = None if parser.override_use_seed is None else str2bool(parser.override_use_seed)
    override_collect_data = None if parser.override_collect_data is None else str2bool(parser.override_collect_data)
    override_episode_num = parser.override_episode_num

    main(
        task_name=task_name,
        task_config=task_config,
        episode_start=episode_start,
        episode_end=episode_end,
        scene_info_input=scene_info_input,
        override_use_seed=override_use_seed,
        override_collect_data=override_collect_data,
        override_episode_num=override_episode_num,
    )
