#!/usr/bin/env python3
"""
Export sid->body mapping for one episode with minimal simulator setup.
This is intended for legacy datasets that do not yet store scene_state/sid_to_body_key_json.
"""

import argparse
import importlib
import json
import os
import sys

import yaml

sys.path.append("./")
from envs import *  # noqa: F401,F403


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_args(task_name, task_config):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    embodiment_type = args.get("embodiment")

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_emb_file(name):
        return emb_types[name]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_emb_file(embodiment_type[0])
        args["right_robot_file"] = get_emb_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_emb_file(embodiment_type[0])
        args["right_robot_file"] = get_emb_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment config length should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    return args


def get_entity_scene_id(entity):
    fn = getattr(entity, "get_per_scene_id", None)
    if callable(fn):
        sid = fn()
        if sid is not None:
            return int(sid)

    val = getattr(entity, "per_scene_id", None)
    if val is not None:
        return int(val)

    ent = getattr(entity, "entity", None)
    ent_val = getattr(ent, "per_scene_id", None) if ent is not None else None
    if ent_val is not None:
        return int(ent_val)

    return None


def build_sid_map(task):
    sid_map = {}

    for actor in task.scene.get_all_actors():
        sid = get_entity_scene_id(actor)
        if sid is None:
            continue
        sid_map[sid] = f"rigid::{actor.get_name()}"

    get_all_articulations = getattr(task.scene, "get_all_articulations", None)
    if callable(get_all_articulations):
        for idx, articulation in enumerate(get_all_articulations()):
            art_name = getattr(articulation, "get_name", lambda: "")()
            if not art_name:
                art_name = f"articulation_{idx}"
            for link in articulation.get_links():
                sid = get_entity_scene_id(link)
                if sid is None:
                    continue
                sid_map[sid] = f"art::{art_name}::{link.get_name()}"

    return sid_map


def main():
    parser = argparse.ArgumentParser(description="Export sid->body map for one episode")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]
    if args.episode < 0 or args.episode >= len(seeds):
        raise RuntimeError(f"episode out of range: {args.episode}")

    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False

    task.setup_demo(now_ep_num=args.episode, seed=seeds[args.episode], **cfg)

    traj = task.load_tran_data(args.episode)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    task.get_obs()
    sid_map = build_sid_map(task)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sid_map.items()}, f, indent=2)

    if hasattr(task, "close"):
        task.close()

    print(f"Saved sid map with {len(sid_map)} entries to {args.output}")


if __name__ == "__main__":
    main()
