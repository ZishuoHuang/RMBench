import argparse
import importlib
import os
import sys

import numpy as np
import sapien.core as sapien
import yaml

sys.path.append("./")
from envs import *


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_args(task_name, task_config):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    emb = args.get("embodiment")

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_emb_file(name):
        robot_file = emb_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(emb) == 1:
        args["left_robot_file"] = get_emb_file(emb[0])
        args["right_robot_file"] = get_emb_file(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = get_emb_file(emb[0])
        args["right_robot_file"] = get_emb_file(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    return args


def get_entity_scene_id(entity):
    fn = getattr(entity, "get_per_scene_id", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    val = getattr(entity, "per_scene_id", None)
    if val is not None:
        return int(val)

    ent = getattr(entity, "entity", None)
    ent_val = getattr(ent, "per_scene_id", None) if ent is not None else None
    if ent_val is not None:
        return int(ent_val)

    return None


def build_scene_entity_map(task_env):
    entity_map = {}

    for actor in task_env.scene.get_all_actors():
        sid = get_entity_scene_id(actor)
        if sid is not None:
            entity_map[sid] = actor

    for link in task_env.robot.left_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    for link in task_env.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    return entity_map


def backproject_pixels_to_world(u, v, depth, K, T_c2w):
    depth = depth * 0.001
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = -(v - K[1, 2]) * depth / K[1, 1]
    z = -depth
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_world = (T_c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def create_markers(scene, n, radius):
    markers = []
    colors = []
    for i in range(n):
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius)
        marker = builder.build_kinematic(name=f"track_marker_{i}")
        markers.append(marker)
        colors.append([
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
        ])
    return markers, colors


def colorize_markers(markers, colors):
    for marker, c in zip(markers, colors):
        try:
            render_body = marker.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                shape.material.set_base_color([c[0], c[1], c[2], 1.0])
        except Exception:
            continue


def split_robot_and_object_ids(task, seg_ids):
    robot_ids = set()
    for link in task.robot.left_entity.get_links() + task.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            robot_ids.add(int(sid))

    robot_visible = [int(s) for s in seg_ids if int(s) in robot_ids]
    object_visible = [int(s) for s in seg_ids if int(s) not in robot_ids]
    return robot_visible, object_visible


def main():
    parser = argparse.ArgumentParser(description="Replay dense point attachment from a configurable camera, including robot links")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--camera", type=str, default="front_camera", help="Source camera for frame-0 point sampling")
    parser.add_argument("--num-points", type=int, default=3000)
    parser.add_argument("--robot-ratio", type=float, default=0.45, help="Fraction of samples reserved for robot links")
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--exclude-background", action="store_true", help="Ignore pixels with seg id 0")
    args = parser.parse_args()

    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)

    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False
    cfg["render_freq"] = 1

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]

    ep = args.episode
    if ep < 0 or ep >= len(seeds):
        raise RuntimeError(f"episode out of range: {ep}, available 0..{len(seeds)-1}")

    task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg)

    traj = task.load_tran_data(ep)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    obs0 = task.get_obs()
    cam_obs = obs0["observation"][args.camera]
    depth0 = cam_obs["depth"].astype(np.float32)
    seg0 = cam_obs["actor_segmentation_raw"].astype(np.int32)
    K0 = cam_obs["intrinsic_cv"].astype(np.float32)
    T_c2w0 = cam_obs["cam2world_gl"].astype(np.float32)

    entity_map = build_scene_entity_map(task)
    visible_ids = np.unique(seg0[(~np.isnan(depth0)) & (depth0 > 0) & (seg0 > 0)])
    robot_ids = set()
    for link in task.robot.left_entity.get_links() + task.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            robot_ids.add(int(sid))

    robot_visible = [int(s) for s in visible_ids if int(s) in robot_ids]
    object_visible = [int(s) for s in visible_ids if int(s) not in robot_ids]

    if len(robot_visible) == 0 and len(object_visible) == 0:
        raise RuntimeError("No valid depth+seg pixels in frame 0")

    target_robot = int(round(args.num_points * args.robot_ratio))
    target_object = max(0, args.num_points - target_robot)

    def sample_for_ids(ids, quota):
        if quota <= 0 or len(ids) == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        per_id = max(1, quota // len(ids))
        sampled_u = []
        sampled_v = []
        for seg_id in ids:
            mask = (seg0 == seg_id) & (~np.isnan(depth0)) & (depth0 > 0)
            if args.exclude_background:
                mask &= (seg0 > 0)
            vv, uu = np.where(mask)
            if len(uu) == 0:
                continue
            take = min(len(uu), per_id)
            pick = np.random.choice(len(uu), take, replace=False)
            sampled_u.append(uu[pick])
            sampled_v.append(vv[pick])
        if len(sampled_u) == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.concatenate(sampled_u), np.concatenate(sampled_v)

    uu_robot, vv_robot = sample_for_ids(robot_visible, target_robot)
    uu_obj, vv_obj = sample_for_ids(object_visible, target_object)

    uu = np.concatenate([uu_robot, uu_obj])
    vv = np.concatenate([vv_robot, vv_obj])

    if len(uu) == 0:
        vv, uu = np.where((~np.isnan(depth0)) & (depth0 > 0) & (seg0 > 0))
        if len(uu) == 0:
            raise RuntimeError("No valid depth+seg pixels in frame 0")
        if len(uu) > args.num_points:
            sel = np.random.choice(len(uu), args.num_points, replace=False)
            uu = uu[sel]
            vv = vv[sel]

    if len(uu) > args.num_points:
        sel = np.random.choice(len(uu), args.num_points, replace=False)
        uu = uu[sel]
        vv = vv[sel]

    d = depth0[vv, uu]
    sid = seg0[vv, uu]
    p_world0 = backproject_pixels_to_world(uu.astype(np.float32), vv.astype(np.float32), d, K0, T_c2w0)
    cloud_min = np.min(p_world0, axis=0)
    cloud_max = np.max(p_world0, axis=0)

    owner_sid = []
    local_points = []
    init_world = []

    for p, s in zip(p_world0, sid):
        entity = entity_map.get(int(s), None)
        if entity is None:
            continue
        pose = entity.get_pose().to_transformation_matrix().astype(np.float32)
        pose_inv = np.linalg.inv(pose)
        p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
        p_local = (pose_inv @ p_h)[:3]
        owner_sid.append(int(s))
        local_points.append(p_local)
        init_world.append(p)

    if len(local_points) == 0:
        raise RuntimeError("Failed to attach any sampled points to scene entities")

    local_points = np.asarray(local_points, dtype=np.float32)
    init_world = np.asarray(init_world, dtype=np.float32)

    markers, colors = create_markers(task.scene, len(local_points), radius=args.marker_radius)
    colorize_markers(markers, colors)

    original_update_render = task._update_render

    def wrapped_update_render():
        current_map = build_scene_entity_map(task)
        for i, marker in enumerate(markers):
            s = owner_sid[i]
            entity = current_map.get(s, None)
            if entity is not None:
                pose = entity.get_pose().to_transformation_matrix().astype(np.float32)
                p_local_h = np.array([local_points[i, 0], local_points[i, 1], local_points[i, 2], 1.0], dtype=np.float32)
                p_world = (pose @ p_local_h)[:3]
            else:
                p_world = init_world[i]
            marker.set_pose(sapien.Pose(p=p_world.tolist()))
        original_update_render()

    task._update_render = wrapped_update_render

    robot_count = sum(1 for s in owner_sid if s in robot_ids)
    object_count = len(owner_sid) - robot_count
    print(f"[Replay] episode={ep}, camera={args.camera}, sampled_points={len(local_points)}")
    print(f"[Replay] robot_points={robot_count}, object_points={object_count}")
    print(f"[Replay] frame0 world bounds min={cloud_min} max={cloud_max}")
    print("Close the SAPIEN viewer window to exit.")

    task.play_once()

    while True:
        try:
            task._update_render()
            task.viewer.render()
            task.scene.step()
        except Exception:
            break


if __name__ == "__main__":
    main()