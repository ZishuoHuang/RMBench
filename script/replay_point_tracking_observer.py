import argparse
import os
import json
import subprocess
import numpy as np
import cv2
from PIL import Image
import sys
sys.path.append("./") # Add this line so Python can find the 'envs' module
try:
    import torch
except ImportError:
    torch = None

from envs import *
import yaml

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

import sapien.core as sapien
import importlib

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No such task")
    return env_instance

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
            entity_map[int(sid)] = actor
            
    # 正确挂载机器人的 Links
    for link in task_env.robot.left_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[int(sid)] = link

    for link in task_env.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[int(sid)] = link
            
    return entity_map

def get_background_scene_ids(task_env):
    background_ids = set()
    table = getattr(task_env, 'table', None)
    if table:
        sid = get_entity_scene_id(table)
        if sid is not None: background_ids.add(int(sid))
    return background_ids

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

def resolve_source_camera(task, camera_name):
    # Try looking in 'task.camera' first, then fallback to 'task'
    camera_container = getattr(task, "cameras", getattr(task, "camera", task))
    obs = getattr(camera_container, "observer_camera", getattr(task, "observer_camera", None))
    world1 = getattr(camera_container, "world_camera1", getattr(task, "world_camera1", None))
    world2 = getattr(camera_container, "world_camera2", getattr(task, "world_camera2", None))
    head_cam = getattr(task.robot, "head_camera", getattr(camera_container, "head_camera", getattr(task, "head_camera", None)))
    left_cam = getattr(task.robot, "left_camera", getattr(camera_container, "left_camera", getattr(task, "left_camera", None)))
    right_cam = getattr(task.robot, "right_camera", getattr(camera_container, "right_camera", getattr(task, "right_camera", None)))

    static_names = []
    if hasattr(task, "static_cameras"):
        static_names = list(task.static_cameras.keys())
    
    if camera_name == "observer_camera" and obs is not None: return obs
    if camera_name == "world_camera1" and world1 is not None: return world1
    if camera_name == "world_camera2" and world2 is not None: return world2
    if camera_name == "head_camera" and head_cam is not None: return head_cam
    if camera_name == "left_camera" and left_cam is not None: return left_cam
    if camera_name == "right_camera" and right_cam is not None: return right_cam
    for name in static_names:
        if camera_name == name:
            return task.static_cameras[name]
    raise KeyError(f"Unknown camera '{camera_name}'.")

def camera_points_to_world(camera, points_cam):
    T_c2w = camera.get_model_matrix().astype(np.float32)
    return points_cam @ T_c2w[:3, :3].T + T_c2w[:3, 3]

def world_points_to_camera_matrix(points_world, T_c2w):
    T_w2c = np.linalg.inv(T_c2w.astype(np.float32))
    return points_world @ T_w2c[:3, :3].T + T_w2c[:3, 3]

def transform_local_points(local_points, pose):
    return local_points @ pose[:3, :3].T + pose[:3, 3]

def get_entity_pose_matrix(entity):
    get_entity_pose = getattr(entity, "get_entity_pose", None)
    if callable(get_entity_pose):
        pose = get_entity_pose()
    else:
        pose = entity.get_pose()
    return pose.to_transformation_matrix().astype(np.float32)

def build_keyframe_indices(num_frames, num_keyframes=5):
    if num_frames <= 1: return [0]*num_keyframes
    num_keyframes = max(1, int(num_keyframes))
    if num_keyframes == 1: return [int(num_frames//2)]
    raw = [int(round(i * (num_frames - 1) / float(num_keyframes - 1))) for i in range(num_keyframes)]
    keyframes = []
    for idx in raw:
        if idx >= num_frames: idx = num_frames - 1
        keyframes.append(idx)
    return keyframes

def sample_reference_points(camera, entity_map, background_ids, include_background):
    camera.take_picture()
    position = camera.get_picture("Position")
    segmentation = camera.get_picture("Segmentation")
    seg_raw = segmentation[..., 1].astype(np.int32)
    valid = position[..., 3] < 1
    if not include_background:
        valid &= ~np.isin(seg_raw, list(background_ids))
        valid &= seg_raw > 0
    vv, uu = np.where(valid)
    if len(uu) == 0: return None
    points_world = camera_points_to_world(camera, position[vv, uu, :3].astype(np.float32))
    owner_sid = seg_raw[vv, uu].astype(np.int32)
    keep_uu, keep_vv, keep_sid, local_points, init_world = [], [], [], [], []
    ref_pose_by_sid = {}
    for p_world, sid, u, v in zip(points_world, owner_sid, uu, vv):
        entity = entity_map.get(int(sid), None)
        if entity is None: continue
        pose = get_entity_pose_matrix(entity)
        pose_inv = np.linalg.inv(pose)
        p_local = (pose_inv @ np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32))[:3]
        keep_uu.append(int(u)); keep_vv.append(int(v)); keep_sid.append(int(sid))
        local_points.append(p_local); init_world.append(p_world)
        ref_pose_by_sid[int(sid)] = pose
    if len(local_points) == 0: raise RuntimeError("Failed to attach reference points")
    return {
        "position": position, "seg_raw": seg_raw, "shape": position.shape[:2],
        "uu": np.asarray(keep_uu, dtype=np.int32), "vv": np.asarray(keep_vv, dtype=np.int32),
        "owner_sid": np.asarray(keep_sid, dtype=np.int32),
        "local_points": np.asarray(local_points, dtype=np.float32),
        "init_world": np.asarray(init_world, dtype=np.float32),
        "ref_camera_matrix": camera.get_model_matrix().astype(np.float32),
        "ref_pose_by_sid": ref_pose_by_sid,
    }

def reconstruct_reference_flow(snapshot, frame_pose_history, output_dir, flow_name, save_dense=True):
    local_points = snapshot["local_points"]
    owner_sid = snapshot["owner_sid"]
    uu = snapshot["uu"]; vv = snapshot["vv"]
    h, w = snapshot["shape"]
    ref_c2w = snapshot["ref_camera_matrix"]
    ref_pose_by_sid = snapshot["ref_pose_by_sid"]
    T = len(frame_pose_history)
    sid_to_indices = {}
    for idx, sid in enumerate(owner_sid): sid_to_indices.setdefault(int(sid), []).append(idx)
    anchor_world = snapshot["init_world"]
    anchor_cam = world_points_to_camera_matrix(anchor_world, ref_c2w)
    anchor_dense = np.zeros((h, w, 3), dtype=np.float32)
    anchor_dense[vv, uu] = anchor_cam
    suffix = flow_name.split("scene_point_flow_")[-1]
    flow_path = os.path.join(output_dir, f"{flow_name}.npy")
    anchor_path = os.path.join(output_dir, f"{flow_name}.anchor.npy")
    seg_path = os.path.join(output_dir, f"segmentation_{suffix}.npy")
    if save_dense:
        flow_mm = np.lib.format.open_memmap(flow_path, mode="w+", dtype=np.float16, shape=(T, h, w, 3))
        anchor_mm = np.lib.format.open_memmap(anchor_path, mode="w+", dtype=np.float16, shape=(h, w, 3))
        seg_mm = np.lib.format.open_memmap(seg_path, mode="w+", dtype=np.int32, shape=(h, w))
        anchor_mm[...] = anchor_dense.astype(np.float16)
        seg_mm[...] = 0; seg_mm[vv, uu] = snapshot["seg_raw"][vv, uu].astype(np.int32)
        for t, pose_map in enumerate(frame_pose_history):
            for sid, idx_list in sid_to_indices.items():
                idx_arr = np.asarray(idx_list, dtype=np.int64)
                pose = pose_map.get(int(sid), None)
                if pose is None: pose = ref_pose_by_sid.get(int(sid), np.eye(4, dtype=np.float32))
                pts_world = transform_local_points(local_points[idx_arr], pose)
                pts_cam = world_points_to_camera_matrix(pts_world, ref_c2w)
                flow_mm[t, vv[idx_arr], uu[idx_arr]] = pts_cam.astype(np.float16)
        flow_mm.flush(); anchor_mm.flush(); seg_mm.flush()
        return {"flow_path": flow_path, "anchor_path": anchor_path, "seg_path": seg_path, "shape": [T, h, w, 3]}

    flow = np.zeros((T, len(local_points), 3), dtype=np.float32)
    for t, pose_map in enumerate(frame_pose_history):
        for sid, idx_list in sid_to_indices.items():
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            pose = pose_map.get(int(sid), None)
            if pose is None: pose = ref_pose_by_sid.get(int(sid), np.eye(4, dtype=np.float32))
            flow[t, idx_arr] = transform_local_points(local_points[idx_arr], pose)
    np.save(flow_path, flow.astype(np.float16)); np.save(anchor_path, anchor_world.astype(np.float32)); np.save(seg_path, snapshot["seg_raw"][vv, uu].astype(np.int32))
    return {"flow_path": flow_path, "anchor_path": anchor_path, "seg_path": seg_path, "shape": list(flow.shape)}

def get_world_points_from_valid_pixels(camera, position_texture, valid_mask, prefer_cuda=True):
    if prefer_cuda and torch is not None and torch.cuda.is_available():
        try:
            pos_cuda = camera.get_picture_cuda("Position").torch()
            valid_cuda = pos_cuda[..., 3] < 1
            points_cam = pos_cuda[..., :3][valid_cuda]
            model_matrix = torch.as_tensor(camera.get_model_matrix(), dtype=torch.float32, device=points_cam.device)
            points_world = points_cam @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            pixel_idx = torch.nonzero(valid_cuda, as_tuple=False)
            vv = pixel_idx[:, 0].cpu().numpy().astype(np.int32)
            uu = pixel_idx[:, 1].cpu().numpy().astype(np.int32)
            return points_world.cpu().numpy().astype(np.float32), uu, vv
        except Exception: pass
    vv, uu = np.where(valid_mask)
    if len(uu) == 0: return np.empty((0, 3), dtype=np.float32), uu.astype(np.int32), vv.astype(np.int32)
    points_cam = position_texture[vv, uu, :3].astype(np.float32)
    points_world = camera_points_to_world(camera, points_cam)
    return points_world.astype(np.float32), uu.astype(np.int32), vv.astype(np.int32)

def to_uint8_rgb(image):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        return None

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        max_val = float(np.nanmax(arr_f)) if arr_f.size > 0 else 1.0
        if max_val <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

def start_ffmpeg_mp4_writer(output_path, width, height, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(max(1.0, fps))),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser(description="Replay point tracking from observer_camera viewpoint")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--camera", type=str, default="observer_camera")
    parser.add_argument("--sampling-mode", type=str, default="pixel_all", choices=["pixel_all", "uniform_grid"])
    parser.add_argument("--num-points", type=int, default=5000)
    parser.add_argument("--max-track-points", type=int, default=5000)
    parser.add_argument("--no-cuda-pcd", action="store_true")
    parser.add_argument("--robot-ratio", type=float, default=0.45)
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--record-camera", type=str, default="head_camera")
    parser.add_argument("--gif-path", type=str, default=None)
    parser.add_argument("--gif-duration", type=int, default=80)
    parser.add_argument("--gif-stride", type=int, default=1)
    parser.add_argument("--gif-max-frames", type=int, default=600)
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--mp4-fps", type=float, default=12.0)
    parser.add_argument("--save-sceneflow", action="store_true")
    parser.add_argument("--sceneflow-dir", type=str, default=None)
    parser.add_argument("--sceneflow-keyframes", type=int, default=5)
    parser.add_argument("--exact-length", type=int, default=0, help="Wait for dry pass to count length automatically")
    args = parser.parse_args()

    # Pass 1: DRy Run to find exact length (1881 frames or whatever)
    if args.exact_length > 0:
        traj_len = args.exact_length
        print(f"[ObserverReplay] Using explicit length {traj_len}")
    else:
        print("[ObserverReplay] Doing a FAST dry-run to count the exact number of physics frames in the entire motion...")
        dry_task = class_decorator(args.task_name)
        cfg_dry = load_args(args.task_name, args.task_config)
        cfg_dry["need_plan"] = False; cfg_dry["save_data"] = False
        with open(os.path.join(cfg_dry["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
            seeds = [int(x) for x in f.read().split()]
        ep = args.episode
        dry_task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg_dry)
        traj = dry_task.load_tran_data(ep)
        cfg_dry["left_joint_path"] = traj["left_joint_path"]
        cfg_dry["right_joint_path"] = traj["right_joint_path"]
        dry_task.set_path_lst(cfg_dry)
        total_sim_frames = 0
        def dry_update():
            nonlocal total_sim_frames; total_sim_frames += 1
            if total_sim_frames % 500 == 0: print(f"  dry-run -> {total_sim_frames} frames", end='\\r')
        dry_task._update_render = dry_update
        try: dry_task.play_once()
        except: pass
        if hasattr(dry_task, "close"): dry_task.close()
        traj_len = max(1, total_sim_frames)
        print(f"\\n[ObserverReplay] Dry run complete! Measured exactly {traj_len} frames of playback.")

    keyframe_targets = build_keyframe_indices(traj_len, args.sceneflow_keyframes)

    # Pass 2
    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)
    cfg["need_plan"] = False; cfg["save_data"] = False; cfg["collect_data"] = False
    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]
    ep = args.episode
    task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg)
    traj = task.load_tran_data(ep)
    cfg["left_joint_path"] = traj["left_joint_path"]; cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    task.get_obs()
    source_camera = resolve_source_camera(task, args.camera)
    try: record_camera = resolve_source_camera(task, args.record_camera)
    except Exception: record_camera = source_camera
    source_camera.take_picture()
    if record_camera is not source_camera: record_camera.take_picture()
    position = source_camera.get_picture("Position")
    segmentation = source_camera.get_picture("Segmentation")
    seg_raw = segmentation[..., 1].astype(np.int32)
    entity_map = build_scene_entity_map(task)
    background_ids = get_background_scene_ids(task)
    robot_ids = set()
    for link in task.robot.left_entity.get_links() + task.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None: robot_ids.add(int(sid))
    valid = position[..., 3] < 1
    if not args.include_background:
        background_id_arr = np.array(sorted(background_ids), dtype=np.int32)
        if len(background_id_arr) > 0: valid &= ~np.isin(seg_raw, background_id_arr)
        valid &= seg_raw > 0

    h, w = position.shape[:2]
    if args.sampling_mode == "pixel_all":
        p_world0, uu, vv = get_world_points_from_valid_pixels(source_camera, position, valid, prefer_cuda=(not args.no_cuda_pcd))
        if p_world0 is None: raise RuntimeError("CUDA returned None, fallback failed")
        sid = seg_raw[vv, uu] if len(uu) > 0 else np.empty((0,), dtype=np.int32)
    else:
        num_points = max(1, int(args.num_points)); grid_area = h * w
        target_density = num_points / max(1, grid_area); grid_step = max(1, int(np.sqrt(1.0 / target_density)))
        vv_grid, uu_grid = np.meshgrid(np.arange(0, h, grid_step), np.arange(0, w, grid_step), indexing="ij")
        vv_grid = vv_grid.flatten().astype(np.int32); uu_grid = uu_grid.flatten().astype(np.int32)
        valid_mask = valid[vv_grid, uu_grid]; uu = uu_grid[valid_mask]; vv = vv_grid[valid_mask]
        if len(uu) > num_points:
            sel = np.random.choice(len(uu), num_points, replace=False); uu = uu[sel]; vv = vv[sel]
        sid = seg_raw[vv, uu]
        if len(uu) > 0:
            p_cam = position[vv, uu, :3].astype(np.float32)
            p_world0 = camera_points_to_world(source_camera, p_cam).astype(np.float32)
        else: p_world0 = np.empty((0, 3), dtype=np.float32)

    if len(p_world0) == 0: raise RuntimeError("No observer-camera points remain")

    if args.max_track_points > 0 and len(p_world0) > args.max_track_points:
        stride = max(1, int(np.ceil(len(p_world0) / float(args.max_track_points))))
        keep_idx = np.arange(0, len(p_world0), stride, dtype=np.int64)[: args.max_track_points]
        p_world0 = p_world0[keep_idx]; uu = uu[keep_idx]; vv = vv[keep_idx]; sid = sid[keep_idx]
    
    owner_sid, local_points, init_world = [], [], []
    for p, s in zip(p_world0, sid):
        entity = entity_map.get(int(s), None)
        if entity is None: continue
        pose = get_entity_pose_matrix(entity)
        pose_inv = np.linalg.inv(pose)
        p_local = (pose_inv @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32))[:3]
        owner_sid.append(int(s)); local_points.append(p_local); init_world.append(p)
    local_points = np.asarray(local_points, dtype=np.float32)
    init_world = np.asarray(init_world, dtype=np.float32)

    markers, colors = create_markers(task.scene, len(local_points), radius=args.marker_radius)
    colorize_markers(markers, colors)

    original_update_render = task._update_render
    gif_frames, video_writer = [], None
    video_ffmpeg = None
    video_writer_mode = None
    render_frame_idx, target_cursor = 0, 0
    max_gif_frames = None if int(args.gif_max_frames) <= 0 else int(args.gif_max_frames)
    gif_cap_warned = False
    frame_pose_history, reference_snapshots = [], []

    sid_to_indices = {}
    for idx, s in enumerate(owner_sid): sid_to_indices.setdefault(int(s), []).append(idx)

    if args.save_sceneflow:
        if args.sceneflow_dir is None: args.sceneflow_dir = os.path.join(cfg["save_path"], f"sceneflow_ep{ep}_{args.camera}")
        os.makedirs(args.sceneflow_dir, exist_ok=True)
        print(f"[ObserverReplay] Planned exactly matching keyframe targets: {keyframe_targets}")

    if args.save_gif and args.gif_path is None:
        args.gif_path = os.path.join(cfg["save_path"], f"observer_replay_ep{ep}_{args.record_camera}.gif")
    if args.save_mp4 and args.mp4_path is None:
        args.mp4_path = os.path.join(cfg["save_path"], f"observer_replay_ep{ep}_{args.record_camera}.mp4")

    if args.save_gif and args.gif_path is not None:
        os.makedirs(os.path.dirname(args.gif_path), exist_ok=True)
    if args.save_mp4 and args.mp4_path is not None:
        os.makedirs(os.path.dirname(args.mp4_path), exist_ok=True)

    def wrapped_update_render():
        nonlocal video_writer, video_ffmpeg, video_writer_mode, gif_cap_warned, render_frame_idx, target_cursor
        current_map = build_scene_entity_map(task)
        frame_pose_history.append({
            int(s): get_entity_pose_matrix(entity)
            for s, entity in current_map.items()
        })
        for s, idx_list in sid_to_indices.items():
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            entity = current_map.get(int(s), None)
            if entity is not None:
                pose = get_entity_pose_matrix(entity)
                p_world = local_points[idx_arr] @ pose[:3, :3].T + pose[:3, 3]
            else: p_world = init_world[idx_arr]
            for j, marker_idx in enumerate(idx_arr.tolist()):
                markers[marker_idx].set_pose(sapien.Pose(p=p_world[j].tolist()))
        original_update_render()
        print(f"[ObserverReplay] Pass 2 Rendering... Frame {render_frame_idx}/{traj_len}", end="\r")

        if args.save_sceneflow and target_cursor < len(keyframe_targets) and render_frame_idx == keyframe_targets[target_cursor]:
            try:
                snap = sample_reference_points(source_camera, current_map, background_ids, args.include_background)
                if snap is not None:
                    snap["ref_frame_idx"] = render_frame_idx
                    snap["keyframe_target"] = keyframe_targets[target_cursor]
                    reference_snapshots.append(snap)
                    print(f"[ObserverReplay] captured reference point cloud from frame={render_frame_idx} (target={keyframe_targets[target_cursor]}) out of {traj_len}")
                    target_cursor += 1
            except Exception as e:
                print(f"[ObserverReplay] failed keyframe {render_frame_idx}: {e}")

        if args.save_gif or args.save_mp4:
            frame_rgb = None
            try:
                record_camera.take_picture()
                frame_rgb = to_uint8_rgb(record_camera.get_picture("Color"))
            except Exception:
                frame_rgb = None

            if frame_rgb is not None:
                if args.save_gif and render_frame_idx % max(1, int(args.gif_stride)) == 0:
                    if max_gif_frames is None or len(gif_frames) < max_gif_frames:
                        gif_frames.append(Image.fromarray(frame_rgb))
                    elif not gif_cap_warned:
                        print(f"\n[ObserverReplay] GIF frame cap reached ({max_gif_frames}), later frames are skipped.")
                        gif_cap_warned = True

                if args.save_mp4:
                    if video_writer is None and video_ffmpeg is None:
                        h0, w0 = frame_rgb.shape[:2]
                        try:
                            video_ffmpeg = start_ffmpeg_mp4_writer(args.mp4_path, w0, h0, args.mp4_fps)
                            video_writer_mode = "ffmpeg"
                        except Exception:
                            video_ffmpeg = None
                            video_writer_mode = None

                    if video_ffmpeg is not None and video_ffmpeg.stdin is not None:
                        try:
                            video_ffmpeg.stdin.write(frame_rgb.tobytes())
                        except Exception:
                            video_ffmpeg = None

                    if video_ffmpeg is None:
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        h0, w0 = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(args.mp4_path, fourcc, float(max(1.0, args.mp4_fps)), (w0, h0))
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Failed to open VideoWriter for {args.mp4_path}")
                        video_writer_mode = "cv2"
                        video_writer.write(frame_bgr)

        render_frame_idx += 1

    task._update_render = wrapped_update_render
    try: task.play_once()
    except Exception as e: print(f"\n[ObserverReplay] end: {e}")

    if video_ffmpeg is not None:
        if video_ffmpeg.stdin is not None:
            video_ffmpeg.stdin.close()
        video_ffmpeg.wait()

    if video_writer is not None:
        video_writer.release()

    if video_writer is not None or video_ffmpeg is not None:
        mode = "H.264/ffmpeg" if video_writer_mode == "ffmpeg" else "mp4v/cv2"
        print(f"[ObserverReplay] MP4 saved to {args.mp4_path} ({mode})")

    if args.save_gif and len(gif_frames) > 0:
        gif_frames[0].save(
            args.gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(1, int(args.gif_duration)),
            loop=0,
        )
        print(f"[ObserverReplay] GIF saved to {args.gif_path}")

    print("\n[ObserverReplay] SAPIEN Simulation Done! Proceeding to save data arrays... Please wait.")



    if args.save_sceneflow and len(reference_snapshots) > 0:
        saved_meta = []
        for i, snap in enumerate(reference_snapshots):
            ref_frame = int(snap.get("ref_frame_idx", i))
            flow_name = f"scene_point_flow_ref{ref_frame:05d}"
            res = reconstruct_reference_flow(snap, frame_pose_history, args.sceneflow_dir, flow_name, save_dense=True)
            saved_meta.append({"reference_index": i, "ref_frame_idx": ref_frame, "keyframe_target": int(snap.get("keyframe_target", ref_frame)), **res})
            if i == 0:
                pc0 = world_points_to_camera_matrix(snap["init_world"], snap["ref_camera_matrix"]).astype(np.float32)
                np.save(os.path.join(args.sceneflow_dir, "pointcloud_frame0.npy"), pc0)
                np.save(os.path.join(args.sceneflow_dir, "segmentation_frame0.npy"), snap["seg_raw"].astype(np.int32))
                np.save(os.path.join(args.sceneflow_dir, "point_seg_ids_frame0.npy"), snap["owner_sid"].astype(np.int32))
        meta_path = os.path.join(args.sceneflow_dir, "sceneflow_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"trajectory_length": traj_len, "outputs": saved_meta}, f, indent=2)
        print(f"[ObserverReplay] Dense outputs successfully saved over {traj_len} frames in {args.sceneflow_dir}")

if __name__ == "__main__":
    main()
