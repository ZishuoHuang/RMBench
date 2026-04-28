import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys

import cv2
import h5py
import numpy as np
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
            entity_map[int(sid)] = actor

    for link in task_env.robot.left_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[int(sid)] = link

    for link in task_env.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[int(sid)] = link

    return entity_map


def resolve_source_camera(task, camera_name):
    camera_container = getattr(task, "cameras", getattr(task, "camera", task))
    world1 = getattr(camera_container, "world_camera1", getattr(task, "world_camera1", None))
    world2 = getattr(camera_container, "world_camera2", getattr(task, "world_camera2", None))
    obs = getattr(camera_container, "observer_camera", getattr(task, "observer_camera", None))

    if camera_name == "world_camera1" and world1 is not None:
        return world1
    if camera_name == "world_camera2" and world2 is not None:
        return world2
    if camera_name == "observer_camera" and obs is not None:
        return obs

    if hasattr(camera_container, "static_camera_name") and hasattr(camera_container, "static_camera_list"):
        for name, cam in zip(camera_container.static_camera_name, camera_container.static_camera_list):
            if camera_name == name:
                return cam

    raise KeyError(f"Unknown camera '{camera_name}'")


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
    if num_frames <= 1:
        return [0] * num_keyframes
    num_keyframes = max(1, int(num_keyframes))
    if num_keyframes == 1:
        return [int(num_frames // 2)]
    raw = [int(round(i * (num_frames - 1) / float(num_keyframes - 1))) for i in range(num_keyframes)]
    keyframes = []
    for idx in raw:
        keyframes.append(min(idx, num_frames - 1))
    return keyframes


def sample_reference_points_all(camera, entity_map):
    camera.take_picture()
    position = camera.get_picture("Position")
    segmentation = camera.get_picture("Segmentation")

    seg_raw = segmentation[..., 1].astype(np.int32)
    valid = position[..., 3] < 1

    vv, uu = np.where(valid)
    if len(uu) == 0:
        return None

    points_cam = position[vv, uu, :3].astype(np.float32)
    T_c2w = camera.get_model_matrix().astype(np.float32)
    points_world = points_cam @ T_c2w[:3, :3].T + T_c2w[:3, 3]

    keep_uu, keep_vv, keep_sid, local_points, init_world = [], [], [], [], []
    ref_pose_by_sid = {}
    ref_pose_by_sid[-1] = np.eye(4, dtype=np.float32)
    total_valid_pixels = len(uu)

    for p_world, sid, u, v in zip(points_world, seg_raw[vv, uu].astype(np.int32), uu, vv):
        entity = entity_map.get(int(sid), None)
        if entity is None:
            # Keep unmatched pixels as world-anchored points so we still preserve all valid pixels.
            sid_keep = -1
            p_local = p_world.astype(np.float32)
        else:
            sid_keep = int(sid)
            pose = get_entity_pose_matrix(entity)
            pose_inv = np.linalg.inv(pose)
            p_local = (pose_inv @ np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32))[:3]
            ref_pose_by_sid[sid_keep] = pose

        keep_uu.append(int(u))
        keep_vv.append(int(v))
        keep_sid.append(sid_keep)
        local_points.append(p_local)
        init_world.append(p_world)

    if len(local_points) == 0:
        raise RuntimeError("Failed to attach all-pixel points to entities")

    return {
        "shape": position.shape[:2],
        "seg_raw": seg_raw,
        "uu": np.asarray(keep_uu, dtype=np.int32),
        "vv": np.asarray(keep_vv, dtype=np.int32),
        "owner_sid": np.asarray(keep_sid, dtype=np.int32),
        "local_points": np.asarray(local_points, dtype=np.float32),
        "init_world": np.asarray(init_world, dtype=np.float32),
        "ref_camera_matrix": T_c2w,
        "ref_pose_by_sid": ref_pose_by_sid,
        "total_valid_pixels": int(total_valid_pixels),
        "kept_pixels": int(len(local_points)),
    }


def reconstruct_reference_flow(snapshot, frame_pose_history, output_dir, flow_name):
    local_points = snapshot["local_points"]
    owner_sid = snapshot["owner_sid"]
    uu = snapshot["uu"]
    vv = snapshot["vv"]
    h, w = snapshot["shape"]
    ref_c2w = snapshot["ref_camera_matrix"]
    ref_pose_by_sid = snapshot["ref_pose_by_sid"]
    T = len(frame_pose_history)

    sid_to_indices = {}
    for idx, sid in enumerate(owner_sid):
        sid_to_indices.setdefault(int(sid), []).append(idx)

    anchor_world = snapshot["init_world"]
    anchor_cam = world_points_to_camera_matrix(anchor_world, ref_c2w)

    anchor_dense = np.zeros((h, w, 3), dtype=np.float32)
    anchor_dense[vv, uu] = anchor_cam

    suffix = flow_name.split("scene_point_flow_")[-1]
    flow_path = os.path.join(output_dir, f"{flow_name}.npy")
    delta_path = os.path.join(output_dir, f"scene_point_delta_{suffix}.npy")
    anchor_path = os.path.join(output_dir, f"{flow_name}.anchor.npy")
    seg_path = os.path.join(output_dir, f"segmentation_{suffix}.npy")

    flow_mm = np.lib.format.open_memmap(flow_path, mode="w+", dtype=np.float16, shape=(T, h, w, 3))
    delta_mm = np.lib.format.open_memmap(delta_path, mode="w+", dtype=np.float16, shape=(T, h, w, 3))
    anchor_mm = np.lib.format.open_memmap(anchor_path, mode="w+", dtype=np.float16, shape=(h, w, 3))
    seg_mm = np.lib.format.open_memmap(seg_path, mode="w+", dtype=np.int32, shape=(h, w))

    anchor_mm[...] = anchor_dense.astype(np.float16)
    seg_mm[...] = 0
    seg_mm[vv, uu] = snapshot["seg_raw"][vv, uu].astype(np.int32)

    for t, pose_map in enumerate(frame_pose_history):
        for sid, idx_list in sid_to_indices.items():
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            pose = pose_map.get(int(sid), None)
            if pose is None:
                pose = ref_pose_by_sid.get(int(sid), np.eye(4, dtype=np.float32))
            pts_world = transform_local_points(local_points[idx_arr], pose)
            pts_cam = world_points_to_camera_matrix(pts_world, ref_c2w)
            flow_mm[t, vv[idx_arr], uu[idx_arr]] = pts_cam.astype(np.float16)
            delta_mm[t, vv[idx_arr], uu[idx_arr]] = (pts_cam - anchor_cam[idx_arr]).astype(np.float16)

    flow_mm.flush()
    delta_mm.flush()
    anchor_mm.flush()
    seg_mm.flush()

    return {
        "flow_path": flow_path,
        "delta_path": delta_path,
        "anchor_path": anchor_path,
        "seg_path": seg_path,
        "shape": [T, h, w, 3],
    }


def to_uint8_rgb(image):
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
        if arr_f.size > 0 and float(np.nanmax(arr_f)) <= 1.0:
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


def camera_points_to_pixels(points_cam, K, width, height):
    z = points_cam[:, 2]
    valid = z < -1e-6
    if not np.any(valid):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), valid

    pts = points_cam[valid]
    u = K[0, 0] * pts[:, 0] / (-pts[:, 2]) + K[0, 2]
    v = -K[1, 1] * pts[:, 1] / (-pts[:, 2]) + K[1, 2]

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return u[in_frame], v[in_frame], valid


def sid_color_lut(owner_sid):
    unique_sid = np.unique(owner_sid)
    lut = {}
    for sid in unique_sid:
        s = int(sid)
        r = (s * 37 + 53) % 256
        g = (s * 97 + 29) % 256
        b = (s * 17 + 131) % 256
        lut[s] = np.array([r, g, b], dtype=np.uint8)
    return lut


def render_attached_points_frame(rgb_frame, snapshot, pose_map, camera_model_matrix, K):
    h, w = rgb_frame.shape[:2]
    out = rgb_frame.copy()

    owner_sid = snapshot["owner_sid"]
    local_points = snapshot["local_points"]
    ref_pose_by_sid = snapshot["ref_pose_by_sid"]
    colors = snapshot["sid_colors"]

    sid_to_indices = snapshot["sid_to_indices"]
    points_world = np.zeros((local_points.shape[0], 3), dtype=np.float32)

    for sid, idx_arr in sid_to_indices.items():
        pose = pose_map.get(int(sid), None)
        if pose is None:
            pose = ref_pose_by_sid.get(int(sid), np.eye(4, dtype=np.float32))
        points_world[idx_arr] = transform_local_points(local_points[idx_arr], pose)

    pts_cam = world_points_to_camera_matrix(points_world, camera_model_matrix)
    uu, vv, valid_z = camera_points_to_pixels(pts_cam, K, w, h)
    if len(uu) == 0:
        return out

    valid_indices = np.where(valid_z)[0]
    u_all = np.round(K[0, 0] * pts_cam[valid_indices, 0] / (-pts_cam[valid_indices, 2]) + K[0, 2]).astype(np.int32)
    v_all = np.round(-K[1, 1] * pts_cam[valid_indices, 1] / (-pts_cam[valid_indices, 2]) + K[1, 2]).astype(np.int32)
    in_frame = (u_all >= 0) & (u_all < w) & (v_all >= 0) & (v_all < h)
    idx_final = valid_indices[in_frame]

    out[v_all[in_frame], u_all[in_frame]] = colors[idx_final]
    return out


def get_episode_length_from_h5(h5_path, camera_name):
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        if "observation" not in f:
            return None
        obs = f["observation"]
        if camera_name in obs and "rgb" in obs[camera_name]:
            return int(obs[camera_name]["rgb"].shape[0])
        for cam_name in obs.keys():
            if "rgb" in obs[cam_name]:
                return int(obs[cam_name]["rgb"].shape[0])
    return None


def is_done(sceneflow_dir):
    meta_path = os.path.join(sceneflow_dir, "sceneflow_meta.json")
    return os.path.exists(meta_path)


def load_traj_pickle(save_path, episode):
    p = os.path.join(save_path, "_traj_data", f"episode{episode}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def count_replay_frames(task_name, cfg, seed, episode, traj_data):
    dry_task = class_decorator(task_name)
    cfg_dry = dict(cfg)
    cfg_dry["need_plan"] = False
    cfg_dry["save_data"] = False
    cfg_dry["collect_data"] = False

    dry_task.setup_demo(now_ep_num=episode, seed=seed, **cfg_dry)
    cfg_dry["left_joint_path"] = traj_data["left_joint_path"]
    cfg_dry["right_joint_path"] = traj_data["right_joint_path"]
    dry_task.set_path_lst(cfg_dry)

    total_frames = 0

    def dry_update():
        nonlocal total_frames
        total_frames += 1

    dry_task._update_render = dry_update
    try:
        dry_task.play_once()
    except Exception:
        pass
    if hasattr(dry_task, "close"):
        dry_task.close()

    return max(1, total_frames)


def main():
    parser = argparse.ArgumentParser(description="Generate dense sceneflow from all valid pixels (832x480) for one episode")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera", type=str, default="world_camera1")
    parser.add_argument("--sceneflow-root", type=str, default=None)
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--exact-length", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-preview-mp4", action="store_true")
    parser.add_argument("--preview-mp4-path", type=str, default=None)
    parser.add_argument("--preview-fps", type=float, default=12.0)
    args = parser.parse_args()

    cfg = load_args(args.task_name, args.task_config)
    ep = int(args.episode)

    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(cfg["save_path"], f"sceneflow_all_points_{args.camera}")
    sceneflow_dir = os.path.join(args.sceneflow_root, f"episode{ep}")
    os.makedirs(sceneflow_dir, exist_ok=True)

    if args.skip_existing and is_done(sceneflow_dir):
        print(f"[SceneFlowAll] skip existing episode {ep}: {sceneflow_dir}")
        return

    h5_path = os.path.join(cfg["save_path"], "data", f"episode{ep}.hdf5")

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]
    if ep < 0 or ep >= len(seeds):
        raise RuntimeError(f"episode out of range: {ep}, available 0..{len(seeds)-1}")

    traj_data = load_traj_pickle(cfg["save_path"], ep)

    print("[SceneFlowAll] dry-run to count exact replay length...")
    traj_len = count_replay_frames(args.task_name, cfg, seeds[ep], ep, traj_data)
    print(f"[SceneFlowAll] exact replay length = {traj_len}")

    if args.exact_length > 0 and int(args.exact_length) != traj_len:
        print(
            f"[SceneFlowAll] warn: --exact-length={int(args.exact_length)} "
            f"differs from measured replay length={traj_len}; using measured length"
        )

    h5_len = get_episode_length_from_h5(h5_path, args.camera)
    if h5_len is not None and h5_len != traj_len:
        print(f"[SceneFlowAll] note: h5 length({h5_len}) != replay length({traj_len})")

    keyframe_targets = build_keyframe_indices(traj_len, args.keyframes)

    task = class_decorator(args.task_name)
    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False

    task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg)
    cfg["left_joint_path"] = traj_data["left_joint_path"]
    cfg["right_joint_path"] = traj_data["right_joint_path"]
    task.set_path_lst(cfg)

    task.get_obs()
    source_camera = resolve_source_camera(task, args.camera)

    frame_pose_history = []
    reference_snapshots = []
    render_frame_idx = 0
    target_cursor = 0
    preview_snapshot = None
    preview_ffmpeg = None

    original_update_render = task._update_render

    def wrapped_update_render():
        nonlocal render_frame_idx, target_cursor, preview_snapshot, preview_ffmpeg
        current_map = build_scene_entity_map(task)
        frame_pose_history.append({int(s): get_entity_pose_matrix(entity) for s, entity in current_map.items()})

        original_update_render()

        if target_cursor < len(keyframe_targets) and render_frame_idx == keyframe_targets[target_cursor]:
            snap = sample_reference_points_all(source_camera, current_map)
            if snap is not None:
                snap["ref_frame_idx"] = render_frame_idx
                snap["keyframe_target"] = keyframe_targets[target_cursor]
                sid_to_indices = {}
                for idx, sid in enumerate(snap["owner_sid"]):
                    sid_to_indices.setdefault(int(sid), []).append(idx)
                for sid in sid_to_indices:
                    sid_to_indices[sid] = np.asarray(sid_to_indices[sid], dtype=np.int64)
                snap["sid_to_indices"] = sid_to_indices

                lut = sid_color_lut(snap["owner_sid"])
                snap["sid_colors"] = np.asarray([lut[int(sid)] for sid in snap["owner_sid"]], dtype=np.uint8)

                reference_snapshots.append(snap)
                print(
                    f"[SceneFlowAll] captured keyframe {target_cursor + 1}/{len(keyframe_targets)} at frame {render_frame_idx}"
                )
                if preview_snapshot is None and render_frame_idx == 0:
                    preview_snapshot = snap
                target_cursor += 1

        if args.save_preview_mp4:
            source_camera.take_picture()
            rgb_frame = to_uint8_rgb(source_camera.get_picture("Color"))
            if rgb_frame is not None:
                if preview_snapshot is not None:
                    cam_model = source_camera.get_model_matrix().astype(np.float32)
                    K = source_camera.get_intrinsic_matrix().astype(np.float32)
                    rgb_frame = render_attached_points_frame(
                        rgb_frame,
                        preview_snapshot,
                        frame_pose_history[-1],
                        cam_model,
                        K,
                    )

                if preview_ffmpeg is None:
                    preview_path = args.preview_mp4_path
                    if preview_path is None:
                        preview_path = os.path.join(sceneflow_dir, f"preview_ep{ep}_{args.camera}.mp4")
                    preview_ffmpeg = start_ffmpeg_mp4_writer(preview_path, rgb_frame.shape[1], rgb_frame.shape[0], args.preview_fps)

                if preview_ffmpeg is not None and preview_ffmpeg.stdin is not None:
                    try:
                        preview_ffmpeg.stdin.write(rgb_frame.tobytes())
                    except Exception:
                        preview_ffmpeg = None

        render_frame_idx += 1

    task._update_render = wrapped_update_render

    try:
        task.play_once()
    except Exception as e:
        print(f"[SceneFlowAll] replay ended with exception: {e}")

    if preview_ffmpeg is not None:
        if preview_ffmpeg.stdin is not None:
            preview_ffmpeg.stdin.close()
        preview_ffmpeg.wait()

    saved_meta = []
    for i, snap in enumerate(reference_snapshots):
        ref_frame = int(snap.get("ref_frame_idx", i))
        flow_name = f"scene_point_flow_ref{ref_frame:05d}"
        res = reconstruct_reference_flow(snap, frame_pose_history, sceneflow_dir, flow_name)
        saved_meta.append(
            {
                "reference_index": i,
                "ref_frame_idx": ref_frame,
                "keyframe_target": int(snap.get("keyframe_target", ref_frame)),
                "total_valid_pixels": int(snap.get("total_valid_pixels", 0)),
                "kept_pixels": int(snap.get("kept_pixels", 0)),
                **res,
            }
        )
        if i == 0:
            pc0 = world_points_to_camera_matrix(snap["init_world"], snap["ref_camera_matrix"]).astype(np.float32)
            np.save(os.path.join(sceneflow_dir, "pointcloud_frame0.npy"), pc0)
            np.save(os.path.join(sceneflow_dir, "segmentation_frame0.npy"), snap["seg_raw"].astype(np.int32))
            np.save(os.path.join(sceneflow_dir, "point_seg_ids_frame0.npy"), snap["owner_sid"].astype(np.int32))

    meta_path = os.path.join(sceneflow_dir, "sceneflow_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": args.task_name,
                "task_config": args.task_config,
                "episode": ep,
                "camera": args.camera,
                "trajectory_length": len(frame_pose_history),
                "keyframe_targets": keyframe_targets,
                "outputs": saved_meta,
            },
            f,
            indent=2,
        )

    if hasattr(task, "close"):
        task.close()

    print(f"[SceneFlowAll] done episode={ep}, outputs in {sceneflow_dir}")


if __name__ == "__main__":
    main()
