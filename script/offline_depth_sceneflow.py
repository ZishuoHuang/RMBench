#!/usr/bin/env python3
"""
Offline sceneflow generation from saved depth/segmentation/pose data.

This script replaces simulator replay by:
1) back-projecting depth to 3D points at reference frames,
2) attaching points to body-local coordinates using segmentation ids,
3) propagating those local points across all frames with saved body poses,
4) converting to reference-camera coordinates and saving dense flow.

Output format matches existing compressor input:
- scene_point_flow_refXXXXX.npy      (T, H, W, 3), float16
- scene_point_flow_refXXXXX.anchor.npy (H, W, 3), float16
- segmentation_refXXXXX.npy          (H, W), int32
- sceneflow_meta.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml


def build_keyframe_indices(num_frames: int, num_keyframes: int = 5) -> List[int]:
    if num_frames <= 1:
        return [0] * max(1, int(num_keyframes))
    num_keyframes = max(1, int(num_keyframes))
    if num_keyframes == 1:
        return [int(num_frames // 2)]
    raw = [int(round(i * (num_frames - 1) / float(num_keyframes - 1))) for i in range(num_keyframes)]
    return [min(idx, num_frames - 1) for idx in raw]


def world_points_to_camera_matrix(points_world: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    t_w2c = np.linalg.inv(t_c2w.astype(np.float32))
    return points_world @ t_w2c[:3, :3].T + t_w2c[:3, 3]


def world_points_to_camera_with_w2c(points_world: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    """Convert world points to camera frame with precomputed T_w2c."""
    return points_world @ t_w2c[:3, :3].T + t_w2c[:3, 3]


def transform_local_points(local_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return local_points @ pose[:3, :3].T + pose[:3, 3]


def load_config_paths(task_name: str, task_config: str) -> Tuple[str, str]:
    cfg_path = os.path.join("task_config", f"{task_config}.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    save_root = cfg.get("save_path", "./data")
    target_root = os.path.join(save_root, task_name, task_config)
    return cfg_path, target_root


def ensure_sid_map_json(task_name: str, task_config: str, episode: int, target_root: str) -> Optional[str]:
    sid_map_json = os.path.join(target_root, "sid_map", f"episode{episode}.json")
    if os.path.exists(sid_map_json):
        return sid_map_json

    os.makedirs(os.path.dirname(sid_map_json), exist_ok=True)
    cmd = [
        sys.executable,
        "script/export_sid_map_for_episode.py",
        task_name,
        task_config,
        "--episode",
        str(episode),
        "--output",
        sid_map_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[OfflineDepth][WARN] sid map export failed")
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip())
        return None

    return sid_map_json if os.path.exists(sid_map_json) else None


def load_sidecar_array(h5_path: str, camera_name: str, key: str) -> Optional[np.ndarray]:
    sidecar_root = os.path.splitext(h5_path)[0] + "_npy"
    arr_path = os.path.join(sidecar_root, "observation", camera_name, f"{key}.npy")
    if os.path.exists(arr_path):
        return np.load(arr_path)
    return None


def _read_h5_string_scalar(h5: h5py.File, path: str) -> Optional[str]:
    if path not in h5:
        return None
    raw = h5[path][...]
    if raw.ndim == 0:
        item = raw.item()
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode("utf-8")
        return str(item)
    for item in raw:
        if isinstance(item, (bytes, np.bytes_)):
            text = item.decode("utf-8")
        else:
            text = str(item)
        if text and text != "nan":
            return text
    return None


def _array_sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def compute_source_signature(h5_path: str, camera_name: str) -> Dict[str, object]:
    h5_mtime_ns = int(os.stat(h5_path).st_mtime_ns)
    signature: Dict[str, object] = {
        "camera_name": str(camera_name),
        "h5_mtime_ns": h5_mtime_ns,
    }

    selected_camera_name = None
    with h5py.File(h5_path, "r") as h5:
        selected_camera_name = _read_h5_string_scalar(h5, "scene_state/selected_camera_name")
    if selected_camera_name:
        signature["selected_camera_name"] = selected_camera_name

    cam_pose = load_sidecar_array(h5_path, camera_name, "cam2world_gl")
    if cam_pose is None:
        cam_pose = load_sidecar_array(h5_path, camera_name, "cam_pose")
    if cam_pose is None:
        cam_pose = load_sidecar_array(h5_path, camera_name, "cam_poses")
    if cam_pose is not None:
        signature["cam_pose_shape"] = list(cam_pose.shape)
        signature["cam_pose_sha1"] = _array_sha1(cam_pose.astype(np.float32))

    intrinsic = load_sidecar_array(h5_path, camera_name, "intrinsic_cv")
    if intrinsic is None:
        intrinsic = load_sidecar_array(h5_path, camera_name, "cam_intrinsics")
    if intrinsic is not None:
        signature["intrinsic_shape"] = list(intrinsic.shape)
        signature["intrinsic_sha1"] = _array_sha1(intrinsic.astype(np.float32))

    depth = load_sidecar_array(h5_path, camera_name, "depth")
    if depth is None:
        depth = load_sidecar_array(h5_path, camera_name, "depth_video")
    if depth is not None:
        signature["num_frames"] = int(depth.shape[0])

    return signature


def should_skip_existing(out_dir: str, h5_path: str, camera_name: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False

    meta_path = os.path.join(out_dir, "sceneflow_meta.json")
    if not os.path.exists(meta_path):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    old_sig = meta.get("source_signature")
    if not isinstance(old_sig, dict):
        return False

    new_sig = compute_source_signature(h5_path, camera_name)
    if old_sig == new_sig:
        return True

    print("[OfflineDepth] source signature changed, regenerate sceneflow")
    return False


def load_depth(h5: h5py.File, h5_path: str, camera_name: str) -> np.ndarray:
    arr = load_sidecar_array(h5_path, camera_name, "depth")
    if arr is not None:
        return arr.astype(np.float32)
    path = f"observation/{camera_name}/depth"
    if path in h5:
        return h5[path][...].astype(np.float32)
    raise KeyError(f"depth not found in h5 or sidecar for camera={camera_name}")


def load_intrinsic(h5: h5py.File, h5_path: str, camera_name: str) -> np.ndarray:
    arr = load_sidecar_array(h5_path, camera_name, "intrinsic_cv")
    if arr is not None:
        return arr.astype(np.float32)
    path = f"observation/{camera_name}/intrinsic_cv"
    if path in h5:
        return h5[path][...].astype(np.float32)
    raise KeyError(f"intrinsic_cv not found in h5 or sidecar for camera={camera_name}")


def load_cam2world(h5: h5py.File, h5_path: str, camera_name: str) -> np.ndarray:
    arr = load_sidecar_array(h5_path, camera_name, "cam2world_gl")
    if arr is not None:
        return arr.astype(np.float32)
    path = f"observation/{camera_name}/cam2world_gl"
    if path in h5:
        return h5[path][...].astype(np.float32)
    raise KeyError(f"cam2world_gl not found in h5 or sidecar for camera={camera_name}")


def load_segmentation(h5: h5py.File, camera_name: str) -> np.ndarray:
    arr = load_sidecar_array(h5.filename, camera_name, "actor_segmentation_raw")
    if arr is not None:
        return arr.astype(np.int32)
    arr = load_sidecar_array(h5.filename, camera_name, "segmentation")
    if arr is not None:
        return arr.astype(np.int32)
    arr = load_sidecar_array(h5.filename, camera_name, "seg")
    if arr is not None:
        return arr.astype(np.int32)

    # segmentation sidecar may not always exist, so h5 is fallback
    path = f"observation/{camera_name}/actor_segmentation_raw"
    if path not in h5:
        raise KeyError(f"actor_segmentation_raw missing at {path}")
    return h5[path][...].astype(np.int32)


def load_sid_to_body_map(h5: h5py.File, sid_map_json: Optional[str]) -> Dict[int, str]:
    if sid_map_json is not None:
        with open(sid_map_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    path = "scene_state/sid_to_body_key_json"
    if path not in h5:
        raise KeyError(
            "Missing scene_state/sid_to_body_key_json in h5. "
            "Re-collect data with updated collector or provide --sid-map-json."
        )

    raw_arr = h5[path][...]
    text = None
    if raw_arr.ndim == 0:
        text = raw_arr.astype(str).item()
    else:
        for item in raw_arr:
            s = item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            if s and s != "nan":
                text = s
                break
    if not text:
        raise RuntimeError("sid_to_body_key_json dataset exists but is empty")

    parsed = json.loads(text)
    return {int(k): str(v) for k, v in parsed.items()}


def resolve_pose_dataset(h5: h5py.File, body_key: str) -> Optional[np.ndarray]:
    if body_key.startswith("rigid::"):
        name = body_key.split("::", 1)[1]
        path = f"scene_state/rigid_actor_poses/{name}"
        if path in h5:
            return h5[path][...].astype(np.float32)
        return None

    if body_key.startswith("art::"):
        parts = body_key.split("::")
        if len(parts) != 3:
            return None
        art_name, link_name = parts[1], parts[2]
        path = f"scene_state/articulation_link_poses/{art_name}/{link_name}"
        if path in h5:
            return h5[path][...].astype(np.float32)
        return None

    return None


def backproject_depth_to_camera(depth_mm: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    d = (depth_mm[valid] / 1000.0).astype(np.float32)  # mm -> m
    u_valid = u[valid]
    v_valid = v[valid]

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    x = (u_valid - cx) / fx * d
    y = -(v_valid - cy) / fy * d
    z = -d

    points_cam = np.stack([x, y, z], axis=-1).astype(np.float32)
    return points_cam, u_valid.astype(np.int32), v_valid.astype(np.int32)


def generate_episode_offline(
    h5_path: str,
    output_dir: str,
    camera_name: str,
    num_keyframes: int,
    include_background: bool,
    sid_map_json: Optional[str],
    source_signature: Optional[Dict[str, object]] = None,
):
    with h5py.File(h5_path, "r") as h5:
        depth = load_depth(h5, h5_path, camera_name)
        seg = load_segmentation(h5, camera_name)
        intrinsic = load_intrinsic(h5, h5_path, camera_name)
        cam2world = load_cam2world(h5, h5_path, camera_name)
        sid_to_body_key = load_sid_to_body_map(h5, sid_map_json)

        t = min(depth.shape[0], seg.shape[0], cam2world.shape[0], intrinsic.shape[0])
        if t <= 0:
            raise RuntimeError("No valid frames found")

        keyframe_targets = build_keyframe_indices(t, num_keyframes)
        os.makedirs(output_dir, exist_ok=True)

        outputs = []
        pc0_saved = False

        for ref_idx in keyframe_targets:
            depth_ref = depth[ref_idx]
            seg_ref = seg[ref_idx]
            k_ref = intrinsic[ref_idx]
            t_c2w_ref = cam2world[ref_idx]

            points_cam_ref, uu, vv = backproject_depth_to_camera(depth_ref, k_ref)
            if len(uu) == 0:
                continue

            sid = seg_ref[vv, uu].astype(np.int32)
            if not include_background:
                keep = sid > 0
                points_cam_ref = points_cam_ref[keep]
                uu = uu[keep]
                vv = vv[keep]
                sid = sid[keep]
                if len(uu) == 0:
                    continue

            points_world_ref = points_cam_ref @ t_c2w_ref[:3, :3].T + t_c2w_ref[:3, 3]

            unique_sid = np.unique(sid)
            sid_pose_seq = {}
            for s in unique_sid.tolist():
                key = sid_to_body_key.get(int(s))
                if key is None:
                    continue
                seq = resolve_pose_dataset(h5, key)
                if seq is None:
                    continue
                if seq.shape[0] < t:
                    continue
                sid_pose_seq[int(s)] = seq[:t]

            keep_uu_blocks: List[np.ndarray] = []
            keep_vv_blocks: List[np.ndarray] = []
            keep_sid_blocks: List[np.ndarray] = []
            local_points_blocks: List[np.ndarray] = []
            init_world_blocks: List[np.ndarray] = []

            # Batch by sid: avoid per-point Python loop.
            for s in unique_sid.tolist():
                idx_arr = np.where(sid == int(s))[0]
                if idx_arr.size == 0:
                    continue

                pts_world_sid = points_world_ref[idx_arr]
                seq = sid_pose_seq.get(int(s))
                if seq is None:
                    if not include_background:
                        continue
                    keep_sid_blocks.append(np.full((idx_arr.size,), -1, dtype=np.int32))
                    local_points_blocks.append(pts_world_sid.astype(np.float32))
                else:
                    pose_ref = seq[ref_idx]
                    r_ref = pose_ref[:3, :3].astype(np.float32)
                    t_ref = pose_ref[:3, 3].astype(np.float32)
                    p_local = (pts_world_sid - t_ref[None, :]) @ r_ref
                    keep_sid_blocks.append(np.full((idx_arr.size,), int(s), dtype=np.int32))
                    local_points_blocks.append(p_local.astype(np.float32))

                init_world_blocks.append(pts_world_sid.astype(np.float32))
                keep_uu_blocks.append(uu[idx_arr].astype(np.int32))
                keep_vv_blocks.append(vv[idx_arr].astype(np.int32))

            if len(local_points_blocks) == 0:
                continue

            keep_uu = np.concatenate(keep_uu_blocks, axis=0)
            keep_vv = np.concatenate(keep_vv_blocks, axis=0)
            keep_sid = np.concatenate(keep_sid_blocks, axis=0)
            local_points = np.concatenate(local_points_blocks, axis=0).astype(np.float32)
            init_world = np.concatenate(init_world_blocks, axis=0).astype(np.float32)

            sid_to_indices = {
                int(s): np.where(keep_sid == int(s))[0].astype(np.int64)
                for s in np.unique(keep_sid).tolist()
            }

            h, w = depth_ref.shape
            t_w2c_ref = np.linalg.inv(t_c2w_ref.astype(np.float32))
            r_w2c_t = t_w2c_ref[:3, :3].T
            t_w2c = t_w2c_ref[:3, 3]

            anchor_cam = world_points_to_camera_with_w2c(init_world, t_w2c_ref)
            anchor_dense = np.zeros((h, w, 3), dtype=np.float32)
            anchor_dense[keep_vv, keep_uu] = anchor_cam

            suffix = f"ref{ref_idx:05d}"
            flow_name = f"scene_point_flow_{suffix}"
            flow_path = os.path.join(output_dir, f"{flow_name}.npy")
            anchor_path = os.path.join(output_dir, f"{flow_name}.anchor.npy")
            seg_path = os.path.join(output_dir, f"segmentation_{suffix}.npy")

            flow_mm = np.lib.format.open_memmap(flow_path, mode="w+", dtype=np.float16, shape=(t, h, w, 3))
            anchor_mm = np.lib.format.open_memmap(anchor_path, mode="w+", dtype=np.float16, shape=(h, w, 3))
            seg_mm = np.lib.format.open_memmap(seg_path, mode="w+", dtype=np.int32, shape=(h, w))

            anchor_mm[...] = anchor_dense.astype(np.float16)
            seg_mm[...] = 0
            seg_mm[keep_vv, keep_uu] = seg_ref[keep_vv, keep_uu].astype(np.int32)

            flow_array = np.zeros((t, h, w, 3), dtype=np.float16)

            for s, idx_arr in sid_to_indices.items():
                if s == -1:
                    flow_array[:, keep_vv[idx_arr], keep_uu[idx_arr]] = anchor_cam[idx_arr].astype(np.float16)
                else:
                    seq = sid_pose_seq.get(int(s))
                    if seq is None:
                        flow_array[:, keep_vv[idx_arr], keep_uu[idx_arr]] = anchor_cam[idx_arr].astype(np.float16)
                    else:
                        seq_t = seq[:t]
                        r_obj = seq_t[:, :3, :3].astype(np.float32)
                        t_obj = seq_t[:, :3, 3].astype(np.float32)
                        
                        a = np.matmul(r_obj.transpose(0, 2, 1), r_w2c_t)
                        b = np.dot(t_obj, r_w2c_t) + t_w2c
                        
                        pts_cam = np.matmul(local_points[idx_arr][None, :, :], a) + b[:, None, :]
                        flow_array[:, keep_vv[idx_arr], keep_uu[idx_arr]] = pts_cam.astype(np.float16)

            flow_mm[...] = flow_array
            flow_mm.flush()
            anchor_mm.flush()
            seg_mm.flush()

            outputs.append(
                {
                    "reference_index": len(outputs),
                    "ref_frame_idx": int(ref_idx),
                    "keyframe_target": int(ref_idx),
                    "flow_path": flow_path,
                    "anchor_path": anchor_path,
                    "seg_path": seg_path,
                    "shape": [int(t), int(h), int(w), 3],
                }
            )

            if not pc0_saved:
                np.save(os.path.join(output_dir, "pointcloud_frame0.npy"), anchor_cam.astype(np.float32))
                np.save(os.path.join(output_dir, "segmentation_frame0.npy"), seg_ref.astype(np.int32))
                np.save(os.path.join(output_dir, "point_seg_ids_frame0.npy"), keep_sid.astype(np.int32))
                pc0_saved = True

        meta = {
            "camera_name": str(camera_name),
            "trajectory_length": int(t),
            "keyframe_targets": [int(x) for x in keyframe_targets],
            "outputs": outputs,
        }
        if source_signature is not None:
            meta["source_signature"] = source_signature
        with open(os.path.join(output_dir, "sceneflow_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[OfflineDepth] done, outputs={len(outputs)} dir={output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Offline depth-based sceneflow generation (no simulator replay)")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera", type=str, default="world_camera1")
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--sid-map-json", type=str, default=None, help="Optional explicit sid->body key mapping json")
    parser.add_argument("--sceneflow-root", type=str, default=None)
    args = parser.parse_args()

    _, target_root = load_config_paths(args.task_name, args.task_config)
    h5_path = os.path.join(target_root, "data", f"episode{args.episode}.hdf5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"h5 not found: {h5_path}")

    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(target_root, f"sceneflow_offline_depth_{args.camera}")
    out_dir = os.path.join(args.sceneflow_root, f"episode{args.episode}")

    if should_skip_existing(out_dir, h5_path, args.camera, args.skip_existing):
        print(f"[OfflineDepth] skip existing: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    if args.sid_map_json is None:
        with h5py.File(h5_path, "r") as h5:
            has_sid_map_in_h5 = ("scene_state" in h5 and "sid_to_body_key_json" in h5["scene_state"])
        if not has_sid_map_in_h5:
            auto_sid_map = ensure_sid_map_json(args.task_name, args.task_config, args.episode, target_root)
            if auto_sid_map is not None:
                args.sid_map_json = auto_sid_map

    generate_episode_offline(
        h5_path=h5_path,
        output_dir=out_dir,
        camera_name=args.camera,
        num_keyframes=args.keyframes,
        include_background=args.include_background,
        sid_map_json=args.sid_map_json,
        source_signature=compute_source_signature(h5_path, args.camera),
    )


if __name__ == "__main__":
    main()
