#!/usr/bin/env python3
"""
Visualize one point trajectory from offline sceneflow outputs.

Expected episode directory contents (offline_depth_sceneflow.py):
- scene_point_flow_refXXXXX.npy OR
- scene_point_delta_refXXXXX.npy + scene_point_flow_refXXXXX.anchor.npy
- segmentation_refXXXXX.npy (optional but recommended)

Example:
  python script/visualize_offline_track.py \
    --episode-dir data/turn_switch/xxx/sceneflow_offline_depth_world_camera1/episode0
"""

import argparse
from pathlib import Path
import numpy as np


def _load_flow_or_delta(episode_dir: Path):
    flow_files = sorted(episode_dir.glob("scene_point_flow_ref*.npy"))
    flow_files = [p for p in flow_files if ".anchor." not in p.name]
    if flow_files:
        fp = flow_files[0]
        flow = np.load(fp, mmap_mode="r")
        return fp, flow

    delta_files = sorted(episode_dir.glob("scene_point_delta_ref*.npy"))
    if not delta_files:
        raise FileNotFoundError("No scene_point_flow_ref*.npy or scene_point_delta_ref*.npy found")

    dp = delta_files[0]
    suffix = dp.stem.replace("scene_point_delta_", "")
    anchor = episode_dir / f"scene_point_flow_{suffix}.anchor.npy"
    if not anchor.exists():
        raise FileNotFoundError(f"Missing anchor file for {dp.name}: {anchor.name}")

    delta = np.load(dp, mmap_mode="r")
    anchor_arr = np.load(anchor, mmap_mode="r")
    flow = anchor_arr[None] + delta
    return dp, flow


def _choose_random_valid_pixel(flow: np.ndarray, seg: np.ndarray | None, seed: int):
    rng = np.random.default_rng(seed)
    t0 = np.asarray(flow[0])
    finite = np.isfinite(t0).all(axis=-1)
    nonzero = np.linalg.norm(t0, axis=-1) > 0
    valid = finite & nonzero

    if seg is not None:
        valid &= seg > 0

    ys, xs = np.where(valid)
    if len(xs) == 0:
        raise RuntimeError("No valid pixel to track")

    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])


def main():
    parser = argparse.ArgumentParser(description="Visualize one random point track from offline sceneflow")
    parser.add_argument("--episode-dir", type=str, required=True)
    parser.add_argument("--u", type=int, default=None, help="Pixel x (optional)")
    parser.add_argument("--v", type=int, default=None, help="Pixel y (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Output png path")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode dir not found: {episode_dir}")

    ref_file, flow = _load_flow_or_delta(episode_dir)
    t, h, w, _ = flow.shape

    seg_files = sorted(episode_dir.glob("segmentation_ref*.npy"))
    seg = np.load(seg_files[0], mmap_mode="r") if seg_files else None

    if args.u is None or args.v is None:
        u, v = _choose_random_valid_pixel(flow, seg, args.seed)
    else:
        u, v = int(args.u), int(args.v)

    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"Pixel out of range: (u={u}, v={v}), image size=({w}, {h})")

    track = np.asarray(flow[:, v, u, :], dtype=np.float32)
    frame_idx = np.arange(t)

    if np.linalg.norm(track[0]) == 0:
        print("[WARN] selected pixel starts as zero point; try another seed/pixel")

    npy_out = episode_dir / f"track_u{u}_v{v}.npy"
    np.save(npy_out, track)

    if args.out is None:
        png_out = episode_dir / f"track_u{u}_v{v}.png"
    else:
        png_out = Path(args.out)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[INFO] matplotlib not available; saved track npy only")
        print(f"[INFO] track: {npy_out}")
        print(f"[INFO] source: {ref_file.name}, shape={flow.shape}, pixel=({u},{v})")
        print(f"[INFO] import error: {e}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(frame_idx, track[:, 0], label="x")
    axes[0, 0].plot(frame_idx, track[:, 1], label="y")
    axes[0, 0].plot(frame_idx, track[:, 2], label="z")
    axes[0, 0].set_title("XYZ over frames")
    axes[0, 0].set_xlabel("frame")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(track[:, 0], track[:, 1], marker="o", markersize=2)
    axes[0, 1].set_title("XY trajectory")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(track[:, 0], track[:, 2], marker="o", markersize=2)
    axes[1, 0].set_title("XZ trajectory")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("z")
    axes[1, 0].grid(True, alpha=0.3)

    speed = np.linalg.norm(track[1:] - track[:-1], axis=-1) if t > 1 else np.array([0.0], dtype=np.float32)
    axes[1, 1].plot(np.arange(len(speed)), speed)
    axes[1, 1].set_title("Per-frame motion magnitude")
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Point track from {ref_file.name} at pixel (u={u}, v={v})")
    fig.tight_layout()
    fig.savefig(png_out, dpi=160)
    plt.close(fig)

    obj_id = int(seg[v, u]) if seg is not None else -1
    print(f"[OK] source={ref_file.name}")
    print(f"[OK] flow_shape={flow.shape}")
    print(f"[OK] pixel=(u={u}, v={v}), seg_id={obj_id}")
    print(f"[OK] track_npy={npy_out}")
    print(f"[OK] figure={png_out}")


if __name__ == "__main__":
    main()
