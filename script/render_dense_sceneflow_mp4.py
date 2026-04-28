import argparse
import glob
import os
import shutil
import subprocess

import cv2
import numpy as np


def pick_delta_file(sceneflow_dir, ref_frame=None):
    if ref_frame is not None:
        name = f"scene_point_delta_ref{int(ref_frame):05d}.npy"
        path = os.path.join(sceneflow_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing delta file: {path}")
        return path

    cands = sorted(glob.glob(os.path.join(sceneflow_dir, "scene_point_delta_ref*.npy")))
    if len(cands) == 0:
        raise FileNotFoundError(f"No delta files found in {sceneflow_dir}")
    return cands[0]


def load_mask(sceneflow_dir, delta_path):
    base = os.path.basename(delta_path)
    ref = base.replace("scene_point_delta_", "").replace(".npy", "")
    seg_path = os.path.join(sceneflow_dir, f"segmentation_{ref}.npy")
    if not os.path.exists(seg_path):
        return None
    seg = np.load(seg_path)
    return (seg != 0)


def flow_to_rgb(dx, dy, mag, mag_scale):
    angle = np.arctan2(dy, dx)
    hue = ((angle + np.pi) / (2.0 * np.pi) * 179.0).astype(np.uint8)

    if mag_scale <= 0:
        mag_norm = np.zeros_like(mag, dtype=np.float32)
    else:
        mag_norm = np.clip(mag / mag_scale, 0.0, 1.0)

    sat = (mag_norm * 255.0).astype(np.uint8)
    val = np.full_like(sat, 255, dtype=np.uint8)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Render dense sceneflow delta to MP4")
    parser.add_argument("sceneflow_dir", type=str)
    parser.add_argument("--ref-frame", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--mag-scale", type=float, default=-1.0, help="<=0 means auto from p99")
    parser.add_argument("--draw-mask", action="store_true", help="Black out invalid pixels using segmentation_ref*.npy")
    parser.add_argument("--codec", type=str, default="h264", choices=["h264", "mp4v"], help="Output codec")
    args = parser.parse_args()

    sceneflow_dir = args.sceneflow_dir
    delta_path = pick_delta_file(sceneflow_dir, args.ref_frame)
    delta = np.load(delta_path).astype(np.float32)  # [T, H, W, 3]

    if delta.ndim != 4 or delta.shape[-1] != 3:
        raise RuntimeError(f"Unexpected delta shape: {delta.shape}")

    t, h, w, _ = delta.shape

    if args.output is None:
        stem = os.path.basename(delta_path).replace(".npy", "")
        args.output = os.path.join(sceneflow_dir, f"{stem}.mp4")

    mask = None
    if args.draw_mask:
        mask = load_mask(sceneflow_dir, delta_path)
        if mask is not None and mask.shape != (h, w):
            mask = None

    mag = np.linalg.norm(delta[..., :2], axis=-1)
    if args.mag_scale > 0:
        mag_scale = float(args.mag_scale)
    else:
        mag_scale = float(np.percentile(mag, 99.0))
        if mag_scale <= 1e-8:
            mag_scale = 1.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temp_output = args.output
    need_transcode = args.codec == "h264"
    if need_transcode:
        temp_output = args.output.replace(".mp4", ".mp4v_tmp.mp4")

    writer = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1.0, args.fps)),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {temp_output}")

    for i in range(t):
        dx = delta[i, :, :, 0]
        dy = delta[i, :, :, 1]
        m = mag[i]
        frame = flow_to_rgb(dx, dy, m, mag_scale)
        if mask is not None:
            frame = frame.copy()
            frame[~mask] = 0
        writer.write(frame)

    writer.release()

    if need_transcode:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; cannot transcode to h264")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            args.output,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.remove(temp_output)
        except OSError:
            pass

    print(f"[DenseFlowVis] saved: {args.output}")
    print(f"[DenseFlowVis] frames={t}, size={w}x{h}, mag_scale={mag_scale:.6f}")


if __name__ == "__main__":
    main()
