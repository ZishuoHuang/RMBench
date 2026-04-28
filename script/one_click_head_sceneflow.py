import argparse
import glob
import importlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

try:
    import cv2
except Exception:
    cv2 = None

import sys

sys.path.append("./")
from generate_sceneflow import SceneFlowGenerator
from envs import CONFIGS_PATH


def find_episode_hdf5(task_name: str, task_config: str, episode: int, explicit_hdf5: str = None) -> str:
    if explicit_hdf5:
        if os.path.isfile(explicit_hdf5):
            return explicit_hdf5
        raise FileNotFoundError(f"HDF5 not found: {explicit_hdf5}")

    ep_name = f"episode{episode}.hdf5"
    roots = [
        os.path.join("data", task_name, task_config, "data"),
        os.path.join("data", task_name, task_config, "data_done"),
        os.path.join("data", task_name, task_config),
    ]

    for r in roots:
        p = os.path.join(r, ep_name)
        if os.path.isfile(p):
            return p

    patterns = [
        os.path.join("data", "**", ep_name),
        os.path.join("**", ep_name),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Cannot find {ep_name}. You can pass --hdf5 /path/to/{ep_name} explicitly."
    )


def stable_seg_color(seg_id: int):
    s = int(seg_id)
    r = (s * 37 + 53) % 256
    g = (s * 97 + 29) % 256
    b = (s * 17 + 131) % 256
    return int(r), int(g), int(b)


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file: str):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_task_args(task_name: str, task_config: str):
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


def get_entity_name(entity):
    fn = getattr(entity, "get_name", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    ent = getattr(entity, "entity", None)
    ent_fn = getattr(ent, "get_name", None) if ent is not None else None
    if callable(ent_fn):
        try:
            return str(ent_fn())
        except Exception:
            pass
    return "unknown"


def build_seg_id_name_map(task_name: str, task_config: str, episode: int):
    try:
        task = class_decorator(task_name)
        cfg = load_task_args(task_name, task_config)
        cfg["need_plan"] = False
        cfg["save_data"] = False
        cfg["collect_data"] = False
        cfg["render_freq"] = 0

        seed_file = os.path.join(cfg["save_path"], "seed.txt")
        if os.path.isfile(seed_file):
            with open(seed_file, "r", encoding="utf-8") as f:
                seeds = [int(x) for x in f.read().split()]
            seed = seeds[episode] if 0 <= episode < len(seeds) else 0
        else:
            seed = 0

        task.setup_demo(now_ep_num=episode, seed=seed, **cfg)

        id_name = {}
        robot_sids = set()
        for link in task.robot.left_entity.get_links() + task.robot.right_entity.get_links():
            sid = get_entity_scene_id(link)
            if sid is not None:
                robot_sids.add(int(sid))
                id_name[int(sid)] = f"robot/{get_entity_name(link)}"

        for actor in task.scene.get_all_actors():
            sid = get_entity_scene_id(actor)
            if sid is None:
                continue
            sid = int(sid)
            if sid in id_name:
                continue
            nm = get_entity_name(actor)
            if sid in robot_sids:
                id_name[sid] = f"robot/{nm}"
            else:
                id_name[sid] = nm

        try:
            task.close_env()
        except Exception:
            pass

        return id_name
    except Exception as e:
        print(f"[OneClick] auto seg-id name mapping failed: {e}")
        return {}


def load_external_name_map(path: str):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"name map json not found: {path}")
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    return out


def project_points(points_3d: np.ndarray, width: int, height: int, xyz_min: np.ndarray, xyz_max: np.ndarray):
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range == 0] = 1.0
    p = (points_3d - xyz_min) / xyz_range
    x = (p[:, 0] * (width - 1)).astype(np.int32)
    y = ((1.0 - p[:, 1]) * (height - 1)).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], valid


def draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, step: int = 40):
    c = (220, 236, 233)
    for xx in range(0, width, step):
        draw.line([(xx, 0), (xx, height)], fill=c, width=1)
    for yy in range(0, height, step):
        draw.line([(0, yy), (width, yy)], fill=c, width=1)


def render_sceneflow_segmentation(
    sceneflow_dir: str,
    output_gif: str,
    output_mp4: str,
    width: int,
    height: int,
    duration_ms: int,
    mp4_fps: float,
    legend_topk: int,
    point_radius: int,
    seg_name_map: dict,
):
    sf_dir = Path(sceneflow_dir)
    pc0 = np.load(sf_dir / "pointcloud_frame0.npy")
    seg_ids = np.load(sf_dir / "point_seg_ids_frame0.npy").astype(np.int32)
    sf_files = sorted(sf_dir.glob("sceneflow_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    deltas = [np.load(p) for p in sf_files]

    if len(deltas) == 0:
        raise RuntimeError(f"No sceneflow_*.npy found under {sceneflow_dir}")

    if len(seg_ids) != len(pc0):
        raise RuntimeError("point_seg_ids_frame0 length mismatch with pointcloud_frame0")

    all_pts = np.concatenate([pc0 + d for d in deltas], axis=0)
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)

    uniq_ids, counts = np.unique(seg_ids, return_counts=True)
    order = np.argsort(-counts)
    top_ids = uniq_ids[order][: max(1, int(legend_topk))]
    top_counts = counts[order][: max(1, int(legend_topk))]

    frames = []
    mp4_writer = None
    if output_mp4 and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        mp4_writer = cv2.VideoWriter(output_mp4, fourcc, float(max(1.0, mp4_fps)), (width, height))

    for t, d in enumerate(deltas):
        pts = pc0 + d
        x, y, valid = project_points(pts, width, height, xyz_min, xyz_max)
        sid = seg_ids[valid]

        img = Image.new("RGB", (width, height), color=(236, 244, 242))
        draw = ImageDraw.Draw(img)
        draw_grid(draw, width, height, step=36)

        for xx, yy, s in zip(x, y, sid):
            col = stable_seg_color(int(s))
            r = max(1, int(point_radius))
            draw.ellipse((xx - r, yy - r, xx + r, yy + r), fill=col)

        title = f"Scene Flow - Frame {t}"
        draw.text((width // 2 - 70, 14), title, fill=(30, 30, 30))

        lx0 = max(10, width - 300)
        ly0 = 40
        legend_h = min(height - 20, 24 + 18 * len(top_ids))
        draw.rectangle((lx0, ly0, width - 10, ly0 + legend_h), fill=(250, 250, 250), outline=(210, 210, 210))
        for i, (s, c) in enumerate(zip(top_ids, top_counts)):
            yy0 = ly0 + 8 + i * 18
            cc = stable_seg_color(int(s))
            draw.rectangle((lx0 + 8, yy0, lx0 + 16, yy0 + 8), fill=cc)
            readable = seg_name_map.get(int(s), f"id={int(s)}")
            draw.text((lx0 + 22, yy0 - 2), f"{readable} ({int(c)})", fill=(40, 40, 40))

        frames.append(img)

        if mp4_writer is not None:
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            mp4_writer.write(bgr)

    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=max(1, duration_ms), loop=0)
    if mp4_writer is not None:
        mp4_writer.release()


def main():
    parser = argparse.ArgumentParser(description="One-click offline SceneFlow (per-pixel, segmentation colors)")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", type=str, default="head_camera", help="Camera name, e.g. head_camera/world_camera1")
    parser.add_argument("--hdf5", type=str, default=None, help="Optional explicit HDF5 path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--duration", type=int, default=80, help="GIF frame duration in ms")
    parser.add_argument("--fps", type=float, default=16.0, help="MP4 fps")
    parser.add_argument("--legend-topk", type=int, default=24)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--name-map-json", type=str, default=None, help="Optional JSON {seg_id: readable_name}")
    parser.add_argument(
        "--no-auto-name-map",
        action="store_true",
        help="Disable automatic seg_id->name mapping via simulator",
    )
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()

    hdf5_path = find_episode_hdf5(args.task_name, args.task_config, args.episode, args.hdf5)
    print(f"[OneClick] HDF5: {hdf5_path}")

    if args.output_dir is None:
        out_dir = Path("data") / args.task_name / args.task_config / f"sceneflow_ep{args.episode}_{args.camera}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OneClick] Generating SceneFlow from {args.camera} (all valid pixels)...")
    gen = SceneFlowGenerator(hdf5_path)
    gen.generate_sceneflow(str(out_dir), camera_name=args.camera, all_frames=True)

    gif_path = str(out_dir / f"scene_flow_{args.camera}_seg.gif")
    mp4_path = "" if args.no_mp4 else str(out_dir / f"scene_flow_{args.camera}_seg.mp4")

    print("[OneClick] Rendering segmentation-colored animation...")
    seg_name_map = {}
    if args.name_map_json is not None:
        seg_name_map.update(load_external_name_map(args.name_map_json))
    if not args.no_auto_name_map:
        auto_map = build_seg_id_name_map(args.task_name, args.task_config, args.episode)
        seg_name_map.update(auto_map)
    if len(seg_name_map) > 0:
        import json

        with open(out_dir / "seg_id_name_map.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in seg_name_map.items()}, f, ensure_ascii=False, indent=2)
        print(f"[OneClick] seg-id map saved: {out_dir / 'seg_id_name_map.json'}")

    render_sceneflow_segmentation(
        sceneflow_dir=str(out_dir),
        output_gif=gif_path,
        output_mp4=mp4_path,
        width=int(args.width),
        height=int(args.height),
        duration_ms=int(args.duration),
        mp4_fps=float(args.fps),
        legend_topk=int(args.legend_topk),
        point_radius=int(args.point_radius),
        seg_name_map=seg_name_map,
    )

    print(f"[OneClick] GIF: {gif_path}")
    if mp4_path:
        if cv2 is None:
            print("[OneClick] MP4 skipped (opencv-python not installed)")
        else:
            print(f"[OneClick] MP4: {mp4_path}")


if __name__ == "__main__":
    main()