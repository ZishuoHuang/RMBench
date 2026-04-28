#!/usr/bin/env python3
import importlib
import json
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import sapien.core as sapien
import yaml

try:
    import cv2
except Exception:
    cv2 = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from envs import *


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as e:
        raise SystemExit(f"No such task: {task_name}, err={e}")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_task_args(task_name, task_config):
    cfg_path = f"./task_config/{task_config}.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config

    return args


def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def quantize_with_step(value, v_min, v_max, step):
    if step <= 0:
        return float(np.clip(value, v_min, v_max))
    k = round((float(value) - float(v_min)) / float(step))
    snapped = float(v_min) + k * float(step)
    return float(np.clip(snapped, v_min, v_max))


def sample_axis(v_min, v_max, step):
    v = random.uniform(v_min, v_max)
    return quantize_with_step(v, v_min, v_max, step)


def sample_candidate(cli_args, camera_name):
    use_front_view = random.random() < float(np.clip(cli_args.front_view_ratio, 0.0, 1.0))

    if use_front_view:
        pos_x_min, pos_x_max = cli_args.front_pos_x_min, cli_args.front_pos_x_max
        pos_y_min, pos_y_max = cli_args.front_pos_y_min, cli_args.front_pos_y_max
        pos_z_min, pos_z_max = cli_args.front_pos_z_min, cli_args.front_pos_z_max

        look_x_min, look_x_max = cli_args.front_look_x_min, cli_args.front_look_x_max
        look_y_min, look_y_max = cli_args.front_look_y_min, cli_args.front_look_y_max
        look_z_min, look_z_max = cli_args.front_look_z_min, cli_args.front_look_z_max
    else:
        pos_x_min, pos_x_max = cli_args.pos_x_min, cli_args.pos_x_max
        pos_y_min, pos_y_max = cli_args.pos_y_min, cli_args.pos_y_max
        pos_z_min, pos_z_max = cli_args.pos_z_min, cli_args.pos_z_max

        look_x_min, look_x_max = cli_args.look_x_min, cli_args.look_x_max
        look_y_min, look_y_max = cli_args.look_y_min, cli_args.look_y_max
        look_z_min, look_z_max = cli_args.look_z_min, cli_args.look_z_max

    pos = np.array(
        [
            sample_axis(pos_x_min, pos_x_max, cli_args.pos_step),
            sample_axis(pos_y_min, pos_y_max, cli_args.pos_step),
            sample_axis(pos_z_min, pos_z_max, cli_args.pos_step),
        ],
        dtype=np.float32,
    )
    look = np.array(
        [
            sample_axis(look_x_min, look_x_max, cli_args.look_step),
            sample_axis(look_y_min, look_y_max, cli_args.look_step),
            sample_axis(look_z_min, look_z_max, cli_args.look_step),
        ],
        dtype=np.float32,
    )

    forward = normalize(look - pos)
    if forward is None:
        return None

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    left = np.cross(world_up, forward)
    left = normalize(left)
    if left is None:
        left = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    return {
        "name": camera_name,
        "view_region": "front" if use_front_view else "default",
        "position": [float(x) for x in pos],
        "forward": [float(x) for x in forward],
        "left": [float(x) for x in left],
    }


def apply_pose_from_vectors(camera, position, forward, left):
    cam_pos = np.asarray(position, dtype=np.float32)
    cam_forward = np.asarray(forward, dtype=np.float32)
    cam_left = np.asarray(left, dtype=np.float32)

    if np.linalg.norm(cam_forward) < 1e-6 or np.linalg.norm(cam_left) < 1e-6:
        return False

    cam_forward = cam_forward / np.linalg.norm(cam_forward)
    cam_left = cam_left / np.linalg.norm(cam_left)
    cam_up = np.cross(cam_forward, cam_left)
    if np.linalg.norm(cam_up) < 1e-6:
        return False
    cam_up = cam_up / np.linalg.norm(cam_up)

    mat44 = np.eye(4, dtype=np.float32)
    mat44[:3, :3] = np.stack([cam_forward, cam_left, cam_up], axis=1)
    mat44[:3, 3] = cam_pos
    camera.entity.set_pose(sapien.Pose(mat44))
    return True


def load_camera_preview_spec(task_config, camera_name):
    cfg_path = f"./task_config/{task_config}.yml"
    width, height, fovy = 640, 480, 50.0

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        camera_cfg = cfg.get("camera", {}) or {}
    except Exception:
        camera_cfg = {}

    if camera_name == "world_camera1":
        width = int(camera_cfg.get("world_camera1_width", width))
        height = int(camera_cfg.get("world_camera1_height", height))
        fovy = float(camera_cfg.get("world_camera1_fovy", fovy))
    elif camera_name == "world_camera2":
        width = int(camera_cfg.get("world_camera2_width", camera_cfg.get("world_camera1_width", width)))
        height = int(camera_cfg.get("world_camera2_height", camera_cfg.get("world_camera1_height", height)))
        fovy = float(camera_cfg.get("world_camera2_fovy", camera_cfg.get("world_camera1_fovy", fovy)))

    return width, height, fovy


class LightweightCameraPreview:

    def __init__(self, camera_name, width, height, fovy_deg):
        self.engine = sapien.Engine()
        from sapien.render import set_global_config

        set_global_config(max_num_materials=4096, max_num_textures=4096)
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240)
        self.scene.add_ground(0)
        self.scene.set_ambient_light([0.55, 0.55, 0.55])
        self.scene.add_directional_light([0.2, 0.5, -1.0], [0.9, 0.9, 0.9], shadow=False)
        self.scene.add_point_light([1.0, -1.0, 2.0], [0.7, 0.7, 0.7], shadow=False)

        # Add a simple table-like reference and one cube as depth/scale cue.
        table = self.scene.create_actor_builder()
        table.add_box_visual(half_size=[0.6, 0.35, 0.025])
        table.add_box_collision(half_size=[0.6, 0.35, 0.025])
        self.table_actor = table.build_kinematic(name="preview_table")
        self.table_actor.set_pose(sapien.Pose([0.0, 0.0, 0.74]))

        cube = self.scene.create_actor_builder()
        cube.add_box_visual(half_size=[0.04, 0.04, 0.04])
        cube.add_box_collision(half_size=[0.04, 0.04, 0.04])
        self.ref_cube = cube.build_kinematic(name="preview_cube")
        self.ref_cube.set_pose(sapien.Pose([0.15, 0.0, 0.82]))

        self.camera = self.scene.add_camera(
            name=camera_name,
            width=int(width),
            height=int(height),
            fovy=np.deg2rad(float(fovy_deg)),
            near=0.1,
            far=100,
        )

    def render_rgb(self):
        self.scene.step()
        self.scene.update_render()
        self.camera.take_picture()
        rgba = self.camera.get_picture("Color")
        return (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)


def render_exact_camera_preview(task_env, camera, window_name="Exact Camera Preview"):
    """Render and show the exact RGB image from the target SAPIEN camera."""
    if cv2 is None:
        return

    task_env._update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imshow(window_name, bgr)
    cv2.waitKey(1)


def save_candidates(output_path, task_name, task_config, camera_name, accepted):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(accepted) == 0 and os.path.exists(output_path):
        keep_count = 0
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
            existing_candidates = existing.get("random_camera_candidates", [])
            if isinstance(existing_candidates, list):
                keep_count = int(len(existing_candidates))
        except Exception:
            keep_count = 0

        # Keep existing file when user quits without accepting a new view.
        return {
            "written": False,
            "count": keep_count,
            "path": output_path,
        }

    payload = {
        "task_name": task_name,
        "task_config": task_config,
        "camera_name": camera_name,
        "count": int(len(accepted)),
        "random_camera_candidates": accepted,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return {
        "written": True,
        "count": int(len(accepted)),
        "path": output_path,
    }


def update_task_config_with_file(task_config, camera_name, candidates_file):
    cfg_path = f"./task_config/{task_config}.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    camera_cfg = cfg.setdefault("camera", {})
    camera_cfg["record_single_camera"] = True
    camera_cfg["record_camera_name"] = camera_name
    camera_cfg["random_camera_candidates_file"] = candidates_file

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def main():
    parser = ArgumentParser(description="Interactive camera view picker for random replay camera candidates")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--camera-name", type=str, default="world_camera1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-update-task-config", action="store_true")
    parser.add_argument("--with-viewer", action="store_true", help="Also open SAPIEN viewer window (higher GPU memory)")
    parser.add_argument(
        "--load-task-scene",
        action="store_true",
        help="Load full task scene for preview (heavier). Default uses lightweight camera-only scene.",
    )

    parser.add_argument("--pos-x-min", type=float, default=-0.35)
    parser.add_argument("--pos-x-max", type=float, default=0.95)
    parser.add_argument("--pos-y-min", type=float, default=-1.0)
    parser.add_argument("--pos-y-max", type=float, default=0.35)
    parser.add_argument("--pos-z-min", type=float, default=0.95)
    parser.add_argument("--pos-z-max", type=float, default=2.05)

    parser.add_argument("--look-x-min", type=float, default=-0.35)
    parser.add_argument("--look-x-max", type=float, default=0.55)
    parser.add_argument("--look-y-min", type=float, default=-0.25)
    parser.add_argument("--look-y-max", type=float, default=0.75)
    parser.add_argument("--look-z-min", type=float, default=0.6)
    parser.add_argument("--look-z-max", type=float, default=1.35)

    parser.add_argument("--front-view-ratio", type=float, default=0.45)
    parser.add_argument("--front-pos-x-min", type=float, default=-0.35)
    parser.add_argument("--front-pos-x-max", type=float, default=0.95)
    parser.add_argument("--front-pos-y-min", type=float, default=0.35)
    parser.add_argument("--front-pos-y-max", type=float, default=1.05)
    parser.add_argument("--front-pos-z-min", type=float, default=0.95)
    parser.add_argument("--front-pos-z-max", type=float, default=2.05)

    parser.add_argument("--front-look-x-min", type=float, default=-0.30)
    parser.add_argument("--front-look-x-max", type=float, default=0.50)
    parser.add_argument("--front-look-y-min", type=float, default=-0.25)
    parser.add_argument("--front-look-y-max", type=float, default=0.35)
    parser.add_argument("--front-look-z-min", type=float, default=0.6)
    parser.add_argument("--front-look-z-max", type=float, default=1.35)

    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.14,
        help="Position sampling step in meters. Larger means coarser spacing.",
    )
    parser.add_argument(
        "--look-step",
        type=float,
        default=0.12,
        help="Look-target sampling step in meters. Larger means coarser spacing.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output is None:
        args.output = os.path.join("task_config", "camera_candidates", f"{args.task_config}_{args.camera_name}.yml")

    task = None
    preview = None
    cam = None
    if args.load_task_scene:
        task = class_decorator(args.task_name)
        env_args = load_task_args(args.task_name, args.task_config)
        env_args["render_freq"] = 1 if args.with_viewer else 0
        env_args["save_data"] = False
        env_args["collect_data"] = False
        env_args["need_plan"] = False
        env_args["eval_mode"] = False

        task.setup_demo(now_ep_num=0, seed=args.seed, **env_args)
        cam = task._resolve_camera_handle(args.camera_name)
        if cam is None:
            task.close_env(clear_cache=False)
            if hasattr(task, "viewer") and task.viewer is not None:
                task.viewer.close()
            raise RuntimeError(f"camera not found: {args.camera_name}")
    else:
        width, height, fovy = load_camera_preview_spec(args.task_config, args.camera_name)
        preview = LightweightCameraPreview(args.camera_name, width, height, fovy)
        cam = preview.camera

    accepted = []
    print("\nInteractive camera picker")
    print("Controls in terminal: y=accept, n=next, q=save+quit")
    print("Use 'Exact Camera Preview' window as the final test camera view.")
    print(
        "Sampling: dual-region (default + front view), "
        f"front_ratio={args.front_view_ratio:.2f}, "
        f"pos_step={args.pos_step:.2f}m, look_step={args.look_step:.2f}m"
    )
    if args.load_task_scene:
        print("Preview source: full task scene")
    else:
        print("Preview source: lightweight scene (no task objects/robot)")
    if not args.with_viewer and args.load_task_scene:
        print("Viewer window disabled to reduce GPU memory usage.")

    try:
        for i in range(args.max_samples):
            candidate = sample_candidate(args, args.camera_name)
            if candidate is None:
                continue

            ok = apply_pose_from_vectors(
                cam,
                candidate["position"],
                candidate["forward"],
                candidate["left"],
            )
            if not ok:
                continue

            if task is not None:
                render_exact_camera_preview(task, cam)
            else:
                if cv2 is not None:
                    rgb = preview.render_rgb()
                    cv2.imshow("Exact Camera Preview", rgb[:, :, ::-1])
                    cv2.waitKey(1)

            print("\nSample", i)
            print(json.dumps(candidate, indent=2))
            cmd = input("Accept this view? [y/n/q]: ").strip().lower()
            if cmd in ("y", "yes"):
                accepted.append(candidate)
                print(f"Accepted count: {len(accepted)}")
            elif cmd in ("q", "quit"):
                break
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        if task is not None:
            task.close_env(clear_cache=False)
            if hasattr(task, "viewer") and task.viewer is not None:
                task.viewer.close()

    save_result = save_candidates(args.output, args.task_name, args.task_config, args.camera_name, accepted)
    if save_result["written"]:
        print(f"Saved {save_result['count']} candidates to: {save_result['path']}")
    else:
        print(
            "No new view accepted; keep existing candidates: "
            f"{save_result['path']} (count={save_result['count']})"
        )

    if not args.no_update_task_config:
        update_task_config_with_file(args.task_config, args.camera_name, args.output)
        print(f"Updated task config to use random_camera_candidates_file: task_config/{args.task_config}.yml")


if __name__ == "__main__":
    main()
