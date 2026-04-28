#!/usr/bin/env python3
"""
Uniform grid point tracking from multiple camera viewpoints.
- Generates densely packed uniform grid points on frame 0
- Automatically tracks all points through the episode
- Includes background elements (table, floor) by default
"""

import os
import sys
import argparse
import importlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import *
import sapien.core as sapien
import yaml


def class_decorator(task_name):
    """Load and instantiate a task class"""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as e:
        raise SystemExit(f"Failed to load task '{task_name}': {e}")
    return env_instance


def get_embodiment_config(robot_file):
    """Load embodiment configuration"""
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_args(task_name, task_config):
    """Load task configuration from YAML"""
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


def build_scene_entity_map(task):
    """Build mapping from per_scene_id to entity"""
    entity_map = {}

    for actor in task.scene.get_all_actors():
        sid = get_entity_scene_id(actor)
        if sid is not None:
            entity_map[sid] = actor

    for link in task.robot.left_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    for link in task.robot.right_entity.get_links():
        sid = get_entity_scene_id(link)
        if sid is not None:
            entity_map[sid] = link

    return entity_map


def create_markers(scene, n, radius):
    """Create n marker spheres"""
    markers = []
    colors = []
    for i in range(n):
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius)
        marker = builder.build_kinematic(name=f"uniform_track_marker_{i}")
        markers.append(marker)
        colors.append([
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
            float(np.random.uniform(0.2, 1.0)),
        ])
    return markers, colors


def colorize_markers(markers, colors):
    """Set colors for marker spheres"""
    for marker, c in zip(markers, colors):
        try:
            render_body = marker.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                shape.material.set_base_color([c[0], c[1], c[2], 1.0])
        except Exception:
            continue


def get_entity_scene_id(entity_or_link):
    """Get per_scene_id from entity or link"""
    if hasattr(entity_or_link, 'per_scene_id'):
        sid = entity_or_link.per_scene_id
        if sid is not None:
            return int(sid)
    # Fallback for PhysxArticulationLinkComponent
    if hasattr(entity_or_link, 'entity') and hasattr(entity_or_link.entity, 'per_scene_id'):
        sid = entity_or_link.entity.per_scene_id
        if sid is not None:
            return int(sid)
    return None


def get_background_scene_ids(task):
    """Get segmentation IDs for background elements (table, wall, ground)"""
    background_ids = set([0])  # ID 0 is invalid/background
    
    # Check for static scene actors
    for actor in task.scene.get_all_actors():
        if actor.name in ['table', 'wall', 'ground', 'plane']:
            if hasattr(actor, 'per_scene_id') and actor.per_scene_id is not None:
                background_ids.add(int(actor.per_scene_id))
    
    return background_ids


def resolve_source_camera(task, camera_name):
    """Resolve camera name to actual camera object"""
    if camera_name == "world_camera":
        camera_name = "world_camera1"

    # First try direct attribute access
    cam = getattr(task.cameras, camera_name, None)
    if cam is not None:
        return cam

    # Fallback checks for specific cameras
    obs = getattr(task.cameras, "observer_camera", None)
    if camera_name == "observer_camera" and obs is not None:
        return obs

    world1 = getattr(task.cameras, "world_camera1", None)
    world2 = getattr(task.cameras, "world_camera2", None)
    head_cam = getattr(task.cameras, "head_camera", None)
    left_cam = getattr(task.cameras, "left_camera", None)
    right_cam = getattr(task.cameras, "right_camera", None)

    if camera_name == "world_camera1" and world1 is not None:
        return world1
    if camera_name == "world_camera2" and world2 is not None:
        return world2
    if camera_name == "head_camera" and head_cam is not None:
        return head_cam
    if camera_name == "left_camera" and left_cam is not None:
        return left_cam
    if camera_name == "right_camera" and right_cam is not None:
        return right_cam

    # List available cameras
    available = []
    if obs is not None:
        available.append("observer_camera")
    if world1 is not None:
        available.append("world_camera1")
    if world2 is not None:
        available.append("world_camera2")
    if head_cam is not None:
        available.append("head_camera")
    if left_cam is not None:
        available.append("left_camera")
    if right_cam is not None:
        available.append("right_camera")

    raise KeyError(
        f"Unknown camera '{camera_name}'. Available cameras: {', '.join(available) if available else 'None found'}"
    )


def point_texture_to_world(camera, position_texture, u, v):
    """Convert position texture coordinates to world space"""
    p_cam = position_texture[v, u, :3].astype(np.float32)
    p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32)
    T_c2w = camera.get_model_matrix().astype(np.float32)
    return (T_c2w @ p_h)[:3].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Uniform grid point tracking with all camera angles and background included"
    )
    parser.add_argument("task_name", type=str, help="Task name")
    parser.add_argument("task_config", type=str, help="Task config name")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="world_camera1",
        choices=["observer_camera", "world_camera1", "world_camera2", "left_camera", "right_camera"],
        help="Source camera for frame-0 point sampling",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Target number of uniformly distributed grid points",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.004,
        help="Marker sphere radius in meters",
    )
    parser.add_argument(
        "--exclude-background",
        action="store_true",
        help="Exclude table/wall/ground from point sampling (default: include all)",
    )
    args = parser.parse_args()

    # Load task config
    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)

    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False
    cfg["render_freq"] = 1

    # Load episode seed
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

    # Capture frame 0 and generate uniform grid points
    task.get_obs()
    source_camera = resolve_source_camera(task, args.camera)
    source_camera.take_picture()

    position = source_camera.get_picture("Position")
    segmentation = source_camera.get_picture("Segmentation")

    seg_raw = segmentation[..., 1].astype(np.int32)
    valid = position[..., 3] < 1

    # Optionally exclude background
    if args.exclude_background:
        background_ids = get_background_scene_ids(task)
        valid &= seg_raw > 0
    
    # Generate uniform grid points
    h, w = position.shape[:2]
    num_points = args.num_points
    grid_area = h * w
    target_density = num_points / max(1, grid_area)
    grid_step = max(1, int(np.sqrt(1.0 / target_density)))
    
    vv_grid, uu_grid = np.meshgrid(
        np.arange(0, h, grid_step),
        np.arange(0, w, grid_step),
        indexing='ij'
    )
    vv_grid = vv_grid.flatten().astype(np.int32)
    uu_grid = uu_grid.flatten().astype(np.int32)
    
    # Filter to valid pixels
    valid_mask = valid[vv_grid, uu_grid]
    uu = uu_grid[valid_mask]
    vv = vv_grid[valid_mask]
    
    # Subsample if still too many
    if len(uu) > num_points:
        sel = np.random.choice(len(uu), num_points, replace=False)
        uu = uu[sel]
        vv = vv[sel]

    if len(uu) == 0:
        raise RuntimeError("No valid points found")

    # Get world space coordinates and metadata
    sid = seg_raw[vv, uu]
    p_world0 = np.array(
        [point_texture_to_world(source_camera, position, u, v) for u, v in zip(uu, vv)],
        dtype=np.float32
    )
    world_min = p_world0.min(axis=0)
    world_max = p_world0.max(axis=0)

    # Build entity map and create markers
    entity_map = build_scene_entity_map(task)
    owners = []
    for seg_id in sid:
        entity_id = int(seg_id)
        if entity_id in entity_map:
            owners.append(entity_map[entity_id])
        else:
            owners.append(None)

    # Create marker spheres
    markers, colors = create_markers(task.scene, len(p_world0), radius=args.marker_radius)
    colorize_markers(markers, colors)
    
    # Initialize marker positions to frame-0 world coordinates
    for i, (marker, p) in enumerate(zip(markers, p_world0)):
        marker.set_pose(sapien.Pose(p=p.tolist()))

    print(f"[UniformTracking] episode={ep}, camera={args.camera}, sampled_points={len(uu)}")
    print(f"[UniformTracking] world bounds min={world_min} max={world_max}")

    # Callback to update marker positions each frame
    def wrapped_update_render(freq):
        for i, (marker, owner) in enumerate(zip(markers, owners)):
            if owner is not None:
                owner_pose = owner.get_pose()
                # Transform point from world space to owner's local space
                p_owner_local = owner_pose.inv() @ sapien.Pose(p=p_world0[i].tolist())
                marker.set_pose(owner_pose @ p_owner_local)

    task.render_callbacks.append(wrapped_update_render)

    # Play the episode
    task.play_once()


if __name__ == "__main__":
    main()
