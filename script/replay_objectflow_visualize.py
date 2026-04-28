import argparse
import importlib
import os
import subprocess
import sys

import cv2
import numpy as np
import sapien.core as sapien
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from envs import *


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


def get_entity_pose_matrix(entity):
    get_entity_pose = getattr(entity, "get_entity_pose", None)
    if callable(get_entity_pose):
        pose = get_entity_pose()
    else:
        pose = entity.get_pose()
    return pose.to_transformation_matrix().astype(np.float32)


def pose_to_matrix(pose):
    return pose.to_transformation_matrix().astype(np.float32)


def transform_local_points(local_points, pose_matrix):
    return local_points @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]


def camera_points_to_world(camera, points_cam):
    t_c2w = camera.get_model_matrix().astype(np.float32)
    return points_cam @ t_c2w[:3, :3].T + t_c2w[:3, 3]


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


def merge_collision_mesh_from_entity(entity):
    vertices_all = []
    faces_all = []
    vertex_offset = 0

    for comp in getattr(entity, "components", []):
        get_collision_shapes = getattr(comp, "get_collision_shapes", None)
        if not callable(get_collision_shapes):
            continue

        for shape in get_collision_shapes():
            get_vertices = getattr(shape, "get_vertices", None)
            get_triangles = getattr(shape, "get_triangles", None)
            if not callable(get_vertices) or not callable(get_triangles):
                continue

            try:
                verts = np.asarray(get_vertices(), dtype=np.float32)
                tris = np.asarray(get_triangles(), dtype=np.int32)
            except Exception:
                continue

            if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
                continue
            if tris.ndim != 2 or tris.shape[1] != 3 or len(tris) == 0:
                continue

            # PhysX convex mesh vertices are often returned in unscaled mesh space.
            get_scale = getattr(shape, "get_scale", None)
            if callable(get_scale):
                try:
                    scale = np.asarray(get_scale(), dtype=np.float32).reshape(1, 3)
                    verts = verts * scale
                except Exception:
                    pass

            get_local_pose = getattr(shape, "get_local_pose", None)
            if callable(get_local_pose):
                local_pose = pose_to_matrix(get_local_pose())
                verts = transform_local_points(verts, local_pose)

            vertices_all.append(verts)
            faces_all.append(tris + vertex_offset)
            vertex_offset += verts.shape[0]

    if not vertices_all:
        return None, None

    return np.concatenate(vertices_all, axis=0), np.concatenate(faces_all, axis=0)


def sample_surface_points(vertices, faces, num_points, rng):
    if vertices is None or faces is None:
        return np.empty((0, 3), dtype=np.float32)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0 or num_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        fallback = vertices
        if len(fallback) > num_points:
            sample_idx = np.sort(rng.choice(len(fallback), size=num_points, replace=False))
            fallback = fallback[sample_idx]
        return fallback.astype(np.float32)

    triangles = triangles[valid]
    areas = areas[valid]
    probs = areas / np.sum(areas)
    face_idx = rng.choice(len(triangles), size=num_points, replace=True, p=probs)
    chosen = triangles[face_idx]

    r1 = rng.random(num_points, dtype=np.float32)
    r2 = rng.random(num_points, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    b0 = 1.0 - sqrt_r1
    b1 = sqrt_r1 * (1.0 - r2)
    b2 = sqrt_r1 * r2

    points = (
        chosen[:, 0] * b0[:, None]
        + chosen[:, 1] * b1[:, None]
        + chosen[:, 2] * b2[:, None]
    )
    return points.astype(np.float32)


def resolve_camera(task, camera_name):
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

    if camera_name == "observer_camera" and obs is not None:
        return obs
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
    for name in static_names:
        if camera_name == name:
            return task.static_cameras[name]

    # Robust fallback for tasks without an explicit head camera object.
    if camera_name == "head_camera":
        if world1 is not None:
            return world1
        if obs is not None:
            return obs
        if left_cam is not None:
            return left_cam
        if right_cam is not None:
            return right_cam

    # Generic fallback: pick the first available stream.
    for cam in [world1, obs, world2, left_cam, right_cam]:
        if cam is not None:
            return cam
    raise KeyError(f"Unknown camera '{camera_name}'.")


def create_markers(scene, n, radius):
    markers = []
    for i in range(n):
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius)
        marker = builder.build_kinematic(name=f"objflow_marker_{i}")
        markers.append(marker)
    return markers


def colorize_markers(markers, color_per_marker):
    for marker, c in zip(markers, color_per_marker):
        try:
            render_body = marker.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for shape in render_body.render_shapes:
                shape.material.set_base_color([c[0], c[1], c[2], 1.0])
        except Exception:
            continue


def sid_to_color(sid):
    rng = np.random.default_rng(int(sid) + 17)
    c = rng.uniform(0.2, 1.0, size=3)
    return [float(c[0]), float(c[1]), float(c[2])]


def make_task_for_episode(task_name, task_config, episode):
    task = class_decorator(task_name)
    cfg = load_args(task_name, task_config)
    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]

    task.setup_demo(now_ep_num=episode, seed=seeds[episode], **cfg)
    traj = task.load_tran_data(episode)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)
    return task, cfg


def detect_grasped_object_names(task, robot_name_set, ignore_name_set):
    grasped_names = set()
    frame_count = 0

    original_update = task._update_render

    def wrapped_update():
        nonlocal frame_count
        frame_count += 1

        for contact in task.scene.get_contacts():
            e0 = contact.bodies[0].entity
            e1 = contact.bodies[1].entity
            n0 = getattr(e0, "name", "")
            n1 = getattr(e1, "name", "")

            sid0 = get_entity_scene_id(e0)
            sid1 = get_entity_scene_id(e1)
            if sid0 is None or sid1 is None:
                continue

            side0_is_gripper = n0 in set(getattr(task.robot, "gripper_name", []))
            side1_is_gripper = n1 in set(getattr(task.robot, "gripper_name", []))
            if not side0_is_gripper and not side1_is_gripper:
                continue

            other_name = n1 if side0_is_gripper else n0

            if other_name in robot_name_set:
                continue
            if other_name in ignore_name_set:
                continue
            if other_name == "":
                continue

            grasped_names.add(other_name)

        original_update()

    task._update_render = wrapped_update
    try:
        task.play_once()
    except Exception:
        pass

    return frame_count, grasped_names


def main():
    parser = argparse.ArgumentParser(description="Replay objectflow-like marker visualization in full simulation scene")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--record-camera", type=str, default="head_camera")
    parser.add_argument("--max-points-per-entity", type=int, default=1500)
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-robot", action="store_true")
    parser.add_argument("--include-grasped", action="store_true")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--mp4-fps", type=float, default=16.0)
    args = parser.parse_args()

    if not args.include_robot and not args.include_grasped:
        args.include_robot = True
        args.include_grasped = True

    episode = int(args.episode)

    # Pass 1: detect grasped object ids and frame length.
    detect_task, cfg = make_task_for_episode(args.task_name, args.task_config, episode)

    robot_name_set = set()
    for link in detect_task.robot.left_entity.get_links() + detect_task.robot.right_entity.get_links():
        name = link.get_name()
        if name:
            robot_name_set.add(name)

    ignore_name_set = {"table", "wall", "ground"}
    frame_len, grasped_names = detect_grasped_object_names(detect_task, robot_name_set, ignore_name_set)

    selected_names = set()
    if args.include_robot:
        selected_names |= robot_name_set
    if args.include_grasped:
        selected_names |= grasped_names

    if len(selected_names) == 0:
        raise RuntimeError("No target entities selected. Try enabling --include-robot or check contact events.")

    # Pass 2: replay with markers and record full scene.
    task, cfg2 = make_task_for_episode(args.task_name, args.task_config, episode)
    entity_map = build_scene_entity_map(task)

    rng = np.random.default_rng(int(args.seed))
    owner_sid = []
    local_points = []
    marker_colors = []

    selected_sids = []
    for sid, entity in sorted(entity_map.items(), key=lambda x: x[0]):
        entity_name = entity.get_name()
        if entity_name not in selected_names:
            continue

        selected_sids.append(int(sid))

        verts, faces = merge_collision_mesh_from_entity(entity)
        if verts is None or faces is None or len(verts) == 0:
            continue

        pts_local = sample_surface_points(verts, faces, int(args.max_points_per_entity), rng)

        max_n = int(args.max_points_per_entity)
        if max_n > 0 and len(pts_local) > max_n:
            idx = np.sort(rng.choice(len(pts_local), size=max_n, replace=False)).astype(np.int64)
            pts = pts_local[idx]
        else:
            pts = pts_local

        c = sid_to_color(sid)
        for p in pts:
            owner_sid.append(int(sid))
            local_points.append(p.astype(np.float32))
            marker_colors.append(c)

    if len(local_points) == 0:
        raise RuntimeError("No local mesh points collected for selected entities")

    local_points = np.asarray(local_points, dtype=np.float32)
    owner_sid = np.asarray(owner_sid, dtype=np.int32)

    markers = create_markers(task.scene, len(local_points), radius=args.marker_radius)
    colorize_markers(markers, marker_colors)

    if args.mp4_path is None:
        args.mp4_path = os.path.join(cfg2["save_path"], f"objectflow_replay_ep{episode}_{args.record_camera}.mp4")
    os.makedirs(os.path.dirname(args.mp4_path), exist_ok=True)

    record_camera = resolve_camera(task, args.record_camera)

    sid_to_indices = {}
    for i, sid in enumerate(owner_sid.tolist()):
        sid_to_indices.setdefault(int(sid), []).append(i)

    video_ffmpeg = None
    video_writer = None
    frame_idx = 0

    original_update = task._update_render

    def wrapped_update():
        nonlocal video_ffmpeg, video_writer, frame_idx

        current_map = build_scene_entity_map(task)
        for sid, idx_list in sid_to_indices.items():
            idx_arr = np.asarray(idx_list, dtype=np.int64)
            entity = current_map.get(int(sid), None)
            if entity is None:
                continue
            pose = get_entity_pose_matrix(entity)
            p_world = transform_local_points(local_points[idx_arr], pose)
            for j, marker_idx in enumerate(idx_arr.tolist()):
                markers[marker_idx].set_pose(sapien.Pose(p=p_world[j].tolist()))

        original_update()

        if args.save_mp4:
            record_camera.take_picture()
            frame_rgb = to_uint8_rgb(record_camera.get_picture("Color"))
            if frame_rgb is not None:
                if video_ffmpeg is None and video_writer is None:
                    h0, w0 = frame_rgb.shape[:2]
                    try:
                        video_ffmpeg = start_ffmpeg_mp4_writer(args.mp4_path, w0, h0, args.mp4_fps)
                    except Exception:
                        video_ffmpeg = None
                    if video_ffmpeg is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(args.mp4_path, fourcc, float(max(1.0, args.mp4_fps)), (w0, h0))
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Failed to open VideoWriter for {args.mp4_path}")

                if video_ffmpeg is not None and video_ffmpeg.stdin is not None:
                    try:
                        video_ffmpeg.stdin.write(frame_rgb.tobytes())
                    except Exception:
                        video_ffmpeg = None
                elif video_writer is not None:
                    video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        print(f"[ObjectFlowViz] Rendering frame {frame_idx}/{frame_len}", end="\r")

    task._update_render = wrapped_update

    try:
        task.play_once()
    except Exception as e:
        print(f"\n[ObjectFlowViz] replay end: {e}")

    if video_ffmpeg is not None:
        if video_ffmpeg.stdin is not None:
            video_ffmpeg.stdin.close()
        video_ffmpeg.wait()

    if video_writer is not None:
        video_writer.release()

    print("\n[ObjectFlowViz] Done")
    if args.save_mp4:
        print(f"[ObjectFlowViz] Saved video: {args.mp4_path}")
    print(f"[ObjectFlowViz] Selected entities: {len(selected_names)}, active sids: {len(selected_sids)}, markers: {len(local_points)}")


if __name__ == "__main__":
    main()
