import argparse
import importlib
import json
import os
import sys

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


def main():
    parser = argparse.ArgumentParser(description="Replay and export object flow from mesh-local sampled points")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--max-points-per-mesh", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objectflow-dir", type=str, default=None)
    parser.add_argument("--exact-length", type=int, default=0, help="Skip dry-run and use provided playback length")
    parser.add_argument("--mesh-only", action="store_true", help="Export meshes and sampled local points only")
    args = parser.parse_args()

    # Pass 1: dry-run to get exact frame count.
    if args.mesh_only:
        traj_len = 0
        print("[ObjectFlow] Mesh-only mode: skip frame counting and replay")
    elif args.exact_length > 0:
        traj_len = int(args.exact_length)
        print(f"[ObjectFlow] Using explicit length {traj_len}")
    else:
        print("[ObjectFlow] Dry-run counting total frames...")
        dry_task = class_decorator(args.task_name)
        cfg_dry = load_args(args.task_name, args.task_config)
        cfg_dry["need_plan"] = False
        cfg_dry["save_data"] = False
        with open(os.path.join(cfg_dry["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
            seeds = [int(x) for x in f.read().split()]

        ep = int(args.episode)
        dry_task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg_dry)
        traj = dry_task.load_tran_data(ep)
        cfg_dry["left_joint_path"] = traj["left_joint_path"]
        cfg_dry["right_joint_path"] = traj["right_joint_path"]
        dry_task.set_path_lst(cfg_dry)

        total_sim_frames = 0

        def dry_update_render():
            nonlocal total_sim_frames
            total_sim_frames += 1

        dry_task._update_render = dry_update_render
        try:
            dry_task.play_once()
        except Exception:
            pass
        if hasattr(dry_task, "close"):
            dry_task.close()

        traj_len = max(1, total_sim_frames)
        print(f"[ObjectFlow] Dry-run complete: {traj_len} frames")

    # Pass 2: replay and collect object flow.
    task = class_decorator(args.task_name)
    cfg = load_args(args.task_name, args.task_config)
    cfg["need_plan"] = False
    cfg["save_data"] = False
    cfg["collect_data"] = False

    with open(os.path.join(cfg["save_path"], "seed.txt"), "r", encoding="utf-8") as f:
        seeds = [int(x) for x in f.read().split()]

    ep = int(args.episode)
    task.setup_demo(now_ep_num=ep, seed=seeds[ep], **cfg)
    traj = task.load_tran_data(ep)
    cfg["left_joint_path"] = traj["left_joint_path"]
    cfg["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(cfg)

    entity_map = build_scene_entity_map(task)
    rng = np.random.default_rng(int(args.seed))

    if args.objectflow_dir is None:
        args.objectflow_dir = os.path.join(cfg["save_path"], f"objectflow_ep{ep}")
    mesh_dir = os.path.join(args.objectflow_dir, "meshes")
    flow_dir = os.path.join(args.objectflow_dir, "flows")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    entities = []
    for sid, entity in sorted(entity_map.items(), key=lambda x: x[0]):
        vertices, faces = merge_collision_mesh_from_entity(entity)
        if vertices is None or faces is None:
            continue

        sampled_local = sample_surface_points(vertices, faces, int(args.max_points_per_mesh), rng)

        entity_name = getattr(entity, "get_name", lambda: "")()
        entity_key = f"sid_{int(sid):06d}"
        mesh_path = os.path.join(mesh_dir, f"{entity_key}.npz")
        np.savez_compressed(
            mesh_path,
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            sampled_idx=np.arange(sampled_local.shape[0], dtype=np.int32),
            sampled_points_local=sampled_local.astype(np.float32),
        )

        entities.append(
            {
                "scene_id": int(sid),
                "entity_name": entity_name,
                "mesh_path": os.path.relpath(mesh_path, args.objectflow_dir),
                "num_vertices": int(vertices.shape[0]),
                "num_faces": int(faces.shape[0]),
                "num_sampled_points": int(sampled_local.shape[0]),
                "sampled_points_local": sampled_local,
            }
        )

    if len(entities) == 0:
        raise RuntimeError("No collision mesh could be extracted from scene entities")

    frame_pose_history = []
    original_update_render = task._update_render

    def wrapped_update_render():
        current_map = build_scene_entity_map(task)
        frame_pose_history.append({
            int(s): get_entity_pose_matrix(entity)
            for s, entity in current_map.items()
        })
        original_update_render()
        print(f"[ObjectFlow] Replaying frame {len(frame_pose_history)}/{traj_len}", end="\r")

    if not args.mesh_only:
        task._update_render = wrapped_update_render
        try:
            task.play_once()
        except Exception as e:
            print(f"\n[ObjectFlow] replay end: {e}")

        T = len(frame_pose_history)
        if T <= 0:
            raise RuntimeError("No frame pose history collected")
    else:
        T = 0

    outputs = []
    for item in entities:
        sid = int(item["scene_id"])
        sampled_local = item.pop("sampled_points_local")
        N = sampled_local.shape[0]

        if args.mesh_only:
            outputs.append({**item})
        else:
            points_world = np.lib.format.open_memmap(
                os.path.join(flow_dir, f"object_points_world_sid{sid:06d}.npy"),
                mode="w+",
                dtype=np.float16,
                shape=(T, N, 3),
            )
            points_delta = np.lib.format.open_memmap(
                os.path.join(flow_dir, f"object_points_delta_sid{sid:06d}.npy"),
                mode="w+",
                dtype=np.float16,
                shape=(T, N, 3),
            )
            pose_seq = np.lib.format.open_memmap(
                os.path.join(flow_dir, f"object_pose_sid{sid:06d}.npy"),
                mode="w+",
                dtype=np.float32,
                shape=(T, 4, 4),
            )

            first_pose = frame_pose_history[0].get(sid, np.eye(4, dtype=np.float32))
            anchor_world = transform_local_points(sampled_local, first_pose).astype(np.float32)

            for t, pose_map in enumerate(frame_pose_history):
                pose = pose_map.get(sid, first_pose)
                pose_seq[t] = pose
                world_t = transform_local_points(sampled_local, pose).astype(np.float32)
                points_world[t] = world_t.astype(np.float16)
                points_delta[t] = (world_t - anchor_world).astype(np.float16)

            points_world.flush()
            points_delta.flush()
            pose_seq.flush()

            outputs.append(
                {
                    **item,
                    "world_points_path": os.path.relpath(
                        os.path.join(flow_dir, f"object_points_world_sid{sid:06d}.npy"),
                        args.objectflow_dir,
                    ),
                    "delta_points_path": os.path.relpath(
                        os.path.join(flow_dir, f"object_points_delta_sid{sid:06d}.npy"),
                        args.objectflow_dir,
                    ),
                    "pose_path": os.path.relpath(
                        os.path.join(flow_dir, f"object_pose_sid{sid:06d}.npy"),
                        args.objectflow_dir,
                    ),
                }
            )

    meta = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "episode": ep,
        "mesh_only": bool(args.mesh_only),
        "trajectory_length": int(T),
        "max_points_per_mesh": int(args.max_points_per_mesh),
        "entities": outputs,
    }
    meta_path = os.path.join(args.objectflow_dir, "objectflow_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[ObjectFlow] Export complete")
    print(f"[ObjectFlow] Root: {args.objectflow_dir}")
    print(f"[ObjectFlow] Entities exported: {len(outputs)}")


if __name__ == "__main__":
    main()
