#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py  # type: ignore
import yaml  # type: ignore

CPU_RE = re.compile(r"Percent of CPU this job got:\s*([\d\.]+)%")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def discover_task_names(repo_root: Path) -> List[str]:
    env_dir = repo_root / "envs"
    if not env_dir.exists():
        return []

    task_names: List[str] = []
    ignore_names = {"__init__", "_base_task", "_GLOBAL_CONFIGS"}

    for py_file in sorted(env_dir.glob("*.py")):
        name = py_file.stem
        if name in ignore_names:
            continue
        if name.startswith("_"):
            continue
        task_names.append(name)

    return task_names


@dataclass
class CmdMetrics:
    wall_s: float
    cpu_percent: Optional[float]
    max_rss_kb: Optional[int]
    gpu_avg_util: Optional[float]
    gpu_peak_util: Optional[float]
    gpu_peak_mem_mb: Optional[int]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def parse_time_metrics(stderr_text: str) -> Dict[str, Optional[float]]:
    cpu = None
    rss = None
    m = CPU_RE.search(stderr_text)
    if m:
        cpu = float(m.group(1))
    m = RSS_RE.search(stderr_text)
    if m:
        rss = int(m.group(1))
    return {"cpu_percent": cpu, "max_rss_kb": rss}


def gpu_sampler(stop_evt: threading.Event, gpus: List[int], samples: List[Dict[str, float]]):
    query = [
        "nvidia-smi",
        f"--id={','.join(str(g) for g in gpus)}",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    while not stop_evt.is_set():
        try:
            out = subprocess.run(query, capture_output=True, text=True, check=False)
            if out.returncode == 0 and out.stdout.strip():
                utils = []
                mems = []
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 2:
                        continue
                    util = float(parts[0])
                    mem = float(parts[1])
                    utils.append(util)
                    mems.append(mem)
                if utils:
                    samples.append(
                        {
                            "util_avg": sum(utils) / len(utils),
                            "util_peak": max(utils),
                            "mem_peak": max(mems),
                        }
                    )
        except Exception:
            pass
        stop_evt.wait(1.0)


def run_timed_command(
    cmd: List[str],
    env: Dict[str, str],
    gpus: List[int],
    log_file: Path,
) -> CmdMetrics:
    stop_evt = threading.Event()
    gpu_samples: List[Dict[str, float]] = []
    sampler = threading.Thread(target=gpu_sampler, args=(stop_evt, gpus, gpu_samples), daemon=True)
    sampler.start()

    start = time.time()
    wrapped = ["/usr/bin/time", "-v"] + cmd
    proc = subprocess.run(wrapped, env=env, capture_output=True, text=True)
    wall = time.time() - start

    stop_evt.set()
    sampler.join(timeout=2.0)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{now_iso()}] CMD: {' '.join(cmd)}\n")
        f.write(f"RET: {proc.returncode}\n")
        if proc.stdout:
            f.write("--- STDOUT ---\n")
            f.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                f.write("\n")
        if proc.stderr:
            f.write("--- STDERR ---\n")
            f.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                f.write("\n")

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    parsed = parse_time_metrics(proc.stderr)

    if gpu_samples:
        avg_util = sum(s["util_avg"] for s in gpu_samples) / len(gpu_samples)
        peak_util = max(s["util_peak"] for s in gpu_samples)
        peak_mem = int(max(s["mem_peak"] for s in gpu_samples))
    else:
        avg_util = None
        peak_util = None
        peak_mem = None

    return CmdMetrics(
        wall_s=wall,
        cpu_percent=parsed["cpu_percent"],
        max_rss_kb=parsed["max_rss_kb"],
        gpu_avg_util=avg_util,
        gpu_peak_util=peak_util,
        gpu_peak_mem_mb=peak_mem,
    )


def write_metrics_line(metrics_path: Path, payload: Dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def enqueue_task_chunks(task: str, episode_num: int, chunk_size: int) -> List[Dict]:
    jobs = []
    s = 0
    while s < episode_num:
        e = min(episode_num, s + chunk_size)
        jobs.append({"task": task, "start": s, "end": e})
        s = e
    return jobs


def resolve_conda_bin(conda_bin: Optional[str]) -> Optional[str]:
    if conda_bin:
        return conda_bin
    found = shutil.which("conda")
    if found:
        return found
    home = Path.home()
    for candidate in [home / "miniconda3" / "bin" / "conda", home / "anaconda3" / "bin" / "conda"]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def build_python_cmd(args, script_path: str) -> List[str]:
    if getattr(args, "python_bin", None):
        return [args.python_bin, script_path]
    return [args.conda_bin, "run", "-n", args.conda_env, "python", script_path]


def clean_episode_temp(repo_root: Path, task: str, cfg: str, camera: str, episode: int):
    cache_dir = repo_root / "data" / task / cfg / ".cache" / f"episode{episode}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    sf_dir = repo_root / "data" / task / cfg / f"sceneflow_offline_depth_{camera}" / f"episode{episode}"
    for p in [
        sf_dir / "pointcloud_frame0.npy",
        sf_dir / "segmentation_frame0.npy",
        sf_dir / "point_seg_ids_frame0.npy",
    ]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def finalize_episode_h5_mesh_only(repo_root: Path, task: str, cfg: str, episode: int) -> bool:
    h5_path = repo_root / "data" / task / cfg / "data" / f"episode{episode}.hdf5"
    if not h5_path.exists():
        return False

    tmp_path = h5_path.with_suffix(".hdf5.meshonly.tmp")
    with h5py.File(h5_path, "r") as src:
        if "scene_state" not in src or "mesh" not in src["scene_state"]:
            return False

        with h5py.File(tmp_path, "w") as dst:
            dst_scene = dst.require_group("scene_state")
            src.copy("scene_state/mesh", dst_scene, name="mesh")
            dst_scene.attrs["finalized_mesh_only"] = True

    os.replace(tmp_path, h5_path)

    traj_alias = h5_path.parent / f"traj_{episode}.h5"
    try:
        if traj_alias.exists():
            traj_alias.unlink()
        os.link(h5_path, traj_alias)
    except Exception:
        shutil.copy2(h5_path, traj_alias)

    return True


def run_objectflow_export(
    repo_root: Path,
    task: str,
    cfg: str,
    episode: int,
    args,
    env: Dict[str, str],
    worker_log: Path,
):
    objectflow_dir = repo_root / "data" / task / cfg / "objectflow" / f"episode{episode}"
    meta_path = objectflow_dir / "objectflow_meta.json"

    if (not args.objectflow_overwrite) and meta_path.exists():
        return None, str(objectflow_dir), True

    objectflow_dir.mkdir(parents=True, exist_ok=True)
    cmd_objectflow = [
        *build_python_cmd(args, "script/replay_object_flow.py"),
        task,
        cfg,
        "--episode",
        str(episode),
        "--max-points-per-mesh",
        str(args.objectflow_max_points_per_mesh),
        "--seed",
        str(args.objectflow_seed),
        "--objectflow-dir",
        str(objectflow_dir),
    ]
    if args.objectflow_mesh_only:
        cmd_objectflow.append("--mesh-only")

    m = run_timed_command(cmd_objectflow, env, args.gpus, worker_log)
    return m, str(objectflow_dir), False


def mainline_worker(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    compress_queue: mp.JoinableQueue,
    args,
    base_cfg: Dict,
):
    repo_root = Path(args.repo_root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    logs_dir = repo_root / "logs" / "async_pipeline"
    worker_log = logs_dir / f"mainline_w{worker_id}.log"
    metrics_path = logs_dir / "metrics.jsonl"

    while True:
        item = task_queue.get()
        if item is None:
            return

        task_name = item
        jobs = enqueue_task_chunks(task_name, args.episode_num, args.chunk_size)

        for job in jobs:
            start_ep = job["start"]
            end_ep = job["end"]

            try:
                cmd_seed = [
                    *build_python_cmd(args, "script/collect_data.py"),
                    task_name, args.task_config,
                    "--override-use-seed", "false",
                    "--override-collect-data", "false",
                    "--override-episode-num", str(end_ep),
                ]
                m = run_timed_command(cmd_seed, env, args.gpus, worker_log)
                write_metrics_line(
                    metrics_path,
                    {
                        "time": now_iso(),
                        "stage": "seed_traj",
                        "task": task_name,
                        "episode_start": start_ep,
                        "episode_end": end_ep,
                        "worker": worker_id,
                        "gpu": gpu_id,
                        "wall_s": round(m.wall_s, 3),
                        "cpu_percent": m.cpu_percent,
                        "max_rss_kb": m.max_rss_kb,
                        "gpu_avg_util": m.gpu_avg_util,
                        "gpu_peak_util": m.gpu_peak_util,
                        "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                    },
                )

                cmd_replay = [
                    *build_python_cmd(args, "script/collect_data.py"),
                    task_name, args.task_config,
                    "--override-use-seed", "true",
                    "--override-collect-data", "true",
                    "--override-episode-num", str(args.episode_num),
                    "--episode-start", str(start_ep),
                    "--episode-end", str(end_ep),
                ]
                m = run_timed_command(cmd_replay, env, args.gpus, worker_log)
                write_metrics_line(
                    metrics_path,
                    {
                        "time": now_iso(),
                        "stage": "collect_hdf5",
                        "task": task_name,
                        "episode_start": start_ep,
                        "episode_end": end_ep,
                        "worker": worker_id,
                        "gpu": gpu_id,
                        "wall_s": round(m.wall_s, 3),
                        "cpu_percent": m.cpu_percent,
                        "max_rss_kb": m.max_rss_kb,
                        "gpu_avg_util": m.gpu_avg_util,
                        "gpu_peak_util": m.gpu_peak_util,
                        "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                    },
                )

                for ep in range(start_ep, end_ep):
                    cmd_offline = [
                        *build_python_cmd(args, "script/offline_depth_sceneflow.py"),
                        task_name, args.task_config,
                        "--episode", str(ep),
                        "--camera", args.camera,
                        "--skip-existing",
                        "--sceneflow-root", f"data/{task_name}/{args.task_config}/sceneflow_offline_depth_{args.camera}",
                    ]
                    m = run_timed_command(cmd_offline, env, args.gpus, worker_log)
                    write_metrics_line(
                        metrics_path,
                        {
                            "time": now_iso(),
                            "stage": "offline_sceneflow",
                            "task": task_name,
                            "episode": ep,
                            "worker": worker_id,
                            "gpu": gpu_id,
                            "wall_s": round(m.wall_s, 3),
                            "cpu_percent": m.cpu_percent,
                            "max_rss_kb": m.max_rss_kb,
                            "gpu_avg_util": m.gpu_avg_util,
                            "gpu_peak_util": m.gpu_peak_util,
                            "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                        },
                    )

                    compress_queue.put({
                        "task": task_name,
                        "episode": ep,
                        "cfg": args.task_config,
                        "camera": args.camera,
                    })
            finally:
                pass


def compress_worker(worker_id: int, compress_queue: mp.JoinableQueue, args):
    repo_root = Path(args.repo_root)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    logs_dir = repo_root / "logs" / "async_pipeline"
    worker_log = logs_dir / f"compress_w{worker_id}.log"
    metrics_path = logs_dir / "metrics.jsonl"

    while True:
        try:
            item = compress_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        if item is None:
            compress_queue.task_done()
            return

        task = item["task"]
        ep = int(item["episode"])
        cfg = item["cfg"]
        camera = item["camera"]

        sf_out_dir = repo_root / "data" / task / cfg / f"sceneflow_offline_depth_{camera}" / f"episode{ep}"
        seg_dir = repo_root / "data" / task / cfg / "data" / f"episode{ep}_npy" / "observation" / camera

        try:
            if sf_out_dir.exists():
                cmd_flow = [
                    *build_python_cmd(args, "script/flow_compress.py"), "compress",
                    "--out_dir", str(sf_out_dir),
                    "--codec", args.flow_codec,
                    "--crf", str(args.flow_crf),
                    "--bits", str(args.flow_bits),
                    "--delete_npy",
                ]
                m = run_timed_command(cmd_flow, env, args.gpus, worker_log)
                write_metrics_line(
                    metrics_path,
                    {
                        "time": now_iso(),
                        "stage": "flow_compress",
                        "task": task,
                        "episode": ep,
                        "camera": camera,
                        "compress_worker": worker_id,
                        "wall_s": round(m.wall_s, 3),
                        "cpu_percent": m.cpu_percent,
                        "max_rss_kb": m.max_rss_kb,
                        "gpu_avg_util": m.gpu_avg_util,
                        "gpu_peak_util": m.gpu_peak_util,
                        "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                    },
                )

            if (not args.skip_point_compress) and seg_dir.exists():
                cmd_point = [
                    *build_python_cmd(args, "script/point_compress.py"),
                    "--mode", "compress",
                    "--seg_dir", str(seg_dir),
                    "--delete-existing",
                ]
                m = run_timed_command(cmd_point, env, args.gpus, worker_log)
                write_metrics_line(
                    metrics_path,
                    {
                        "time": now_iso(),
                        "stage": "point_compress",
                        "task": task,
                        "episode": ep,
                        "camera": camera,
                        "compress_worker": worker_id,
                        "wall_s": round(m.wall_s, 3),
                        "cpu_percent": m.cpu_percent,
                        "max_rss_kb": m.max_rss_kb,
                        "gpu_avg_util": m.gpu_avg_util,
                        "gpu_peak_util": m.gpu_peak_util,
                        "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                    },
                )

            if not args.disable_objectflow:
                m, objectflow_dir, skipped = run_objectflow_export(
                    repo_root=repo_root,
                    task=task,
                    cfg=cfg,
                    episode=ep,
                    args=args,
                    env=env,
                    worker_log=worker_log,
                )
                if skipped:
                    write_metrics_line(
                        metrics_path,
                        {
                            "time": now_iso(),
                            "stage": "objectflow_skip_existing",
                            "task": task,
                            "episode": ep,
                            "camera": camera,
                            "compress_worker": worker_id,
                            "objectflow_dir": objectflow_dir,
                        },
                    )
                else:
                    stage_name = "objectflow_mesh_export" if args.objectflow_mesh_only else "objectflow_export"
                    write_metrics_line(
                        metrics_path,
                        {
                            "time": now_iso(),
                            "stage": stage_name,
                            "task": task,
                            "episode": ep,
                            "camera": camera,
                            "compress_worker": worker_id,
                            "objectflow_dir": objectflow_dir,
                            "wall_s": round(m.wall_s, 3),
                            "cpu_percent": m.cpu_percent,
                            "max_rss_kb": m.max_rss_kb,
                            "gpu_avg_util": m.gpu_avg_util,
                            "gpu_peak_util": m.gpu_peak_util,
                            "gpu_peak_mem_mb": m.gpu_peak_mem_mb,
                        },
                    )

            if args.finalize_mesh_only_h5:
                t0 = time.time()
                finalized = finalize_episode_h5_mesh_only(repo_root, task, cfg, ep)
                write_metrics_line(
                    metrics_path,
                    {
                        "time": now_iso(),
                        "stage": "finalize_h5_mesh_only",
                        "task": task,
                        "episode": ep,
                        "compress_worker": worker_id,
                        "finalized": finalized,
                        "wall_s": round(time.time() - t0, 3),
                    },
                )

            clean_episode_temp(repo_root, task, cfg, camera, ep)
            write_metrics_line(
                metrics_path,
                {
                    "time": now_iso(),
                    "stage": "cleanup",
                    "task": task,
                    "episode": ep,
                    "compress_worker": worker_id,
                },
            )
        except Exception as e:
            write_metrics_line(
                metrics_path,
                {
                    "time": now_iso(),
                    "stage": "compress_error",
                    "task": task,
                    "episode": ep,
                    "compress_worker": worker_id,
                    "error": str(e),
                },
            )
        finally:
            compress_queue.task_done()


def scaler_thread_fn(
    stop_evt: threading.Event,
    compress_queue: mp.JoinableQueue,
    compress_workers: List[mp.Process],
    args,
):
    next_id = len(compress_workers)
    while not stop_evt.is_set():
        try:
            qsize = compress_queue.qsize()
        except Exception:
            qsize = 0

        current = len([p for p in compress_workers if p.is_alive()])
        if qsize > current * 3 and current < args.compress_workers_max:
            p = mp.Process(target=compress_worker, args=(next_id, compress_queue, args), daemon=True)
            p.start()
            compress_workers.append(p)
            next_id += 1
        stop_evt.wait(10.0)


def parse_args():
    p = argparse.ArgumentParser(description="Async collect/offline/compress pipeline launcher")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--task-config", default="aloha-agilex_worldcamera1_randomized_500")
    p.add_argument("--camera", default="world_camera1")
    p.add_argument("--tasks", default="all", help="Comma separated tasks or 'all'")
    p.add_argument("--episode-num", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--main-workers", type=int, default=1)
    p.add_argument("--compress-workers-start", type=int, default=1)
    p.add_argument("--compress-workers-max", type=int, default=1)
    p.add_argument("--flow-codec", default="libx265", choices=["libx265", "ffv1", "libvpx-vp9"])
    p.add_argument("--flow-crf", type=int, default=0)
    p.add_argument("--flow-bits", type=int, default=10, choices=[8, 10])
    p.add_argument("--skip-point-compress", action="store_true")
    p.add_argument("--disable-objectflow", action="store_true")
    p.add_argument("--objectflow-mesh-only", action="store_true")
    p.add_argument("--objectflow-max-points-per-mesh", type=int, default=5000)
    p.add_argument("--objectflow-seed", type=int, default=0)
    p.add_argument("--objectflow-overwrite", action="store_true")
    p.add_argument("--finalize-mesh-only-h5", dest="finalize_mesh_only_h5", action="store_true", default=True)
    p.add_argument("--no-finalize-mesh-only-h5", dest="finalize_mesh_only_h5", action="store_false")
    p.add_argument("--conda-bin", default=None)
    p.add_argument("--conda-env", default="RoboTwin")
    p.add_argument("--python-bin", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    os.chdir(repo_root)

    args.conda_bin = resolve_conda_bin(args.conda_bin)
    if not args.conda_bin or not os.access(args.conda_bin, os.X_OK):
        raise SystemExit(
            "Could not find an executable conda binary. "
            "Set --conda-bin /path/to/conda or add conda to PATH."
        )

    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if len(args.gpus) < 1:
        raise SystemExit("At least one GPU id is required, e.g. --gpus 0 or --gpus 0,1,2,3")

    cfg_path = repo_root / "task_config" / f"{args.task_config}.yml"
    if not cfg_path.exists():
        raise SystemExit(f"task config missing: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    episode_num = int(base_cfg.get("episode_num", 100)) if args.episode_num is None else int(args.episode_num)
    args.episode_num = episode_num

    discovered_tasks = discover_task_names(repo_root)
    if args.tasks == "all":
        task_list = discovered_tasks
    else:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if not task_list:
        raise SystemExit("No tasks resolved. Check envs/*.py or --tasks setting.")

    unknown_tasks = [t for t in task_list if t not in discovered_tasks]
    if unknown_tasks:
        raise SystemExit(
            "Unknown task(s): "
            + ", ".join(unknown_tasks)
            + ". Available tasks: "
            + ", ".join(discovered_tasks)
        )

    logs_dir = repo_root / "logs" / "async_pipeline"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "summary.json"

    t0 = time.time()

    task_queue: mp.Queue = mp.Queue()
    compress_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=2048)

    for t in task_list:
        task_queue.put(t)
    for _ in range(args.main_workers):
        task_queue.put(None)

    gpu_bindings = [args.gpus[i % len(args.gpus)] for i in range(args.main_workers)]

    main_workers = []
    for wid in range(args.main_workers):
        p = mp.Process(
            target=mainline_worker,
            args=(wid, gpu_bindings[wid], task_queue, compress_queue, args, base_cfg),
            daemon=True,
        )
        p.start()
        main_workers.append(p)

    compress_workers: List[mp.Process] = []
    for cid in range(args.compress_workers_start):
        p = mp.Process(target=compress_worker, args=(cid, compress_queue, args), daemon=True)
        p.start()
        compress_workers.append(p)

    scaler_stop = threading.Event()
    scaler = threading.Thread(
        target=scaler_thread_fn,
        args=(scaler_stop, compress_queue, compress_workers, args),
        daemon=True,
    )
    scaler.start()

    for p in main_workers:
        p.join()

    failed_main_workers = [
        {"pid": p.pid, "exitcode": p.exitcode}
        for p in main_workers
        if p.exitcode not in (0, None)
    ]
    if failed_main_workers:
        raise SystemExit(f"Mainline worker failure detected: {failed_main_workers}")

    compress_queue.join()

    scaler_stop.set()
    scaler.join(timeout=5.0)

    live_workers = [p for p in compress_workers if p.is_alive()]
    for _ in live_workers:
        compress_queue.put(None)
    compress_queue.join()
    for p in live_workers:
        p.join(timeout=2.0)

    t1 = time.time()

    summary = {
        "start_time": t0,
        "end_time": t1,
        "wall_s": round(t1 - t0, 3),
        "task_config": args.task_config,
        "camera": args.camera,
        "episode_num": args.episode_num,
        "chunk_size": args.chunk_size,
        "tasks": task_list,
        "gpus": args.gpus,
        "main_workers": args.main_workers,
        "compress_workers_start": args.compress_workers_start,
        "compress_workers_max": args.compress_workers_max,
        "objectflow_enabled": not args.disable_objectflow,
        "objectflow_mesh_only": args.objectflow_mesh_only,
        "objectflow_max_points_per_mesh": args.objectflow_max_points_per_mesh,
        "objectflow_seed": args.objectflow_seed,
        "objectflow_overwrite": args.objectflow_overwrite,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] async pipeline finished")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] metrics: {logs_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
