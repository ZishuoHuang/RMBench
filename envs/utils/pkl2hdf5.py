import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def _create_string_dataset(hdf5_group, key, value_list):
    data = []
    for item in value_list:
        if isinstance(item, bytes):
            data.append(item.decode("utf-8", errors="replace"))
        else:
            data.append(str(item))
    hdf5_group.create_dataset(key, data=np.asarray(data, dtype=object), dtype=h5py.string_dtype("utf-8"))


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    # Keep for backward compatibility; dynamic schema accumulation is handled
    # by `append_data_to_structure` + `finalize_data_structure` below.
    if isinstance(data, dict):
        return {}
    return []


def _make_placeholder_like(value):
    if isinstance(value, dict):
        return {k: _make_placeholder_like(v) for k, v in value.items()}
    return None


def _append_missing(node):
    if isinstance(node, dict):
        for child in node.values():
            _append_missing(child)
    else:
        node.append(None)


def _ensure_node_for_new_key(value, frame_idx):
    if isinstance(value, dict):
        node = {k: _ensure_node_for_new_key(v, frame_idx) for k, v in value.items()}
        for _ in range(frame_idx):
            _append_missing(node)
        return node
    return [None] * frame_idx


def append_data_to_structure(data_structure, data):
    def _append(dst, src, frame_idx):
        if isinstance(src, dict):
            if not isinstance(dst, dict):
                return

            # Existing keys missing in current frame get placeholders.
            for key in list(dst.keys()):
                if key == "__frame_count__":
                    continue
                if key not in src:
                    _append_missing(dst[key])

            # New keys discovered after frame0 must be backfilled.
            for key, value in src.items():
                if key == "__frame_count__":
                    continue
                if key not in dst:
                    dst[key] = _ensure_node_for_new_key(value, frame_idx)
                _append(dst[key], value, frame_idx)
            return

        # Leaf node: append frame value directly.
        if isinstance(dst, list):
            dst.append(src)

    frame_idx = int(data_structure.get("__frame_count__", 0))
    _append(data_structure, data, frame_idx)
    data_structure["__frame_count__"] = frame_idx + 1


def _normalize_leaf_sequence(value_list):
    if len(value_list) == 0:
        return value_list

    sample = None
    for v in value_list:
        if v is not None:
            sample = v
            break

    if sample is None:
        return []

    out = []
    if isinstance(sample, np.ndarray):
        zero = np.zeros_like(sample)
        for v in value_list:
            out.append(zero if v is None else v)
        return out

    if isinstance(sample, (str, bytes, np.str_, np.bytes_)):
        fill = "" if isinstance(sample, (str, np.str_)) else b""
        for v in value_list:
            out.append(fill if v is None else v)
        return out

    # Numeric scalar fallback.
    if np.isscalar(sample):
        fill = type(sample)(0)
        for v in value_list:
            out.append(fill if v is None else v)
        return out

    # Generic object fallback: stringify None to keep h5 serialization stable.
    for v in value_list:
        out.append("" if v is None else v)
    return out


def finalize_data_structure(data_structure):
    if isinstance(data_structure, dict):
        out = {}
        for key, value in data_structure.items():
            if key == "__frame_count__":
                continue
            out[key] = finalize_data_structure(value)
        return out
    return _normalize_leaf_sequence(data_structure)


def resolve_rgb_camera_name(data_list):
    observation = data_list.get("observation", {})
    if not isinstance(observation, dict) or len(observation) == 0:
        raise KeyError("observation")

    preferred_names = ["head_camera", "world_camera1", "observer_camera", "third_view"]
    for name in preferred_names:
        camera_data = observation.get(name)
        if isinstance(camera_data, dict) and "rgb" in camera_data:
            return name

    for name, camera_data in observation.items():
        if isinstance(camera_data, dict) and "rgb" in camera_data:
            return name

    raise KeyError("rgb camera")


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            if len(value) == 0:
                continue
            value_np = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value_np)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                if value_np.dtype.kind == "U":
                    _create_string_dataset(hdf5_group, key, value)
                elif value_np.dtype.kind == "O":
                    if all(isinstance(v, (str, bytes, np.str_, np.bytes_)) for v in value):
                        _create_string_dataset(hdf5_group, key, value)
                    else:
                        hdf5_group.create_dataset(key, data=value_np)
                else:
                    hdf5_group.create_dataset(key, data=value_np)
        else:
            if isinstance(value, (str, np.str_)):
                hdf5_group.create_dataset(key, data=str(value), dtype=h5py.string_dtype("utf-8"))
            elif isinstance(value, (bytes, np.bytes_)):
                hdf5_group.create_dataset(key, data=bytes(value))
            else:
                hdf5_group.create_dataset(key, data=value)


def export_camera_sidecar_and_strip_data(
    data_list,
    sidecar_root,
    record_single_camera=False,
    selected_camera_name="world_camera1",
    segmentation_level="actor",
    strip_h5_rgb=True,
    strip_h5_extrinsic=True,
    export_h5_depth_to_sidecar=True,
    export_h5_intrinsic_to_sidecar=True,
    export_h5_cam_pose_to_sidecar=True,
    export_h5_segmentation_to_sidecar=True,
    strip_h5_segmentation=True,
):
    observation = data_list.get("observation")
    if not isinstance(observation, dict):
        return

    seg_level = str(segmentation_level).lower()
    if seg_level not in ("actor", "mesh"):
        seg_level = "actor"

    for camera_name, camera_data in observation.items():
        if not isinstance(camera_data, dict):
            continue

        if record_single_camera and camera_name != selected_camera_name:
            continue

        camera_sidecar_dir = os.path.join(sidecar_root, "observation", camera_name)
        os.makedirs(camera_sidecar_dir, exist_ok=True)

        # 1) Save camera pose outside HDF5; remove extrinsics from HDF5 payload.
        if export_h5_cam_pose_to_sidecar and "cam2world_gl" in camera_data:
            np.save(os.path.join(camera_sidecar_dir, "cam2world_gl.npy"), np.asarray(camera_data["cam2world_gl"]))

        if strip_h5_extrinsic:
            camera_data.pop("cam2world_gl", None)
            camera_data.pop("extrinsic_cv", None)

        # 2) Save depth outside HDF5 payload.
        if export_h5_depth_to_sidecar and "depth" in camera_data:
            np.save(os.path.join(camera_sidecar_dir, "depth.npy"), np.asarray(camera_data["depth"]))
            camera_data.pop("depth", None)

        # 3) Save intrinsic outside HDF5 payload.
        if export_h5_intrinsic_to_sidecar and "intrinsic_cv" in camera_data:
            np.save(os.path.join(camera_sidecar_dir, "intrinsic_cv.npy"), np.asarray(camera_data["intrinsic_cv"]))
            camera_data.pop("intrinsic_cv", None)

        # 4) Keep mp4 video but remove rgb from HDF5 payload.
        if strip_h5_rgb:
            camera_data.pop("rgb", None)

        # 5) Export chosen segmentation and strip segmentation payload.
        if export_h5_segmentation_to_sidecar:
            if seg_level == "mesh":
                if "mesh_segmentation_raw" in camera_data:
                    np.save(os.path.join(camera_sidecar_dir, "seg.npy"), np.asarray(camera_data["mesh_segmentation_raw"]))
            else:
                if "actor_segmentation_raw" in camera_data:
                    np.save(os.path.join(camera_sidecar_dir, "seg.npy"), np.asarray(camera_data["actor_segmentation_raw"]))

        if strip_h5_segmentation:
            for seg_key in [
                "actor_segmentation",
                "actor_segmentation_raw",
                "mesh_segmentation",
                "mesh_segmentation_raw",
            ]:
                camera_data.pop(seg_key, None)


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path,
    video_fps=30.0,
    sidecar_root=None,
    record_single_camera=False,
    selected_camera_name="world_camera1",
    segmentation_level="actor",
    strip_h5_rgb=True,
    strip_h5_extrinsic=True,
    export_h5_depth_to_sidecar=True,
    export_h5_intrinsic_to_sidecar=True,
    export_h5_cam_pose_to_sidecar=True,
    export_h5_segmentation_to_sidecar=True,
    strip_h5_segmentation=True,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    data_list = finalize_data_structure(data_list)

    # images_to_video(np.array(data_list["third_view_rgb"]), out_path=video_path)
    camera_name = resolve_rgb_camera_name(data_list)
    images_to_video(np.array(data_list["observation"][camera_name]["rgb"]), out_path=video_path, fps=float(video_fps))

    if sidecar_root:
        export_camera_sidecar_and_strip_data(
            data_list,
            sidecar_root=sidecar_root,
            record_single_camera=record_single_camera,
            selected_camera_name=selected_camera_name,
            segmentation_level=segmentation_level,
            strip_h5_rgb=strip_h5_rgb,
            strip_h5_extrinsic=strip_h5_extrinsic,
            export_h5_depth_to_sidecar=export_h5_depth_to_sidecar,
            export_h5_intrinsic_to_sidecar=export_h5_intrinsic_to_sidecar,
            export_h5_cam_pose_to_sidecar=export_h5_cam_pose_to_sidecar,
            export_h5_segmentation_to_sidecar=export_h5_segmentation_to_sidecar,
            strip_h5_segmentation=strip_h5_segmentation,
        )

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path,
    video_fps=30.0,
    sidecar_root=None,
    record_single_camera=False,
    selected_camera_name="world_camera1",
    segmentation_level="actor",
    strip_h5_rgb=True,
    strip_h5_extrinsic=True,
    export_h5_depth_to_sidecar=True,
    export_h5_intrinsic_to_sidecar=True,
    export_h5_cam_pose_to_sidecar=True,
    export_h5_segmentation_to_sidecar=True,
    strip_h5_segmentation=True,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path,
        video_fps=video_fps,
        sidecar_root=sidecar_root,
        record_single_camera=record_single_camera,
        selected_camera_name=selected_camera_name,
        segmentation_level=segmentation_level,
        strip_h5_rgb=strip_h5_rgb,
        strip_h5_extrinsic=strip_h5_extrinsic,
        export_h5_depth_to_sidecar=export_h5_depth_to_sidecar,
        export_h5_intrinsic_to_sidecar=export_h5_intrinsic_to_sidecar,
        export_h5_cam_pose_to_sidecar=export_h5_cam_pose_to_sidecar,
        export_h5_segmentation_to_sidecar=export_h5_segmentation_to_sidecar,
        strip_h5_segmentation=strip_h5_segmentation,
    )
