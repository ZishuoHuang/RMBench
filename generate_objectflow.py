import os
import argparse
import h5py
import torch
import numpy as np
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

class ObjectFlowGenerator:
    def __init__(self, h5_path: str, obj_mesh_path: str, actor_name: str, num_samples: int = 1024):
        """
        Args:
            h5_path: 你的包含SAPIEN数据的 .h5 文件路径
            obj_mesh_path: 被夹取物体的 .obj, .urdf, .dae 或其他3D模型路径
            actor_name: HDF5里被夹物体的actor_name名字 (如 "can", "bottle_0")
            num_samples: Pytorch3D 从表面采样的点数
        """
        self.h5_path = h5_path
        self.obj_mesh_path = obj_mesh_path
        self.actor_name = actor_name
        self.num_samples = num_samples
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sampled_points_local = None # 物体局部坐标系下的点集 (N, 3)
        self.num_frames = 0
        
    def sample_mesh_surface(self):
        """ 1. 提取仿真里被夹物体mesh的结构，使用Pytorch3D采样 """
        # 解析模型的缩放系数 
        import re, json
        from pathlib import Path
        path = Path(self.obj_mesh_path)
        modeldir = path.parent
        match = re.search(r'(?:base|textured)(\d+)\.(?:glb|obj)', path.name)
        model_id = match.group(1) if match else None
        json_file = modeldir / f"model_data{model_id}.json" if model_id is not None else modeldir / "model_data.json"
        
        scale = np.array([1.0, 1.0, 1.0])
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                scale = np.array(data.get('scale', [1.0, 1.0, 1.0]))
                
        print(f"[1] Parsed model scale from {json_file.name}: {scale}")

        # 使用 Trimesh 加载模型 (支持多种格式)
        print(f"[1] Loading mesh from {self.obj_mesh_path}...")
        mesh = trimesh.load(self.obj_mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in mesh.geometry.values()]
            )
            
        print(f"      Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # 乘以 json 中的 scale 进行缩放，与Sapien对齐
        mesh.vertices = mesh.vertices * scale

        # 转为 Pytorch3D Meshes 格式
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
        pt3d_mesh = Meshes(verts=[verts], faces=[faces])
        
        # 使用 Pytorch3D 在表面均匀采样
        print(f"[1] Sampling {self.num_samples} points using PyTorch3D...")
        sampled_points = sample_points_from_meshes(pt3d_mesh, num_samples=self.num_samples)
        
        self.sampled_points_local = sampled_points.squeeze(0).cpu().numpy() # (N, 3)
        return self.sampled_points_local

    def track_object_flow(self, camera_name: str = "head_camera"):
        """ 2. 和sceneflow类似，算出每一时刻采样点的位置信息 """
        print(f"[2] Opening HDF5 file: {self.h5_path}...")
        
        # (num_frames, N, 3) 保存每一帧中采样点的世界坐标
        flow_world_coords = []
        # (num_frames, N, 3) 保存每一帧中采样点的某相机坐标系坐标
        flow_camera_coords = [] 
        
        with h5py.File(self.h5_path, 'r') as f:
            # 检查是否有该 actor 的位姿序列 (通常RoboTwin会存为Nx4x4或时间序)
            poses_path = f"scene_state/rigid_actor_poses/{self.actor_name}"
            if poses_path not in f:
                raise ValueError(f"Actor '{self.actor_name}' not found at {poses_path} in h5 file!")
                
            # 读取所有时间步的World Pose (T, 4, 4) 
            actor_poses = f[poses_path][...]
            self.num_frames = len(actor_poses)
            
            print(f"[2] Found {self.num_frames} frames for {self.actor_name}. Computing tracking trajectories...")
            
            # 将点云扩展为齐次坐标 (N, 4)
            points_homo = np.hstack([self.sampled_points_local, np.ones((self.num_samples, 1))]).T # (4, N)
            
            for i in tqdm(range(self.num_frames), desc="Tracking object"):
                # 获取该帧物体在世界坐标系下的变换矩阵
                T_obj2world = actor_poses[i]  # (4, 4)
                
                # 计算该时刻这些点在世界坐标系的位置
                # P_world = T_obj2world * P_local
                p_world = (T_obj2world @ points_homo).T[:, :3] # 取前三维 (N, 3)
                flow_world_coords.append(p_world)
                
                # 如果还需要转化为相机视角：
                # (先看 h5 里相机外参是集中存的还是怎么存的)
                extrinsic_path = f"observation/{camera_name}/cam2world_gl"
                if extrinsic_path in f:
                    T_cam2world = f[extrinsic_path][i] if len(f[extrinsic_path].shape) == 3 else f[extrinsic_path][...]
                    T_world2cam = np.linalg.inv(T_cam2world)
                    
                    p_world_homo = np.hstack([p_world, np.ones((self.num_samples, 1))]).T
                    p_cam = (T_world2cam @ p_world_homo).T[:, :3]
                    flow_camera_coords.append(p_cam)
                else:
                    # 如果找不到相机外参，相机坐标序列保存全零/不保存即可
                    pass
                    
        return np.array(flow_world_coords), np.array(flow_camera_coords)

    def save_results(self, flow_world, flow_camera):
        """ 3. 和原始数据保存在一起或者后期转换就可以了 """
        print(f"[3] Saving object flow to {self.h5_path}...")
        
        with h5py.File(self.h5_path, 'a') as f:
            # 创建保存 objectflow 的组
            grp_name = f"observation/objectflow/{self.actor_name}"
            
            if grp_name in f:
                del f[grp_name] # 覆盖之前旧的
                
            grp = f.create_group(grp_name)
            
            # 保存局部的结构（可选）
            grp.create_dataset("local_points", data=self.sampled_points_local)
            
            # 保存世界坐标系点流 (T, N, 3) 
            grp.create_dataset("world_coordinates", data=flow_world)
            
            # 保存相机坐标系点流 (T, N, 3) （如果有的话）
            if len(flow_camera) > 0:
                grp.create_dataset("camera_coordinates", data=flow_camera)
                
        print(f"[✔] Successfully saved tracking points of {self.actor_name} to HDF5!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to episode HDF5 file")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to object .obj/.urdf file in assets/")
    parser.add_argument("--actor_name", type=str, required=True, help="Actor name inside HDF5 scene_state (e.g. 'bottle_0')")
    parser.add_argument("--camera", type=str, default="head_camera")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()
    
    flow_gen = ObjectFlowGenerator(args.h5_path, args.mesh_path, args.actor_name, args.num_samples)
    
    # 步骤1：加载模型并用PyTorch3D采样
    flow_gen.sample_mesh_surface()
    
    # 步骤2：结合SAPIEN保存的位姿，计算全过程Flow
    w_coords, c_coords = flow_gen.track_object_flow(camera_name=args.camera)
    
    # 步骤3：保存进原本的HDF5
    flow_gen.save_results(w_coords, c_coords)
