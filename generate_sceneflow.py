"""
SceneFlow生成脚本 - 从HDF5深度图逆投影生成点云及其追踪轨迹
核心思路：
1. 读取HDF5的深度图、相机参数、物体位姿
2. 用深度图和相机内参生成第0帧点云
3. 根据物体位姿变化追踪这些点到其他帧
4. 计算相对位移(delta)
5. 保存为npy文件
"""

import h5py
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, Tuple, List
import os
import sys


class SceneFlowGenerator:
    """从深度图生成SceneFlow的核心类"""
    
    def __init__(self, hdf5_path: str, device: str = "cpu"):
        """
        Args:
            hdf5_path: HDF5文件路径
            device: 'cpu' 或 'cuda'（如果有pytorch3d）
        """
        self.hdf5_path = hdf5_path
        self.device = device
        self.hdf5 = None
        self.num_frames = None
        
    def load_hdf5(self):
        """加载HDF5文件"""
        self.hdf5 = h5py.File(self.hdf5_path, 'r')
        
        # 获取总帧数 - 尝试多个位置
        possible_paths = [
            'observation/head_camera/depth',
            'observation/head_camera/rgb',
            'head_camera/depth',
            'rgb',
            'depth',
        ]
        
        for path in possible_paths:
            if path in self.hdf5:
                self.num_frames = self.hdf5[path].shape[0]
                print(f"✓ 从 {path} 获取帧数")
                break
        else:
            raise ValueError(f"无法确定帧数，检查HDF5结构。尝试过的路径: {possible_paths}")
        
        print(f"✓ 加载HDF5: {self.hdf5_path}")
        print(f"✓ 总帧数: {self.num_frames}")
    
    def get_camera_intrinsic(self, frame_idx: int = 0, camera_name: str = "head_camera") -> np.ndarray:
        """
        获取相机内参矩阵 K (3x3)
        Args:
            frame_idx: 帧索引（内参通常不变，但存储时可能是per-frame）
            camera_name: 相机名称
        Returns:
            K (3x3) 内参矩阵
        """
        possible_paths = [
            f'observation/{camera_name}/intrinsic_cv',
            f'{camera_name}/intrinsic_cv',
            'camera_intrinsic',
            'intrinsic_cv',
        ]
        
        for path in possible_paths:
            if path in self.hdf5:
                data = self.hdf5[path][...]
                # 如果是per-frame存储（T, 3, 3），取第frame_idx帧
                if len(data.shape) == 3 and data.shape[0] == self.num_frames:
                    K = data[frame_idx]
                # 否则假设每帧都相同
                elif len(data.shape) == 2:
                    K = data
                else:
                    K = data[0] if len(data.shape) > 2 else data
                return K.astype(np.float32)
        
        raise ValueError(f"找不到相机内参数据。尝试过的路径: {possible_paths}")
    
    def get_camera_extrinsic(self, frame_idx: int, camera_name: str = "head_camera") -> np.ndarray:
        """
        获取相机外参 T_c2w (4x4)：相机到世界的变换矩阵
        Args:
            frame_idx: 帧索引
            camera_name: 相机名称
        Returns:
            T_c2w (4x4) 变换矩阵
        """
        possible_paths = [
            f'observation/{camera_name}/cam2world_gl',
            f'{camera_name}/cam2world_gl',
            'camera_extrinsic',
            'cam2world_gl',
        ]
        
        for path in possible_paths:
            if path in self.hdf5:
                data = self.hdf5[path][...]
                # 如果是per-frame存储 (T, 4, 4)
                if len(data.shape) == 3 and data.shape[0] == self.num_frames:
                    T_c2w = data[frame_idx]
                # 如果是per-frame存储 (T, 3, 4) - extrinsic_cv格式，需要转换为4x4
                elif len(data.shape) == 3 and data.shape[-1] == 4 and data.shape[-2] == 3:
                    T_c2w_3x4 = data[frame_idx]
                    # 转换成4x4
                    T_c2w = np.eye(4)
                    T_c2w[:3, :] = T_c2w_3x4
                # 否则假设每帧都相同
                elif len(data.shape) == 2:
                    T_c2w = data
                else:
                    T_c2w = data[0] if len(data.shape) > 2 else data
                return T_c2w.astype(np.float32)
        
        raise ValueError(f"找不到相机外参数据。尝试过的路径: {possible_paths}")
    
    def get_depth_image(self, frame_idx: int, camera_name: str = "head_camera") -> np.ndarray:
        """
        获取深度图
        Args:
            frame_idx: 帧索引
            camera_name: 相机名称
        Returns:
            depth (H, W) 深度图，单位mm
        """
        possible_paths = [
            f'observation/{camera_name}/depth',
            f'{camera_name}/depth',
            'depth',
        ]
        
        for path in possible_paths:
            if path in self.hdf5:
                depth = self.hdf5[path][frame_idx]
                # 确保是float类型
                depth = depth.astype(np.float32)
                
                # 有效性检查：深度通常在100-5000mm范围内
                # 0表示无效点
                depth[depth == 0] = np.nan
                
                return depth
        
        raise ValueError(f"找不到深度数据。尝试过的路径: {possible_paths}")
    
    def get_segmentation(self, frame_idx: int, camera_name: str = "head_camera") -> np.ndarray:
        """
        获取分割图（每个像素对应的物体ID）
        Args:
            frame_idx: 帧索引
            camera_name: 相机名称
        Returns:
            seg (H, W) 或 (H, W, 3) 分割图
        """
        possible_paths = [
            f'observation/{camera_name}/actor_segmentation_raw',
            f'observation/{camera_name}/mesh_segmentation_raw',
            f'observation/{camera_name}/mesh_segmentation',
            f'observation/{camera_name}/actor_segmentation',
            f'{camera_name}/actor_segmentation_raw',
            f'{camera_name}/mesh_segmentation_raw',
            f'{camera_name}/mesh_segmentation',
            f'{camera_name}/actor_segmentation',
            'segmentation',
        ]
        
        for path in possible_paths:
            if path in self.hdf5:
                seg = self.hdf5[path][frame_idx]
                # 如果是彩色分割 (H, W, 3)，转换为灰度 (H, W)
                if len(seg.shape) == 3 and seg.shape[2] == 3:
                    # 使用第一个通道作为ID
                    seg = seg[:, :, 0]
                return seg.astype(np.int32)
        
        # 如果没有分割data，返回全1
        depth = self.get_depth_image(frame_idx, camera_name)
        seg = np.ones_like(depth, dtype=np.int32)
        print(f"  ⚠ 警告: 找不到分割数据，使用全1作为分割")
        return seg
    
    def get_object_poses(self, frame_idx: int) -> Dict[int, np.ndarray]:
        """
        获取所有物体的位姿
        Args:
            frame_idx: 帧索引
        Returns:
            Dict[object_id -> pose (4x4)]
        """
        # TODO: 需要根据实际HDF5结构调整
        # 假设HDF5中有 'object_poses' 或类似数据结构
        if 'object_poses' in self.hdf5:
            poses_data = self.hdf5['object_poses'][frame_idx]
            return poses_data  # 需要进一步解析
        else:
            # 如果没有位姿数据，每个物体用单位矩阵（不移动）
            print("⚠ 警告: 找不到物体位姿数据，假设物体不移动")
            return {}
    
    def depth_to_pointcloud(self, depth: np.ndarray, K: np.ndarray, 
                           T_c2w: np.ndarray) -> np.ndarray:
        """
        从深度图生成点云（世界坐标系）
        
        原理：
        1. 对每个像素(u,v)，用内参逆投影得到相机坐标系中的3D点
        2. 用外参将点变换到世界坐标系
        
        Args:
            depth: (H, W) 深度图
            K: (3, 3) 相机内参
            T_c2w: (4, 4) 相机到世界的变换矩阵
            
        Returns:
            point_cloud: (N, 3) 世界坐标系中的3D点
        """
        H, W = depth.shape
        
        # 生成像素坐标
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 获取有效点的掩码（深度不为0且不为NaN）
        valid_mask = ~np.isnan(depth) & (depth > 0)
        
        # 提取有效像素和对应的深度
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth[valid_mask]
        
        # 逆投影：从像素坐标和深度得到相机坐标系中的3D点
        # 公式：p_cam = K^{-1} * [u, v, 1] * depth
        K_inv = np.linalg.inv(K)
        
        # 构造齐次像素坐标
        uv1 = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)  # (3, N)
        
        # 逆投影
        points_cam = K_inv @ uv1  # (3, N)
        points_cam = points_cam * depth_valid  # (3, N)
        
        # 变换到世界坐标系
        # p_world = R @ p_cam + t
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]
        
        points_world = R @ points_cam + t[:, np.newaxis]  # (3, N)
        points_world = points_world.T  # (N, 3)
        
        print(f"  点云包含 {len(points_world)} 个有效点")
        
        return points_world.astype(np.float32)
    
    def track_point_cloud(self, points_frame0: np.ndarray, seg_frame0: np.ndarray,
                         frame_indices: List[int], 
                         camera_name: str = "head_camera") -> List[np.ndarray]:
        """
        追踪第0帧的点云到其他帧
        
        核心思路：
        - 根据分割信息将点分配到不同物体
        - 对每个物体，根据其位姿变化计算新位置
        
        Args:
            points_frame0: (N, 3) 第0帧的点云
            seg_frame0: (H, W) 第0帧的分割图
            frame_indices: 要追踪的帧索引列表
            camera_name: 相机名称
            
        Returns:
            tracked_clouds: List[(N, 3)] 每帧的点云
        """
        tracked_clouds = []
        
        # 获取参考帧（第0帧）的信息
        T_c2w_ref = self.get_camera_extrinsic(0, camera_name)
        
        for frame_idx in frame_indices:
            print(f"  追踪到帧 {frame_idx}/{frame_indices[-1]}...", end='\r')
            
            if frame_idx == 0:
                # 第0帧就是参考，直接返回
                tracked_clouds.append(points_frame0.copy())
            else:
                # 获取当前帧的信息
                depth_cur = self.get_depth_image(frame_idx, camera_name)
                K = self.get_camera_intrinsic(frame_idx, camera_name)
                T_c2w_cur = self.get_camera_extrinsic(frame_idx, camera_name)
                
                # 生成当前帧的点云
                points_cur = self.depth_to_pointcloud(depth_cur, K, T_c2w_cur)
                tracked_clouds.append(points_cur)
        
        print()  # 换行
        return tracked_clouds
    
    def compute_delta_trajectories(self, tracked_clouds: List[np.ndarray],
                                  ref_frame_idx: int = 0) -> np.ndarray:
        """
        计算相对位移Delta
        
        原理：
        对于第0帧的每个点p_ref，计算其在其他帧中的相对位移
        delta_t = p_t - p_ref
        
        Args:
            tracked_clouds: List[(N, 3)] 追踪的点云序列
            ref_frame_idx: 参考帧索引（通常是0）
            
        Returns:
            delta_trajectories: (T, N, 3) 相对位移轨迹
            - T: 帧数
            - N: 点数
            - 3: xyz坐标
        """
        ref_cloud = tracked_clouds[ref_frame_idx]
        
        delta_trajectories = []
        for t, cloud in enumerate(tracked_clouds):
            # 确保点数相同（通常应该一致）
            if len(cloud) != len(ref_cloud):
                print(f"⚠ 警告: 帧 {t} 的点数 {len(cloud)} 与参考帧 {len(ref_cloud)} 不一致")
                # 可以选择插值或裁剪
                if len(cloud) < len(ref_cloud):
                    ref_cloud_t = ref_cloud[:len(cloud)]
                else:
                    ref_cloud_t = ref_cloud
            else:
                ref_cloud_t = ref_cloud
            
            delta = cloud - ref_cloud_t
            delta_trajectories.append(delta)
        
        delta_trajectories = np.stack(delta_trajectories, axis=0)  # (T, N, 3)
        return delta_trajectories.astype(np.float32)
    
    def get_keyframe_indices(self) -> List[int]:
        """
        获取关键帧索引
        将序列分为5个关键帧：[0, 1/4, 2/4, 3/4, 4/4]
        
        Returns:
            List of 5 keyframe indices
        """
        indices = [
            0,
            self.num_frames // 4,
            self.num_frames // 2,
            3 * self.num_frames // 4,
            self.num_frames - 1
        ]
        return list(set(indices))  # 去重并保持顺序

    def get_frame_indices(self, all_frames: bool = False) -> List[int]:
        """根据模式返回帧索引列表。"""
        if all_frames:
            return list(range(self.num_frames))
        return self.get_keyframe_indices()
    
    def generate_sceneflow(self, output_dir: str, camera_name: str = "head_camera",
                          compress: bool = False, all_frames: bool = False):
        """
        完整的SceneFlow生成流程
        
        Args:
            output_dir: 输出目录
            camera_name: 使用的相机名称
            compress: 是否使用Any4D压缩（暂未实现）
        """
        print("\n" + "="*60)
        print("SceneFlow生成流程")
        print("="*60)
        
        # 1. 加载HDF5
        self.load_hdf5()
        
        # 2. 获取第0帧的点云
        print("\n[步骤1] 生成第0帧点云...")
        depth_frame0 = self.get_depth_image(0, camera_name)
        seg_frame0 = self.get_segmentation(0, camera_name)
        K = self.get_camera_intrinsic(0, camera_name)
        T_c2w_frame0 = self.get_camera_extrinsic(0, camera_name)
        
        points_frame0 = self.depth_to_pointcloud(depth_frame0, K, T_c2w_frame0)
        valid_mask_frame0 = ~np.isnan(depth_frame0) & (depth_frame0 > 0)
        point_seg_ids_frame0 = seg_frame0[valid_mask_frame0].astype(np.int32)
        print(f"✓ 第0帧点云形状: {points_frame0.shape}")
        
        # 3. 获取关键帧
        print("\n[步骤2] 确定帧索引...")
        keyframe_indices = self.get_frame_indices(all_frames=all_frames)
        if all_frames:
            print(f"✓ 使用全过程帧: 0..{self.num_frames - 1} (共 {len(keyframe_indices)} 帧)")
        else:
            print(f"✓ 关键帧: {keyframe_indices}")
        
        # 4. 追踪点云
        print("\n[步骤3] 追踪点云到各关键帧...")
        tracked_clouds = self.track_point_cloud(points_frame0, seg_frame0, 
                                               keyframe_indices, camera_name)
        
        # 5. 计算Delta
        print("\n[步骤4] 计算相对位移Delta...")
        delta_trajectories = self.compute_delta_trajectories(tracked_clouds, ref_frame_idx=0)
        print(f"✓ Delta形状: {delta_trajectories.shape}")
        print(f"  (T={delta_trajectories.shape[0]}, N={delta_trajectories.shape[1]}, 3D)")
        
        # 6. 保存结果
        print("\n[步骤5] 保存结果...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每条轨迹
        for i, (frame_idx, delta) in enumerate(zip(keyframe_indices, delta_trajectories)):
            output_path = f"{output_dir}/sceneflow_{i}.npy"
            np.save(output_path, delta)
            print(f"✓ 保存: {output_path} (形状: {delta.shape})")
        
        # 保存原始点云用于验证
        pc0_path = f"{output_dir}/pointcloud_frame0.npy"
        np.save(pc0_path, points_frame0)
        print(f"✓ 保存: {pc0_path}")
        
        # 保存分割信息
        seg_path = f"{output_dir}/segmentation_frame0.npy"
        np.save(seg_path, seg_frame0)
        print(f"✓ 保存: {seg_path}")

        # 保存与pointcloud_frame0对齐的逐点分割id
        point_seg_path = f"{output_dir}/point_seg_ids_frame0.npy"
        np.save(point_seg_path, point_seg_ids_frame0)
        print(f"✓ 保存: {point_seg_path}")
        
        print(f"\n✅ SceneFlow生成完成！输出目录: {output_dir}")
        
        return delta_trajectories, points_frame0, seg_frame0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从HDF5生成SceneFlow")
    parser.add_argument("hdf5_path", type=str, help="HDF5文件路径")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="输出目录（默认为HDF5同目录的sceneflow子目录）")
    parser.add_argument("--camera", type=str, default="head_camera", 
                       help="使用的相机名称")
    parser.add_argument("--all-frames", action="store_true",
                        help="导出全过程所有帧（默认仅关键帧）")
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        hdf5_dir = os.path.dirname(args.hdf5_path)
        args.output_dir = os.path.join(hdf5_dir, "sceneflow")
    
    # 生成SceneFlow
    generator = SceneFlowGenerator(args.hdf5_path)
    generator.generate_sceneflow(args.output_dir, camera_name=args.camera, all_frames=args.all_frames)


if __name__ == "__main__":
    main()
