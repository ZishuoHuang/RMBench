"""
SceneFlow可视化脚本 - 用Open3D可视化点云追踪效果
无需SAPIEN，完全独立运行

这个脚本可以帮助你：
1. 可视化第0帧的点云
2. 看点云追踪到其他帧的轨迹
3. 检查是否有跳跃或异常
"""

import numpy as np
from pathlib import Path
import os
import sys

try:
    import open3d as o3d
except ImportError:
    print("❌ 需要安装 open3d: pip install open3d")
    sys.exit(1)


class SceneFlowVisualizer:
    """SceneFlow可视化类"""
    
    def __init__(self, sceneflow_dir: str):
        """
        Args:
            sceneflow_dir: SceneFlow输出目录，包含 sceneflow_*.npy 文件
        """
        self.sceneflow_dir = Path(sceneflow_dir)
        self.pointcloud_frame0 = None
        self.segmentation_frame0 = None
        self.delta_trajectories = []
        self.colors_map = {}
        
    def load_results(self):
        """加载生成的SceneFlow文件"""
        print("加载SceneFlow结果...")
        
        # 加载第0帧点云
        pc0_path = self.sceneflow_dir / "pointcloud_frame0.npy"
        if pc0_path.exists():
            self.pointcloud_frame0 = np.load(pc0_path)
            print(f"✓ 加载点云: {self.pointcloud_frame0.shape}")
        
        # 加载分割图
        seg_path = self.sceneflow_dir / "segmentation_frame0.npy"
        if seg_path.exists():
            self.segmentation_frame0 = np.load(seg_path)
            print(f"✓ 加载分割: {self.segmentation_frame0.shape}")
        
        # 加载Delta轨迹
        sceneflow_files = sorted(self.sceneflow_dir.glob("sceneflow_*.npy"))
        for f in sceneflow_files:
            delta = np.load(f)
            self.delta_trajectories.append(delta)
            print(f"✓ 加载 {f.name}: {delta.shape}")
    
    def create_color_map(self):
        """为不同物体创建颜色映射"""
        unique_ids = np.unique(self.segmentation_frame0)
        
        # 为每个物体ID分配不同的颜色
        colors = [
            [1, 0, 0],      # 红
            [0, 1, 0],      # 绿
            [0, 0, 1],      # 蓝
            [1, 1, 0],      # 黄
            [1, 0, 1],      # 紫
            [0, 1, 1],      # 青
            [1, 0.5, 0],    # 橙
            [0.5, 0, 1],    # 紫蓝
        ]
        
        for obj_id in unique_ids:
            color_idx = int(obj_id) % len(colors)
            self.colors_map[int(obj_id)] = colors[color_idx]
    
    def visualize_frame_with_colors(self, points: np.ndarray, segmentation: np.ndarray = None):
        """
        可视化一帧的点云，按物体着色
        
        Args:
            points: (N, 3) 点坐标
            segmentation: (H, W) 分割图，或 (N,) 点的物体ID
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if segmentation is not None:
            # 如果分割图是2D（H, W），需要重建为点的ID
            # 这里假设分割图已经是点ID的形式
            if len(segmentation.shape) == 1:
                # 1D数组：直接是点的物体ID
                colors = np.array([
                    self.colors_map.get(int(obj_id), [0.5, 0.5, 0.5])
                    for obj_id in segmentation
                ])
            else:
                # 2D数组：需要从像素映射回点（通常不需要，因为点已经展平）
                colors = np.ones((len(points), 3)) * 0.5
        else:
            # 没有分割信息，统一灰色
            colors = np.ones((len(points), 3)) * 0.7
        
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        
        return pcd
    
    def visualize_delta_trajectory(self, point_idx: int = None, max_points: int = 100):
        """
        可视化单个点的追踪轨迹
        
        Args:
            point_idx: 点的索引（如果为None，随机选择）
            max_points: 最多显示的轨迹线段数
        """
        if len(self.delta_trajectories) < 2:
            print("⚠ 需要至少2帧的数据")
            return
        
        if point_idx is None:
            point_idx = np.random.randint(0, len(self.pointcloud_frame0))
        
        print(f"\n可视化点 {point_idx} 的追踪轨迹...")
        
        # 获取这个点在各帧的位置
        ref_pos = self.pointcloud_frame0[point_idx]
        
        # 创建可视化对象
        geometries = []
        
        # 添加参考点（第0帧）
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(ref_pos)
        sphere.paint_uniform_color([1, 0, 0])  # 红色
        geometries.append(sphere)
        
        # 添加轨迹线
        for t, delta in enumerate(self.delta_trajectories):
            if t == 0:
                continue  # 跳过第0帧
            
            pos_t = ref_pos + delta[point_idx]
            
            # 创建从参考点到当前点的线
            points_line = np.array([ref_pos, pos_t])
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(points_line)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            
            # 根据帧数分配颜色
            color = [0, t / len(self.delta_trajectories), 1 - t / len(self.delta_trajectories)]
            line.paint_uniform_color(color)
            geometries.append(line)
            
            # 添加终点
            sphere_t = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere_t.translate(pos_t)
            sphere_t.paint_uniform_color(color)
            geometries.append(sphere_t)
        
        # 可视化
        o3d.visualization.draw_geometries(geometries, window_name=f"Point {point_idx} Trajectory")
    
    def visualize_all_frames(self):
        """
        顺序可视化所有关键帧的点云
        """
        print("\n按顺序可视化所有关键帧...")
        
        reference_pc = self.visualize_frame_with_colors(
            self.pointcloud_frame0, 
            self.segmentation_frame0
        )
        reference_pc.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 第0帧
        print("显示第0帧... (按任意键继续)")
        o3d.visualization.draw_geometries(
            [reference_pc],
            window_name="Frame 0 (Reference)"
        )
        
        # 其他帧
        for t, delta in enumerate(self.delta_trajectories[1:], start=1):
            # 计算当前帧的点云
            points_t = self.pointcloud_frame0 + delta
            
            pcd_t = self.visualize_frame_with_colors(
                points_t, 
                self.segmentation_frame0
            )
            pcd_t.paint_uniform_color([0.5, 0.5, 0.7])
            
            # 同时显示参考点云和当前帧
            print(f"显示第 {t} 帧... (按任意键继续)")
            o3d.visualization.draw_geometries(
                [reference_pc, pcd_t],
                window_name=f"Frame {t} (Red) vs Frame 0 (Blue)"
            )
    
    def compare_consecutive_frames(self):
        """
        可视化相邻两帧的点云对比
        """
        print("\n对比相邻帧...")
        
        for t in range(1, len(self.delta_trajectories)):
            # 上一帧
            points_prev = self.pointcloud_frame0 + self.delta_trajectories[t-1]
            pcd_prev = self.visualize_frame_with_colors(points_prev, self.segmentation_frame0)
            pcd_prev.paint_uniform_color([1, 0, 0])  # 红色
            
            # 当前帧
            points_cur = self.pointcloud_frame0 + self.delta_trajectories[t]
            pcd_cur = self.visualize_frame_with_colors(points_cur, self.segmentation_frame0)
            pcd_cur.paint_uniform_color([0, 0, 1])  # 蓝色
            
            print(f"对比帧 {t-1}(红) 和帧 {t}(蓝)... (按任意键继续)")
            o3d.visualization.draw_geometries(
                [pcd_prev, pcd_cur],
                window_name=f"Frame {t-1} vs Frame {t}"
            )
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("SceneFlow统计信息")
        print("="*60)
        
        print(f"\n点云统计:")
        print(f"  第0帧点数: {len(self.pointcloud_frame0)}")
        print(f"  点的坐标范围: ")
        print(f"    X: [{self.pointcloud_frame0[:, 0].min():.3f}, {self.pointcloud_frame0[:, 0].max():.3f}]")
        print(f"    Y: [{self.pointcloud_frame0[:, 1].min():.3f}, {self.pointcloud_frame0[:, 1].max():.3f}]")
        print(f"    Z: [{self.pointcloud_frame0[:, 2].min():.3f}, {self.pointcloud_frame0[:, 2].max():.3f}]")
        
        print(f"\nDelta统计:")
        for t, delta in enumerate(self.delta_trajectories):
            max_delta = np.max(np.abs(delta))
            mean_delta = np.mean(np.abs(delta))
            print(f"  Frame {t}: max_delta={max_delta:.4f}, mean_delta={mean_delta:.4f}")
        
        print(f"\n分割物体数: {len(np.unique(self.segmentation_frame0))}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化SceneFlow")
    parser.add_argument("sceneflow_dir", type=str, help="SceneFlow输出目录")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "consecutive", "trajectory", "stats"],
                       help="可视化模式")
    parser.add_argument("--point_idx", type=int, default=None,
                       help="追踪轨迹模式下的点索引")
    
    args = parser.parse_args()
    
    # 创建可视化器
    viz = SceneFlowVisualizer(args.sceneflow_dir)
    viz.load_results()
    viz.create_color_map()
    
    # 打印统计信息
    viz.print_statistics()
    
    # 根据模式执行
    if args.mode == "all":
        viz.visualize_all_frames()
    elif args.mode == "consecutive":
        viz.compare_consecutive_frames()
    elif args.mode == "trajectory":
        viz.visualize_delta_trajectory(point_idx=args.point_idx)
    elif args.mode == "stats":
        pass  # 已经打印过了
    
    print("\n✅ 可视化完成")


if __name__ == "__main__":
    main()
