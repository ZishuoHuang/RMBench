import h5py
import cv2
import numpy as np
import argparse
import imageio
from tqdm import tqdm

def visualize_flow(h5_path, actor_name, camera_name, output_path):
    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Load RGB sequence (T, H, W, 3)
        rgbs = f[f"observation/{camera_name}/rgb"][...]
        
        # Load Object Flow in World coordinates (T, N, 3)
        flow_path = f"observation/objectflow/{actor_name}/world_coordinates"
        if flow_path not in f:
            raise ValueError(f"Could not find {flow_path}. Did the ObjectFlowGenerator run successfully?")
        world_pts = f[flow_path][...]
        
        # Load Camera Matrices
        intrinsics = f[f"observation/{camera_name}/intrinsic_cv"][...]
        extrinsics = f[f"observation/{camera_name}/extrinsic_cv"][...]

        T = len(rgbs)
        
        frames = []
        for i in tqdm(range(T), desc="Rendering Video"):
            # The RGB array from SAPIEN is usually uint8 (0-255)
            img = rgbs[i].copy() 
            
            # Handle if per-frame extrinsics (T, 4, 4) or static (4, 4)
            extrinsic = extrinsics[i] if extrinsics.ndim == 3 else extrinsics
            # Handle if per-frame intrinsics (T, 3, 3) or static (3, 3)
            intrinsic = intrinsics[i] if intrinsics.ndim == 3 else intrinsics

            # 1. World Points (N, 3)
            pts_w = world_pts[i]
            
            # 2. Convert to camera coordinates: P_cam = Extrinsic * P_world
            pts_w_homo = np.hstack([pts_w, np.ones((len(pts_w), 1))]).T # (4, N)
            
            pts_c = (extrinsic @ pts_w_homo).T # (N, 4)
            pts_c = pts_c[:, :3] # Throw away homogenous W
            
            # Filter points behind the camera (Z <= 0)
            valid_mask = pts_c[:, 2] > 0
            pts_c = pts_c[valid_mask]

            # 3. Project to pixels: P_pixel = Intrinsic * P_cam / Z
            pts_c_homo = pts_c.T # (3, M)
            uv1 = (intrinsic @ pts_c_homo).T # (M, 3)
            
            # Pixel coordinates (u, v)
            u = uv1[:, 0] / uv1[:, 2]
            v = uv1[:, 1] / uv1[:, 2]
            
            # 4. Draw on the image
            for pt_u, pt_v in zip(u, v):
                # Only draw if within image boundaries
                if 0 <= int(pt_u) < img.shape[1] and 0 <= int(pt_v) < img.shape[0]:
                    # BGR color for cv2, but imageio assumes RGB. We use green: (0, 255, 0)
                    cv2.circle(img, (int(pt_u), int(pt_v)), radius=1, color=(0, 255, 0), thickness=-1)
                    
            frames.append(img)
            
        print(f"Saving video to {output_path}...")
        # Since RGB is usually 255, imageio will handle it smoothly
        imageio.mimsave(output_path, frames, fps=30)
        print("Visualization saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--actor_name", type=str, required=True, help="Actor name inside HDF5 like '001_bottle'")
    parser.add_argument("--camera", type=str, default="head_camera", help="Which camera view to render")
    parser.add_argument("--output", type=str, default="objectflow_video.mp4", help="Output video path")
    args = parser.parse_args()
    
    visualize_flow(args.h5_path, args.actor_name, args.camera, args.output)
