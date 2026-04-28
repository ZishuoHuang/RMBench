import os
import cv2
import numpy as np
from pathlib import Path

def render_2d_flow_video(episode_dir, out_mp4, ref_idx="00000"):
    episode_dir = Path(episode_dir)
    delta_file = episode_dir / f"scene_point_delta_ref{ref_idx}.npy"
    anchor_file = episode_dir / f"scene_point_flow_ref{ref_idx}.anchor.npy"
    seg_file = episode_dir / f"segmentation_ref{ref_idx}.npy"
    
    print(f"Loading {delta_file.name}...")
    delta = np.load(delta_file, mmap_mode="r")
    anchor = np.load(anchor_file, mmap_mode="r")
    seg = np.load(seg_file)
    
    # flow is (T, H, W, 3)
    # delta is (T, H, W, 3), anchor is (H, W, 3)
    T, H, W, C = delta.shape
    
    # We only care about objects, not background (seg > 0)
    # Background might be 0 or 255 depending on how it's saved. Typically 0 is background.
    obj_mask = (seg > 0) & (np.linalg.norm(anchor, axis=-1) > 0)
    
    points_h, points_w = np.where(obj_mask)
    num_points = len(points_h)
    print(f"Valid object points to render: {num_points}")
    
    # Assign random colors per object ID
    unique_ids = np.unique(seg[obj_mask])
    colors = {}
    np.random.seed(42)
    for uid in unique_ids:
        colors[uid] = np.random.randint(50, 255, size=3).tolist()
    
    point_colors = np.zeros((num_points, 3), dtype=np.uint8)
    for i in range(num_points):
        uid = seg[points_h[i], points_w[i]]
        point_colors[i] = colors[uid]
        
    out = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
    
    print(f"Rendering {T} frames...")
    
    # dynamically solve for camera projection from frame 0
    Z_0 = anchor[points_h, points_w, 2].astype(np.float32)
    valid_z = Z_0 < -0.01
    if valid_z.sum() > 10:
        X_0 = anchor[points_h, points_w, 0][valid_z].astype(np.float32)
        Z_neg = -Z_0[valid_z]
        A_u = np.vstack([X_0 / Z_neg, np.ones_like(X_0)]).T
        fx, cx = np.linalg.lstsq(A_u, points_w[valid_z], rcond=None)[0]
        
        Y_0 = anchor[points_h, points_w, 1][valid_z].astype(np.float32)
        A_v = np.vstack([Y_0 / Z_neg, np.ones_like(Y_0)]).T
        fy_neg, cy = np.linalg.lstsq(A_v, points_h[valid_z], rcond=None)[0]
    else:
        fx, cx = 515.0, W/2
        fy_neg, cy = -515.0, H/2
    print(f"Estimated camera projection: fx={fx:.2f}, cx={cx:.2f}, fy_neg={fy_neg:.2f}, cy={cy:.2f}")

    for t in range(T):
        # Frame canvas
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Get 3D coordinates of tracked points at frame t
        pos_3d = anchor[points_h, points_w] + delta[t, points_h, points_w]
        
        Z_neg_t = -pos_3d[:, 2]
        # strictly avoid div by zero
        Z_neg_t[Z_neg_t == 0] = 1e-5
        
        # Project 3D back to 2D pixel coordinates exactly like the original camera
        canvas_x = ((pos_3d[:, 0] / Z_neg_t) * fx + cx).astype(np.int32)
        canvas_y = ((pos_3d[:, 1] / Z_neg_t) * fy_neg + cy).astype(np.int32)
        
        # Keep only points that project inside the camera frame
        valid_cam = (canvas_x >= 0) & (canvas_x < W) & (canvas_y >= 0) & (canvas_y < H) & (Z_neg_t > 0)
        
        cx_valid = canvas_x[valid_cam]
        cy_valid = canvas_y[valid_cam]
        color_valid = point_colors[valid_cam]
        
        # Assign colors to canvas
        canvas[cy_valid, cx_valid] = color_valid
        
        out.write(canvas)
        
    out.release()
    
    # Convert the video to standard H.264 format using ffmpeg so it's widely supported (browsers/VS Code)
    import subprocess
    print("Converting video to H.264 format for better viewer compatibility...")
    tmp_mp4 = str(out_mp4) + ".tmp.mp4"
    os.rename(out_mp4, tmp_mp4)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", 
        "-i", tmp_mp4, 
        "-vcodec", "libx264", 
        "-pix_fmt", "yuv420p", 
        str(out_mp4)
    ], check=True)
    os.remove(tmp_mp4)
    
    print(f"Video saved to {out_mp4}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ep_dir = sys.argv[1]
    else:
        ep_dir = "data/turn_switch/turn_switch_worldcamera1_randomized_10/sceneflow_offline_depth_world_camera1/episode0"
        
    ref_idx = "00000"
    if len(sys.argv) > 2:
        ref_idx = sys.argv[2]
        
    out_name = os.path.join(ep_dir, f"sceneflow_2D_viz_ref{ref_idx}.mp4")
    render_2d_flow_video(ep_dir, out_name, ref_idx=ref_idx)

