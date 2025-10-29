import os, time
import numpy as np
import cv2
import pyzed.sl as sl

def ensure_dir(path="zed_out"):
    os.makedirs(path, exist_ok=True)
    return path

def save_left_png(img_mat, stamp_ms, outdir):
    # img_mat is sl.Mat U8_C4 (BGRA)
    bgra = img_mat.get_data()
    bgr  = bgra[:, :, :3]                 # drop alpha
    cv2.imwrite(os.path.join(outdir, f"left_{stamp_ms}.png"), bgr)

def save_depth_pngs(depth_mat, stamp_ms, outdir):
    # depth_mat is float32 in MILLIMETER (per your init)
    d = depth_mat.get_data().astype(np.float32)   # shape (H,W)
    # Replace inf/NaN with 0
    valid = np.isfinite(d) & (d > 0)
    d_u16 = np.zeros_like(d, dtype=np.uint16)
    d_u16[valid] = np.clip(d[valid], 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(outdir, f"depth_mm_{stamp_ms}.png"), d_u16)

    # Colorized depth for quick viewing (auto range each frame)
    if np.any(valid):
        d_vis = d.copy()
        d_vis[~valid] = 0
        norm = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX)
        color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    else:
        color = np.zeros((*d.shape, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(outdir, f"depth_color_{stamp_ms}.png"), color[:, :, ::-1])  # BGR

def try_save_pointcloud_ply(pc_mat, left_img_mat, stamp_ms, outdir, ds=3):
    # Optional: needs open3d installed
    try:
        import open3d as o3d
    except ImportError:
        return  # skip if Open3D not available

    pc = pc_mat.get_data()                 # (H,W,4) float32: X,Y,Z,RGBA (ZED)
    img = left_img_mat.get_data()[:, :, :3]# (H,W,3) uint8 BGR

    # Downsample both the same way to keep them aligned
    pc  = pc[::ds, ::ds, :]
    img = img[::ds, ::ds, :]

    xyz = pc[..., :3].reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size == 0:
        return

    # Use the left image for color (avoids packed RGBA aliasing)
    bgr = img.reshape(-1, 3)[finite].astype(np.float32) / 255.0
    rgb = bgr[:, ::-1]

    # Build and write PLY
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(outdir, f"cloud_{stamp_ms}.ply"), pcd, write_ascii=False)

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    runtime = sl.RuntimeParameters()

    outdir = ensure_dir("zed_out")
    image = sl.Mat()                 # U8_C4 by default (BGRA)
    depth = sl.Mat()                 # F32 depth
    point_cloud = sl.Mat()           # F32_C4 XYZRGBA

    for i in range(10):              # save 10 frames; adjust as needed
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.002); continue

        # Timestamp for filenames (monotonic in camera domain)
        ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

        # Retrieve aligned outputs
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

        save_left_png(image, ts_ms, outdir)
        save_depth_pngs(depth, ts_ms, outdir)
        try_save_pointcloud_ply(point_cloud, image, ts_ms, outdir, ds=3)

        print(f"Saved frame {i+1} → left/depth/ply @ zed_out/*_{ts_ms}.*")

    zed.close()

if __name__ == "__main__":
    main()
