# depth_plane_segment.py
# Colors: RED = walls, YELLOW = floor, GREEN = objects (darker = farther)
import math, random, time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl

# ---------- Tunables ----------
CAM_RES          = sl.RESOLUTION.VGA
DEPTH_MODE       = sl.DEPTH_MODE.NEURAL_LIGHT
CONF_TH          = 80

Z_MIN, Z_MAX     = 0.25, 6.0      # valid depth band [m]
PROC_W           = 320            # downsample width for RANSAC/seg
FLOOR_DTH        = 0.03           # plane inlier dist threshold [m]
WALL_DTH         = 0.03
FLOOR_ALIGN_DEG  = 25             # floor normal within ± this to camera "up"
WALL_ALIGN_DEG   = 20             # wall normals ~ perpendicular to "up"
MAX_WALLS        = 2              # try to find up to N wall planes
FLOOR_ROI_FRAC   = 0.45           # bias RANSAC sampling to bottom x% of image

NEAR_FOR_OBJ     = 4.5            # cap obj mask to nearer than this [m]
OBJ_MIN_PIX_FRAC = 0.0005         # remove tiny specks

# Visualization opacities
ALPHA_FLOOR = 0.40
ALPHA_WALL  = 0.35
ALPHA_OBJ   = 0.60

# ---------- Small helpers ----------
def deg2rad(d): return d * math.pi / 180.0

def ransac_plane(pts, valid_mask, dist_th, iters=400, sample_mask=None):
    """
    pts: HxWx3 float32 (NaN for invalid)
    valid_mask: HxW bool
    sample_mask: optional HxW bool to bias sampling
    Returns (best_inliers_mask HxW bool, normal[3], d)
    Plane eq: n.x + d = 0  (n normalized)
    """
    H, W, _ = pts.shape
    idx = np.argwhere(valid_mask if sample_mask is None else (valid_mask & sample_mask))
    if idx.shape[0] < 100:  # not enough points
        return np.zeros((H, W), bool), None, None

    flat_pts = pts.reshape(-1, 3)
    best_inliers = np.zeros((H, W), bool)
    best_count = 0
    n_best, d_best = None, None

    for _ in range(iters):
        # random 3 points, ensure they exist and not collinear
        i0 = idx[random.randrange(idx.shape[0])]
        i1 = idx[random.randrange(idx.shape[0])]
        i2 = idx[random.randrange(idx.shape[0])]
        p0 = pts[i0[0], i0[1]]
        p1 = pts[i1[0], i1[1]]
        p2 = pts[i2[0], i2[1]]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        v1, v2 = (p1 - p0), (p2 - p0)
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -np.dot(n, p0)

        # distances for valid pixels only (vectorized)
        vmask_flat = valid_mask.reshape(-1)
        dists = np.abs(np.dot(flat_pts, n) + d)
        inliers_flat = (vmask_flat) & np.isfinite(dists) & (dists <= dist_th)
        count = int(np.count_nonzero(inliers_flat))
        if count > best_count:
            best_count = count
            inliers = inliers_flat.reshape(H, W)
            best_inliers = inliers
            n_best, d_best = n, d

    return best_inliers, n_best, d_best

def up_vector_cam():
    """
    Approximate camera 'up' in the ZED camera frame.
    ZED camera coords (SDK): X right, Y down, Z forward.
    So up � -Y.
    """
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)

def angle_between(n1, n2):
    c = float(np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-9), -1, 1))
    return math.degrees(math.acos(c))

def make_colored_overlay(base_bgr, floor_mask, wall_mask, obj_mask, depth_m,
                         near=0.35, far=NEAR_FOR_OBJ):
    disp = base_bgr.copy()

    # Floor = yellow
    overlay = disp.copy()
    overlay[floor_mask] = (0, 255, 255)
    disp = cv2.addWeighted(overlay, ALPHA_FLOOR, disp, 1-ALPHA_FLOOR, 0)

    # Walls = red
    overlay = disp.copy()
    overlay[wall_mask] = (0, 0, 255)
    disp = cv2.addWeighted(overlay, ALPHA_WALL, disp, 1-ALPHA_WALL, 0)

    # Objects = green, darker when farther
    obj_idx = np.where(obj_mask)
    green = np.zeros_like(disp, dtype=np.uint8)
    d = depth_m[obj_idx]
    # intensity 0..1
    inten = np.clip((far - d) / (far - near), 0.0, 1.0)
    gvals = (inten * 255.0).astype(np.uint8)
    green[obj_idx[0], obj_idx[1], 1] = gvals
    disp = cv2.addWeighted(green, ALPHA_OBJ, disp, 1-ALPHA_OBJ, 0)

    return disp

# ---------- ZED init ----------
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = CAM_RES
init.depth_mode = DEPTH_MODE
init.coordinate_units = sl.UNIT.METER
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    raise SystemExit("Failed to open ZED")

runtime = sl.RuntimeParameters()
runtime.confidence_threshold = CONF_TH

mat_depth = sl.Mat()
mat_xyz   = sl.Mat()
mat_left  = sl.Mat()

print("Running. ESC to quit.")
while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        continue

    zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)    # float32 meters, NaN for invalid
    zed.retrieve_measure(mat_xyz,   sl.MEASURE.XYZ)      # float32 XYZ in meters (NaN invalid)
    zed.retrieve_image(mat_left,    sl.VIEW.LEFT)        # for a nice background

    depth = mat_depth.get_data().copy().astype(np.float32)
    xyz   = mat_xyz.get_data().copy().astype(np.float32)[:, :, :3]
    left  = mat_left.get_data().copy()[:, :, :3]         # BGR
    H, W  = depth.shape

    # Downsample for fast RANSAC
    proc_h = int(PROC_W * H / W)
    depth_s = cv2.resize(depth, (PROC_W, proc_h), interpolation=cv2.INTER_NEAREST)
    xyz_s   = cv2.resize(xyz,   (PROC_W, proc_h), interpolation=cv2.INTER_NEAREST)

    # Valid mask
    valid = np.isfinite(depth_s) & (depth_s > Z_MIN) & (depth_s < Z_MAX)

    # ----- Find FLOOR plane -----
    # Bias sampling to the bottom of the image to lock onto the floor quickly
    floor_roi = np.zeros_like(valid)
    y0 = int(proc_h * (1.0 - FLOOR_ROI_FRAC))
    floor_roi[y0:, :] = True

    floor_inliers, n_floor, d_floor = ransac_plane(
        xyz_s, valid, dist_th=FLOOR_DTH, iters=500, sample_mask=floor_roi
    )

    # Require alignment with camera up
    floor_mask = np.zeros_like(valid)
    if n_floor is not None:
        if angle_between(n_floor, up_vector_cam()) <= FLOOR_ALIGN_DEG:
            floor_mask = floor_inliers

    # ----- Find WALL planes (remove floor first) -----
    remaining = valid & (~floor_mask)
    all_wall_mask = np.zeros_like(valid)
    for k in range(MAX_WALLS):
        if np.count_nonzero(remaining) < 500:
            break
        wall_inliers, n_wall, d_wall = ransac_plane(
            xyz_s, remaining, dist_th=WALL_DTH, iters=400
        )
        if n_wall is None:
            break

        # Wall � perpendicular to "up" (i.e., |angle to up - 90°| <= WALL_ALIGN_DEG)
        a = angle_between(n_wall, up_vector_cam())
        if abs(a - 90.0) <= WALL_ALIGN_DEG:
            all_wall_mask |= wall_inliers
            remaining &= (~wall_inliers)
        else:
            # Not vertical enough�discard this plane
            remaining &= (~wall_inliers)

    # ----- Objects = near, valid, not on floor/walls -----
    objs = valid & (depth_s <= NEAR_FOR_OBJ) & (~floor_mask) & (~all_wall_mask)

    # Remove tiny specks at PROC scale
    objs_u8 = (objs.astype(np.uint8) * 255)
    objs_u8 = cv2.morphologyEx(objs_u8, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    objs = objs_u8.astype(bool)

    # Upsample masks back to full res
    floor_full = cv2.resize(floor_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    wall_full  = cv2.resize(all_wall_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    obj_full   = cv2.resize(objs.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Compose colored view over the left image (looks nicer than raw depth)
    disp = make_colored_overlay(left, floor_full, wall_full, obj_full, depth)

    # HUD
    det3d = int(np.any(obj_full))
    walls = int(np.max(wall_full))  # 1 if any wall pixels
    txt1 = f"Detections: {det3d} | Floor px: {np.count_nonzero(floor_full)} | Wall px: {np.count_nonzero(wall_full)}"
    cv2.putText(disp, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(disp, "WALL: vertical planes   FLOOR: horizontal plane   OBJECT: residual near geometry",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)

    cv2.imshow("ZED depth plane segmentation", disp)
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

cv2.destroyAllWindows()
zed.close()
