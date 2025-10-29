# realtime_planes_fast.py
# RED = walls, YELLOW = floor, GREEN = objects (darker when farther)
import time, math, random
import numpy as np
import cv2
import pyzed.sl as sl

# -------- Tunables (start here) --------
CAM_RES          = sl.RESOLUTION.VGA      # try sl.RESOLUTION.VGA for more FPS
DEPTH_MODE       = sl.DEPTH_MODE.NEURAL_LIGHT
CONF_TH          = 80

Z_MIN, Z_MAX     = 0.25, 6.0                # valid depth band [m]
NEAR_FOR_OBJ     = 4.5                      # only consider objects closer than this
FLOOR_TH         = 0.03                     # dist to floor plane for inliers [m]
WALL_TH          = 0.03                     # dist to wall planes [m]
FLOOR_ALIGN_DEG  = 25                       # plane normal within ± this of "up"
WALL_ALIGN_DEG   = 20                       # |angle(up) - 90°| for walls
MAX_WALLS        = 2                        # number of walls to keep

PROC_W           = 160                      # working width (lower = faster)
REFIT_SEC        = 0.7                      # how often to re-run RANSAC (s)

ALPHA_FLOOR, ALPHA_WALL, ALPHA_OBJ = 0.40, 0.35, 0.60

# -------- Helpers --------
def up_vector_cam():  # ZED coords: X right, Y down, Z forward ? "up" � -Y
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)

def precompute_dirs(W, H, fx, fy, cx, cy):
    """Backprojection directions for a depth image with intrinsics (scaled)."""
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) / fx
    Y = (vv - cy) / fy
    Z = np.ones_like(X, dtype=np.float32)
    return np.stack([X, Y, Z], axis=2)      # HxWx3

def build_points(depth_s, dirs):
    return depth_s[..., None] * dirs        # HxWx3

def angle_to_up(n, up):
    # n: (...,3), up: (3,)
    num = np.clip((n * up).sum(axis=-1) / (np.linalg.norm(n, axis=-1) * (np.linalg.norm(up)+1e-9) + 1e-9), -1, 1)
    return np.degrees(np.arccos(num))

def ransac_plane(P, mask, want='floor', up=None, dist_th=0.03, iters=180, subsample=4000):
    """
    P: HxWx3 points, mask: HxW bool valid
    want: 'floor' (parallel to up) or 'wall' (perpendicular to up)
    Returns (n, d, inlier_mask) for plane n·x + d = 0  (n normalized)
    """
    idx = np.argwhere(mask)
    if idx.shape[0] < 200:
        return None, None, np.zeros_like(mask)

    choose = np.random.choice(idx.shape[0], size=min(subsample, idx.shape[0]), replace=False)
    sub = idx[choose]
    P_sub = P[sub[:,0], sub[:,1]]  # Nx3

    best = (0, None, None, None)
    for _ in range(iters):
        i = np.random.choice(idx.shape[0], size=3, replace=False)
        p0, p1, p2 = P[idx[i[0],0], idx[i[0],1]], P[idx[i[1],0], idx[i[1],1]], P[idx[i[2],0], idx[i[2],1]]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-6: 
            continue
        n = n / nn
        if up is not None:
            a = float(angle_to_up(n[None,:], up)[0])
            if want == 'floor':
                if a > FLOOR_ALIGN_DEG: 
                    continue
            else:  # wall
                if abs(a - 90.0) > WALL_ALIGN_DEG:
                    continue
        d = -float(np.dot(n, p0))
        # count inliers using subsample first (fast)
        dist = np.abs(P_sub @ n + d)
        count = int((dist <= dist_th).sum())
        if count > best[0]:
            best = (count, n, d, None)

    n, d = best[1], best[2]
    if n is None:
        return None, None, np.zeros_like(mask)
    # refine inliers on full mask once
    Pf = P.reshape(-1,3)
    mflat = mask.reshape(-1)
    dist_all = np.abs(Pf @ n + d)
    inliers = (mflat) & (dist_all <= dist_th)
    return n, d, inliers.reshape(mask.shape)

def overlay_colors(base_bgr, floor, wall, obj, depth, near=0.35, far=NEAR_FOR_OBJ):
    disp = base_bgr.copy()
    if floor.any():
        over = disp.copy(); over[floor] = (0,255,255)
        disp = cv2.addWeighted(over, ALPHA_FLOOR, disp, 1-ALPHA_FLOOR, 0)
    if wall.any():
        over = disp.copy(); over[wall] = (0,0,255)
        disp = cv2.addWeighted(over, ALPHA_WALL, disp, 1-ALPHA_WALL, 0)
    if obj.any():
        yy, xx = np.where(obj)
        green = np.zeros_like(disp, np.uint8)
        d = depth[yy, xx]
        inten = np.clip((far - d) / (far - near), 0.0, 1.0)
        g = (inten * 255.0).astype(np.uint8)
        green[yy, xx, 1] = g
        disp = cv2.addWeighted(green, ALPHA_OBJ, disp, 1-ALPHA_OBJ, 0)
    return disp

# -------- ZED init --------
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution  = CAM_RES
init.depth_mode         = DEPTH_MODE
init.coordinate_units   = sl.UNIT.METER
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    raise SystemExit(f"ZED open failed: {err}")

runtime = sl.RuntimeParameters()
runtime.confidence_threshold = CONF_TH

cam_info = zed.get_camera_information()
W_full = cam_info.camera_configuration.resolution.width
H_full = cam_info.camera_configuration.resolution.height
fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx
cy = cam_info.camera_configuration.calibration_parameters.left_cam.cy

# Scale intrinsics to PROC size
proc_h = int(PROC_W * H_full / W_full)
sx = PROC_W / W_full
sy = proc_h / H_full
fx_p, fy_p, cx_p, cy_p = fx*sx, fy*sy, cx*sx, cy*sy
dirs = precompute_dirs(PROC_W, proc_h, fx_p, fy_p, cx_p, cy_p)  # HxWx3

m_depth = sl.Mat()
m_left  = sl.Mat()

up = up_vector_cam()
last_refit = 0
planes = {"floor": None, "walls": []}  # (n, d)

ema_fps = None
t_prev = time.time()

print("Running. ESC to quit.")
while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        continue
    zed.retrieve_measure(m_depth, sl.MEASURE.DEPTH)  # float32 meters
    zed.retrieve_image(m_left, sl.VIEW.LEFT)

    depth = m_depth.get_data().copy().astype(np.float32)
    left  = m_left.get_data().copy()[:, :, :3]
    H, W  = depth.shape

    # Downsample
    depth_s = cv2.resize(depth, (PROC_W, proc_h), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth_s) & (depth_s > Z_MIN) & (depth_s < Z_MAX)

    # Points (camera frame)
    P = build_points(depth_s, dirs)

    # Occasional plane re-fit (cheap at PROC size)
    now = time.time()
    need_refit = (now - last_refit) >= REFIT_SEC
    if need_refit:
        # FLOOR: sample bottom 45% to bias
        floor_roi = np.zeros_like(valid); floor_roi[int(proc_h*0.55):, :] = True
        nF, dF, inF = ransac_plane(P, valid & floor_roi, want='floor', up=up,
                                   dist_th=FLOOR_TH, iters=160, subsample=3500)
        if nF is not None:
            planes["floor"] = (nF.astype(np.float32), float(dF))
        # WALLS: mask out floor inliers, then find up to MAX_WALLS
        walls = []
        remaining = valid & (~inF if nF is not None else valid)
        for _ in range(MAX_WALLS):
            if remaining.sum() < 400: break
            nW, dW, inW = ransac_plane(P, remaining, want='wall', up=up,
                                       dist_th=WALL_TH, iters=140, subsample=3000)
            if nW is None: break
            walls.append((nW.astype(np.float32), float(dW)))
            remaining &= (~inW)
        planes["walls"] = walls
        last_refit = now

    # Per-frame classification = distance to stored planes (fast)
    floor_full = np.zeros((H, W), bool)
    wall_full  = np.zeros((H, W), bool)
    obj_full   = np.zeros((H, W), bool)

    # Work at PROC size first
    floor_s = np.zeros_like(valid)
    if planes["floor"] is not None:
        nF, dF = planes["floor"]
        distF = np.abs((P @ nF) + dF)
        floor_s = valid & (distF <= FLOOR_TH)

    wall_s = np.zeros_like(valid)
    for (nW, dW) in planes["walls"]:
        distW = np.abs((P @ nW) + dW)
        wall_s |= (valid & ~floor_s & (distW <= WALL_TH))

    obj_s = valid & ~floor_s & ~wall_s & (depth_s <= NEAR_FOR_OBJ)
    # de-speckle objects a bit
    obj_u8 = (obj_s.astype(np.uint8) * 255)
    obj_u8 = cv2.morphologyEx(obj_u8, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    obj_s  = obj_u8.astype(bool)

    # Upsample masks back to full resolution for display only
    floor_full = cv2.resize(floor_s.astype(np.uint8), (W, H), cv2.INTER_NEAREST).astype(bool)
    wall_full  = cv2.resize(wall_s.astype(np.uint8),  (W, H), cv2.INTER_NEAREST).astype(bool)
    obj_full   = cv2.resize(obj_s.astype(np.uint8),   (W, H), cv2.INTER_NEAREST).astype(bool)

    disp = overlay_colors(left, floor_full, wall_full, obj_full, depth)

    # FPS HUD
    t = time.time()
    fps = 1.0 / max(1e-6, (t - t_prev)); t_prev = t
    ema_fps = fps if ema_fps is None else (0.9*ema_fps + 0.1*fps)
    cv2.putText(disp, f"FPS: {ema_fps:4.1f}  planes: floor={'Y' if planes['floor'] else 'N'}, walls={len(planes['walls'])}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Realtime plane tracking (fast)", disp)
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

cv2.destroyAllWindows()
zed.close()
