# realtime_planes_fast_controls_sticky_colors.py
# RED = walls (default), YELLOW = floor (default), GREEN = near objects (darker when farther)

import time, math, json, os
import numpy as np
import cv2
import pyzed.sl as sl

# ======= Fixed tunables you probably don't want to live-tune =======
CAM_RES    = sl.RESOLUTION.VGA
DEPTH_MODE = sl.DEPTH_MODE.NEURAL_LIGHT
PROC_W     = 160

# ======= Persistence =======
CFG_PATH = os.path.join(os.path.dirname(__file__), "planes_params.json")

DEFAULTS = {
    # Depth band [m] (entered as cm in UI)
    "Z_MIN_cm": 25,    # 0.25 m
    "Z_MAX_cm": 508,   # 5.08 m
    # Objects
    "NEAR_FOR_OBJ_cm": 25,  # 0.25 m (near-only objects)
    # Plane inlier thresholds [mm]
    "FLOOR_TH_mm": 48,
    "WALL_TH_mm": 200,
    # Angle tolerances [deg]
    "FLOOR_ALIGN_DEG": 25,
    "WALL_ALIGN_DEG": 20,
    # Walls / confidence / alphas
    "MAX_WALLS": 4,
    "CONF_TH": 46,
    "ALPHA_FLOOR_pct": 40,
    "ALPHA_WALL_pct": 34,
    "ALPHA_OBJ_pct": 60,
    # Refit cadence
    "REFIT_ms": 502,
    # Sticky & decay
    "STICKY_ON": 1,
    "DECAY_ms": 8000,     # 0 = never forget; >0 = exponential decay
    "STICKY_THR": 64,     # display threshold for sticky maps (0..255)
    "DILATE_px": 1,

    # ---- Live color pickers (BGR) ----
    # Default floor = yellow (0,255,255), wall = red (0,0,255)
    "FLOOR_B": 0,  "FLOOR_G": 255, "FLOOR_R": 255,
    "WALL_B":  0,  "WALL_G":  0,   "WALL_R":  255,
}

def load_params(path=CFG_PATH):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        out = DEFAULTS.copy()
        # keep only known keys; coerce to int for sliders
        for k, v in data.items():
            if k in out:
                out[k] = int(v)
        return out
    except Exception:
        return DEFAULTS.copy()

def save_params(values, path=CFG_PATH):
    try:
        with open(path, "w") as f:
            json.dump(values, f, indent=2)
        return True
    except Exception:
        return False

# ======= Control Panel (trackbars) =======
class Controls:
    def __init__(self, win="Params", initial=None):
        self.win = win
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        self._mins = {}
        vals = (initial or DEFAULTS)

        # Depth band [m]
        self._add("Z_MIN_cm",          vals["Z_MIN_cm"],          5, 800)
        self._add("Z_MAX_cm",          vals["Z_MAX_cm"],          5, 800)
        # Objects
        self._add("NEAR_FOR_OBJ_cm",   vals["NEAR_FOR_OBJ_cm"],  25, 800)

        # Plane inlier distances [mm]
        self._add("FLOOR_TH_mm",       vals["FLOOR_TH_mm"],       5, 400)
        self._add("WALL_TH_mm",        vals["WALL_TH_mm"],        5, 400)

        # Angle tolerances [deg]
        self._add("FLOOR_ALIGN_DEG",   vals["FLOOR_ALIGN_DEG"],   5, 45)
        self._add("WALL_ALIGN_DEG",    vals["WALL_ALIGN_DEG"],    5, 45)

        # Max walls
        self._add("MAX_WALLS",         vals["MAX_WALLS"],         0, 4)

        # ZED depth confidence threshold (0�100)
        self._add("CONF_TH",           vals["CONF_TH"],           0, 100)

        # Overlay alphas (0�100 ? 0�1.0)
        self._add("ALPHA_FLOOR_pct",   vals["ALPHA_FLOOR_pct"],   0, 100)
        self._add("ALPHA_WALL_pct",    vals["ALPHA_WALL_pct"],    0, 100)
        self._add("ALPHA_OBJ_pct",     vals["ALPHA_OBJ_pct"],     0, 100)

        # Refit cadence (ms)
        self._add("REFIT_ms",          vals["REFIT_ms"],        100, 2000)

        # ---- Sticky/decay controls ----
        self._add("STICKY_ON",         vals["STICKY_ON"],          0, 1)
        # allow exact 0 for "never forget": don't clamp to min here
        cv2.createTrackbar("DECAY_ms", self.win, int(vals["DECAY_ms"]), 30000, lambda *_: None)
        self._mins["DECAY_ms"] = 0
        self._add("STICKY_THR",        vals["STICKY_THR"],         1, 255)
        self._add("DILATE_px",         vals["DILATE_px"],          0, 5)

        # ---- Color sliders (BGR) ----
        self._add("FLOOR_B", vals["FLOOR_B"], 0, 255)
        self._add("FLOOR_G", vals["FLOOR_G"], 0, 255)
        self._add("FLOOR_R", vals["FLOOR_R"], 0, 255)

        self._add("WALL_B",  vals["WALL_B"],  0, 255)
        self._add("WALL_G",  vals["WALL_G"],  0, 255)
        self._add("WALL_R",  vals["WALL_R"],  0, 255)

    def _add(self, name, val, lo, hi):
        cv2.createTrackbar(name, self.win, int(val), int(hi), lambda *_: None)
        self._mins[name] = lo
        if val < lo:
            cv2.setTrackbarPos(name, self.win, int(lo))

    def read_raw(self):
        # Return raw integer slider positions for saving
        raw = {}
        for k in DEFAULTS.keys():
            if k == "DECAY_ms":
                raw[k] = cv2.getTrackbarPos("DECAY_ms", self.win)
            else:
                raw[k] = cv2.getTrackbarPos(k, self.win)
        return raw

    def get(self):
        gp = lambda k: max(self._mins[k], cv2.getTrackbarPos(k, self.win)) if k in self._mins else cv2.getTrackbarPos(k, self.win)

        zmin = gp("Z_MIN_cm") / 100.0
        zmax = gp("Z_MAX_cm") / 100.0
        if zmax <= zmin + 0.05: zmax = zmin + 0.05

        near_obj = gp("NEAR_FOR_OBJ_cm") / 100.0
        floor_th = gp("FLOOR_TH_mm") / 1000.0
        wall_th  = gp("WALL_TH_mm")  / 1000.0
        floor_align = gp("FLOOR_ALIGN_DEG")
        wall_align  = gp("WALL_ALIGN_DEG")
        max_walls   = gp("MAX_WALLS")
        conf_th     = gp("CONF_TH")
        a_floor     = gp("ALPHA_FLOOR_pct") / 100.0
        a_wall      = gp("ALPHA_WALL_pct")  / 100.0
        a_obj       = gp("ALPHA_OBJ_pct")   / 100.0
        refit_sec   = gp("REFIT_ms") / 1000.0

        sticky_on   = gp("STICKY_ON") > 0
        decay_ms    = cv2.getTrackbarPos("DECAY_ms", self.win)  # exact 0 ok
        sticky_thr  = gp("STICKY_THR")
        dilate_px   = gp("DILATE_px")

        # Colors
        floor_color = (gp("FLOOR_B"), gp("FLOOR_G"), gp("FLOOR_R"))
        wall_color  = (gp("WALL_B"),  gp("WALL_G"),  gp("WALL_R"))

        return {
            "Z_MIN": zmin, "Z_MAX": zmax,
            "NEAR_FOR_OBJ": near_obj,
            "FLOOR_TH": floor_th, "WALL_TH": wall_th,
            "FLOOR_ALIGN_DEG": floor_align, "WALL_ALIGN_DEG": wall_align,
            "MAX_WALLS": max_walls,
            "CONF_TH": conf_th,
            "ALPHA_FLOOR": a_floor, "ALPHA_WALL": a_wall, "ALPHA_OBJ": a_obj,
            "REFIT_SEC": refit_sec,
            "STICKY_ON": sticky_on, "DECAY_MS": decay_ms,
            "STICKY_THR": sticky_thr, "DILATE_PX": dilate_px,
            "FLOOR_COLOR": floor_color, "WALL_COLOR": wall_color
        }

    def get_persistable(self):
        # Return integer slider positions suitable for saving
        raw = self.read_raw()
        out = {}
        for k, v in raw.items():
            if k == "DECAY_ms":
                out[k] = int(v)  # allow 0
            else:
                out[k] = int(max(self._mins.get(k, 0), v))
        return out

# -------- Helpers --------
def up_vector_cam():
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)

def precompute_dirs(W, H, fx, fy, cx, cy):
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) / fx
    Y = (vv - cy) / fy
    Z = np.ones_like(X, dtype=np.float32)
    return np.stack([X, Y, Z], axis=2)

def build_points(depth_s, dirs):
    return depth_s[..., None] * dirs

def angle_to_up(n, up):
    num = np.clip((n * up).sum(axis=-1) / (np.linalg.norm(n, axis=-1) * (np.linalg.norm(up)+1e-9) + 1e-9), -1, 1)
    return np.degrees(np.arccos(num))

def ransac_plane(P, mask, want='floor', up=None, dist_th=0.03, iters=180, subsample=4000,
                 floor_align_deg=25, wall_align_deg=20):
    idx = np.argwhere(mask)
    if idx.shape[0] < 200:
        return None, None, np.zeros_like(mask)
    choose = np.random.choice(idx.shape[0], size=min(subsample, idx.shape[0]), replace=False)
    sub = idx[choose]
    P_sub = P[sub[:,0], sub[:,1]]

    best = (0, None, None)
    for _ in range(iters):
        i = np.random.choice(idx.shape[0], size=3, replace=False)
        p0, p1, p2 = P[idx[i[0],0], idx[i[0],1]], P[idx[i[1],0], idx[i[1],1]], P[idx[i[2],0], idx[i[2],1]]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6: continue
        n = n / nn
        if up is not None:
            a = float(angle_to_up(n[None,:], up)[0])
            if want == 'floor':
                if a > floor_align_deg: continue
            else:
                if abs(a - 90.0) > wall_align_deg: continue
        d = -float(np.dot(n, p0))
        dist = np.abs(P_sub @ n + d)
        count = int((dist <= dist_th).sum())
        if count > best[0]:
            best = (count, n, d)

    n, d = best[1], best[2]
    if n is None:
        return None, None, np.zeros_like(mask)
    Pf = P.reshape(-1,3)
    mflat = mask.reshape(-1)
    dist_all = np.abs(Pf @ n + d)
    inliers = (mflat) & (dist_all <= dist_th)
    return n, d, inliers.reshape(mask.shape)

def overlay_colors(base_bgr, floor, wall, obj, depth,
                   alpha_floor, alpha_wall, alpha_obj,
                   floor_color=(0,255,255), wall_color=(0,0,255),
                   near=0.35, far=4.5):
    disp = base_bgr.copy()
    if floor.any():
        over = disp.copy(); over[floor] = floor_color
        disp = cv2.addWeighted(over, alpha_floor, disp, 1-alpha_floor, 0)
    if wall.any():
        over = disp.copy(); over[wall] = wall_color
        disp = cv2.addWeighted(over, alpha_wall, disp, 1-alpha_wall, 0)
    if obj.any():
        yy, xx = np.where(obj)
        green = np.zeros_like(disp, np.uint8)
        d = depth[yy, xx]
        inten = np.clip((far - d) / (far - near + 1e-6), 0.0, 1.0)
        g = (inten * 255.0).astype(np.uint8)
        green[yy, xx, 1] = g
        disp = cv2.addWeighted(green, alpha_obj, disp, 1-alpha_obj, 0)
    return disp

# ---- Sticky helpers ----
def decay_conf(conf_u8, dt_ms, decay_ms):
    if decay_ms <= 0:
        return conf_u8
    factor = math.exp(-float(dt_ms) / float(decay_ms))
    out = (conf_u8.astype(np.float32) * factor).astype(np.uint8)
    return out

def commit_mask(conf_u8, new_bool, dilate_px):
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        new_bool = cv2.dilate(new_bool.astype(np.uint8), k, 1).astype(bool)
    add = np.where(new_bool, 255, 0).astype(np.uint8)
    return np.maximum(conf_u8, add)

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
dirs = precompute_dirs(PROC_W, proc_h, fx_p, fy_p, cx_p, cy_p)

m_depth = sl.Mat()
m_left  = sl.Mat()

up = up_vector_cam()
last_refit = 0
planes = {"floor": None, "walls": []}

ema_fps = None
t_prev = time.time()

# ---- Controls & sticky state (auto-load) ----
initial_cfg = load_params()
ctrl = Controls(initial=initial_cfg)

sticky_floor_conf = None
sticky_wall_conf  = None
last_frame_time_ms = int(time.time() * 1000)

print("Running. ESC=quit (auto-save), P=save, L=load, F=force re-fit, R=reset sticky, S=toggle sticky.")
saved_flash_ms = 0
loaded_flash_ms = 0

while True:
    PVAL = ctrl.get()
    Z_MIN, Z_MAX = PVAL["Z_MIN"], PVAL["Z_MAX"]
    NEAR_FOR_OBJ = PVAL["NEAR_FOR_OBJ"]
    FLOOR_TH, WALL_TH = PVAL["FLOOR_TH"], PVAL["WALL_TH"]
    FLOOR_ALIGN_DEG, WALL_ALIGN_DEG = PVAL["FLOOR_ALIGN_DEG"], PVAL["WALL_ALIGN_DEG"]
    MAX_WALLS = PVAL["MAX_WALLS"]
    runtime.confidence_threshold = int(PVAL["CONF_TH"])
    ALPHA_FLOOR, ALPHA_WALL, ALPHA_OBJ = PVAL["ALPHA_FLOOR"], PVAL["ALPHA_WALL"], PVAL["ALPHA_OBJ"]
    REFIT_SEC = PVAL["REFIT_SEC"]
    STICKY_ON, DECAY_MS, STICKY_THR, DILATE_PX = PVAL["STICKY_ON"], PVAL["DECAY_MS"], PVAL["STICKY_THR"], PVAL["DILATE_PX"]
    FLOOR_COLOR, WALL_COLOR = PVAL["FLOOR_COLOR"], PVAL["WALL_COLOR"]

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        continue
    zed.retrieve_measure(m_depth, sl.MEASURE.DEPTH)
    zed.retrieve_image(m_left, sl.VIEW.LEFT)

    depth = m_depth.get_data().copy().astype(np.float32)
    left  = m_left.get_data().copy()[:, :, :3]
    H, W  = depth.shape

    if sticky_floor_conf is None:
        sticky_floor_conf = np.zeros((H, W), np.uint8)
        sticky_wall_conf  = np.zeros((H, W), np.uint8)

    # Downsample for plane work
    depth_s = cv2.resize(depth, (PROC_W, proc_h), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth_s) & (depth_s > Z_MIN) & (depth_s < Z_MAX)
    Ppts = build_points(depth_s, dirs)

    # Timing for decay
    now_ms = int(time.time() * 1000)
    dt_ms = now_ms - last_frame_time_ms
    last_frame_time_ms = now_ms

    key = cv2.waitKey(1) & 0xFF
    need_refit = (time.time() - last_refit) >= REFIT_SEC
    if key in (ord('f'), ord('F')): need_refit = True
    if key in (ord('r'), ord('R')):
        sticky_floor_conf[:] = 0
        sticky_wall_conf[:]  = 0
    if key in (ord('s'), ord('S')):
        cur = 0 if STICKY_ON else 1
        cv2.setTrackbarPos("STICKY_ON", ctrl.win, cur)
        STICKY_ON = not STICKY_ON
    if key in (ord('p'), ord('P')):
        ok = save_params(ctrl.get_persistable())
        saved_flash_ms = 800 if ok else 0
    if key in (ord('l'), ord('L')):
        cfg = load_params()
        for k, v in cfg.items():
            cv2.setTrackbarPos(k if k != "DECAY_ms" else "DECAY_ms", ctrl.win, int(v))
        loaded_flash_ms = 800
    if key == 27:  # ESC: auto-save on exit
        save_params(ctrl.get_persistable())
        break

    # Refit planes
    if need_refit:
        floor_roi = np.zeros_like(valid); floor_roi[int(proc_h*0.55):, :] = True
        nF, dF, inF = ransac_plane(
            Ppts, valid & floor_roi, want='floor', up=up,
            dist_th=FLOOR_TH, iters=160, subsample=3500,
            floor_align_deg=FLOOR_ALIGN_DEG, wall_align_deg=WALL_ALIGN_DEG
        )
        if nF is not None:
            planes["floor"] = (nF.astype(np.float32), float(dF))
        walls = []
        remaining = valid & (~inF if nF is not None else valid)
        for _ in range(int(MAX_WALLS)):
            if remaining.sum() < 400: break
            nW, dW, inW = ransac_plane(
                Ppts, remaining, want='wall', up=up,
                dist_th=WALL_TH, iters=140, subsample=3000,
                floor_align_deg=WALL_ALIGN_DEG, wall_align_deg=WALL_ALIGN_DEG
            )
            if nW is None: break
            walls.append((nW.astype(np.float32), float(dW)))
            remaining &= (~inW)
        planes["walls"] = walls
        last_refit = time.time()

    # Classify at PROC size
    floor_s = np.zeros_like(valid)
    if planes["floor"] is not None:
        nF, dF = planes["floor"]
        distF = np.abs((Ppts @ nF) + dF)
        floor_s = valid & (distF <= FLOOR_TH)

    wall_s = np.zeros_like(valid)
    for (nW, dW) in planes["walls"]:
        distW = np.abs((Ppts @ nW) + dW)
        wall_s |= (valid & ~floor_s & (distW <= WALL_TH))

    obj_s = valid & ~floor_s & ~wall_s & (depth_s <= NEAR_FOR_OBJ)
    obj_u8 = (obj_s.astype(np.uint8) * 255)
    obj_u8 = cv2.morphologyEx(obj_u8, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    obj_full = cv2.resize(obj_u8, (W, H), cv2.INTER_NEAREST).astype(bool)

    # Upsample to full res
    floor_full_new = cv2.resize(floor_s.astype(np.uint8), (W, H), cv2.INTER_NEAREST).astype(bool)
    wall_full_new  = cv2.resize(wall_s.astype(np.uint8),  (W, H), cv2.INTER_NEAREST).astype(bool)

    # Sticky logic + decay
    if STICKY_ON:
        sticky_floor_conf = decay_conf(sticky_floor_conf, dt_ms, DECAY_MS)
        sticky_wall_conf  = decay_conf(sticky_wall_conf,  dt_ms, DECAY_MS)
        sticky_floor_conf = commit_mask(sticky_floor_conf, floor_full_new, DILATE_PX)
        sticky_wall_conf  = commit_mask(sticky_wall_conf,  wall_full_new,  DILATE_PX)
        floor_full = (sticky_floor_conf >= STICKY_THR)
        wall_full  = (sticky_wall_conf  >= STICKY_THR)
    else:
        floor_full = floor_full_new
        wall_full  = wall_full_new

    disp = overlay_colors(
        left, floor_full, wall_full, obj_full, depth,
        alpha_floor=ALPHA_FLOOR, alpha_wall=ALPHA_WALL, alpha_obj=ALPHA_OBJ,
        floor_color=FLOOR_COLOR, wall_color=WALL_COLOR,
        near=0.35, far=NEAR_FOR_OBJ
    )

    # HUD
    t = time.time()
    fps = 1.0 / max(1e-6, (t - (t_prev))); t_prev = t
    ema_fps = fps if ema_fps is None else (0.9*ema_fps + 0.1*fps)
    hud1 = (f"FPS:{ema_fps:4.1f}  floor={'Y' if planes['floor'] else 'N'}  walls={len(planes['walls'])}  "
            f"Z=[{Z_MIN:.2f},{Z_MAX:.2f}]  NEAR_OBJ={NEAR_FOR_OBJ:.2f}m  conf={runtime.confidence_threshold}")
    hud2 = (f"TH: floor={FLOOR_TH*1000:.0f}mm wall={WALL_TH*1000:.0f}mm  "
            f"ALIGN: floor<=±{FLOOR_ALIGN_DEG}° wall<=±{WALL_ALIGN_DEG}°  "
            f"refit={REFIT_SEC:.2f}s  F=force  R=reset  S=sticky:{int(STICKY_ON)}  "
            f"P=save  L=load  FloorColor(BGR)={FLOOR_COLOR} WallColor(BGR)={WALL_COLOR}")
    cv2.putText(disp, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(disp, hud2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Realtime plane tracking (fast)", disp)

# Cleanup
cv2.destroyAllWindows()
zed.close()
