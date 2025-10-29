# blind_nav_overhang_strict.py
# ZED blind-walk assist with robust trip detection + overhang-only head hazards.
# - Trip hazards: floor-plane heights (1–8 in), cleaned with morphology + CCs
# - Bump hazards: torso band (zone + size test)
# - Head hazards: ONLY if overhang (head obstacle but torso clear nearby)
# - Wall suppression per column (smooth top->bottom fill)
# - Chest-mounted camera (4.5 ft) accounted for in head band
#
# Minimal GUI: colorized depth + short text labels only.

import time
import numpy as np
import cv2
import pyzed.sl as sl

# ===================== Config =====================
CAM_RES  = sl.RESOLUTION.HD720
CAM_FPS  = 30
DEPTH_MODE = sl.DEPTH_MODE.NEURAL
UNITS = sl.UNIT.METER

# Camera mounting height (meters) ~ chest at 4.5 ft
CAMERA_HEIGHT_M = 4.5 * 0.3048  # 1.3716 m

# Downscale for analysis (overlay stays full-res)
PROCESS_SCALE = 0.45            # 0.35–0.5 is good

# Heavy work cadence
PLANE_EVERY_N = 6               # fit floor every N frames (reuse otherwise)
WALL_EVERY_N  = 4               # check walls every M frames

# Valid depth limits (meters)
DEPTH_MIN, DEPTH_MAX = 0.25, 6.0

# Max forward range (how far ahead to warn)
TRIP_DIST_MAX = 1.8
BUMP_DIST_MAX = 1.6
HEAD_DIST_MAX = 1.8

# Height bands above floor (meters)
TRIP_H_MIN, TRIP_H_MAX = 0.025, 0.20      # ~1–8 in (include 1" for thin lips)
BUMP_H_MIN, BUMP_H_MAX = 0.25, 1.20
HEAD_H_MIN = CAMERA_HEIGHT_M + 0.20       # chest + ~8 in
HEAD_H_MAX = CAMERA_HEIGHT_M + 0.60       # chest + ~24 in

# Zone layout (fractions of image)
TOP_FRAC, MID_FRAC, BOT_FRAC = 0.30, 0.40, 0.30
LEFT_FRAC, CENT_FRAC, RIGHT_FRAC = 0.33, 0.34, 0.33

# Morphology for masks (downscaled)
MORPH_K = (3,3)
TRIP_MIN_AREA   = 80     # min CC area (proc pixels) to count as real trip blob
BUMP_MIN_AREA   = 80
HEAD_MIN_AREA   = 80

# Coverage thresholds (zone-wise % of valid pixels inside mask)
COVER_TRIP_MIN = 0.010   # 1%+
COVER_BUMP_MIN = 0.015
COVER_HEAD_MIN = 0.015

# Overhang logic (for head hazards)
HEAD_TORSO_GAP_M       = 0.35  # need torso to be farther than head by this margin (~14 in)
TORSO_CLEAR_COVER_MAX  = 0.008 # torso coverage at ~head distance must be very small to call overhang

# Robust stats
ROBUST_K = 120
SMOOTH_GAUSS_K = 3  # 0/None to disable light smoothing

# RANSAC floor plane (downscaled PC)
RANSAC_ITERS = 70
PLANE_INLIER_THR = 0.02
PLANE_MIN_INLIERS = 800
PLANE_HEIGHT_TOL = 0.35

# Wall heuristic (downscaled depth)
WALL_NEAR_MAX_M     = 2.5
WALL_VERT_COVER_MIN = 0.80
WALL_GRAD_STD_MAX   = 0.02
WALL_ABRUPT_JUMP_M  = 0.30

# GUI
FONT = cv2.FONT_HERSHEY_SIMPLEX
FG   = (255,255,255)
BG   = (0,0,0)

# ================= Utilities ======================
def format_imperial(m):
    if m is None or not np.isfinite(m) or m <= 0:
        return "--"
    inches = m * 39.3701
    if inches < 12:
        return f"{inches:.1f} in"
    ft = int(inches // 12)
    rem = inches % 12
    return f"{ft} ft {rem:.1f} in"

def colorize_depth(depth_m, vmin=0.4, vmax=4.0):
    d = depth_m.copy()
    valid = np.isfinite(d) & (d > 0)
    d[~valid] = 0
    d = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    vis = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def put_label(img, text, cx, cy, scale=0.9, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(cx - tw/2), int(cy + th/2)
    cv2.putText(img, text, (x, y), FONT, scale, BG, thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, FG, thickness,   cv2.LINE_AA)

def robust_nearest(arr, k=ROBUST_K):
    vals = arr[np.isfinite(arr) & (arr > 0)]
    if vals.size == 0:
        return None
    k = min(k, vals.size)
    return float(np.median(np.partition(vals, k-1)[:k]))

# ================= Slicing helpers =================
def make_slices(h, w):
    t = int(h * TOP_FRAC); m = int(h * (TOP_FRAC + MID_FRAC))
    l = int(w * LEFT_FRAC); c = int(w * (LEFT_FRAC + CENT_FRAC))
    centers = {
        "trip_left":   ( (l*0.5),            (m + (h-m)*0.7) ),
        "trip_front":  ( (l + (c-l)*0.5),    (m + (h-m)*0.7) ),
        "trip_right":  ( (c + (w-c)*0.5),    (m + (h-m)*0.7) ),
        "bump_left":   ( (l*0.5),            (t + (m-t)*0.5) ),
        "bump_right":  ( (c + (w-c)*0.5),    (t + (m-t)*0.5) ),
        "head_left":   ( (l*0.5),            (t*0.5) ),
        "head_center": ( (l + (c-l)*0.5),    (t*0.5) ),
        "head_right":  ( (c + (w-c)*0.5),    (t*0.5) ),
        "wall_left":   ( (l*0.5),            h*0.1 ),
        "wall_center": ( (l + (c-l)*0.5),    h*0.1 ),
        "wall_right":  ( (c + (w-c)*0.5),    h*0.1 ),
    }
    return {
        # trip
        "trip_left":   (slice(m, h), slice(0, l)),
        "trip_front":  (slice(m, h), slice(l, c)),
        "trip_right":  (slice(m, h), slice(c, w)),
        # bump
        "bump_left":   (slice(t, m), slice(0, l)),
        "bump_right":  (slice(t, m), slice(c, w)),
        # head
        "head_left":   (slice(0, t), slice(0, l)),
        "head_center": (slice(0, t), slice(l, c)),
        "head_right":  (slice(0, t), slice(c, w)),
        # columns for walls
        "col_left":    (slice(0, h), slice(0, l)),
        "col_center":  (slice(0, h), slice(l, c)),
        "col_right":   (slice(0, h), slice(c, w)),
        "_centers": centers
    }

# =============== Plane fitting (downscaled PC) ===============
def fit_floor_plane_ransac_downscaled(pc_proc):
    """Fit plane n·X + d = 0 using RANSAC on bottom band."""
    H, W, _ = pc_proc.shape
    r0 = int(H * 0.65)  # bottom ~35%
    pts = pc_proc[r0:, :, :].reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] < 300:
        return None, None

    best_n = None; best_d = None; best_count = -1
    N = pts.shape[0]
    for _ in range(RANSAC_ITERS):
        ids = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts[ids]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-6: continue
        n = n / nn
        d = -np.dot(n, p0)

        dist = np.abs(pts @ n + d)
        count = int((dist < PLANE_INLIER_THR).sum())
        if count > best_count:
            best_count = count; best_n = n.copy(); best_d = float(d)

    if best_n is None or best_count < PLANE_MIN_INLIERS:
        return None, None

    # Orient so camera origin is on positive side (d positive)
    if best_d < 0:
        best_n = -best_n; best_d = -best_d

    # sanity (optional)
    # if abs(best_d - CAMERA_HEIGHT_M) > PLANE_HEIGHT_TOL: pass

    return best_n, best_d

def height_above_plane_proc(n, d_unit, pc_proc):
    """Height above plane (floor ≈ 0), clipped at 0."""
    h = (pc_proc[...,0]*n[0] + pc_proc[...,1]*n[1] + pc_proc[...,2]*n[2]) + d_unit
    return np.where(h < 0, 0, h)

# ================= Wall heuristic ===================
def detect_wall_in_column(depth_proc, slc_vertcol):
    col = depth_proc[slc_vertcol]
    h, w = col.shape
    valid = np.isfinite(col) & (col > 0) & (col <= WALL_NEAR_MAX_M)
    vert_cover = np.count_nonzero(np.any(valid, axis=1)) / float(h) if h > 0 else 0.0
    if vert_cover < WALL_VERT_COVER_MIN:
        return False, None

    row_med = np.full((h,), np.nan, dtype=np.float32)
    for i in range(h):
        v = valid[i]
        if np.any(v):
            row_med[i] = np.median(col[i][v])

    mask = np.isfinite(row_med)
    if mask.sum() < h * 0.6:
        return False, None

    idx = np.arange(h)
    filled = row_med.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], filled[mask])

    filled_sm = cv2.GaussianBlur(filled.reshape(-1,1), (5,1), 0).ravel()
    grad = np.diff(filled_sm)
    if grad.size == 0:
        return False, None
    grad_std = float(np.nanstd(grad))
    big_jumps = int(np.count_nonzero(np.abs(grad) > WALL_ABRUPT_JUMP_M))

    is_wall = (grad_std <= WALL_GRAD_STD_MAX) and (big_jumps == 0)
    rep = float(np.nanmedian(filled_sm))
    return is_wall, rep

# =============== Component tools ===================
def clean_and_cc(mask, min_area):
    """Morphologically clean a binary mask (uint8 {0,255}), return CC stats."""
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_K)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mk, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, mk, iterations=1)
    num, lbl, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep.append(i)
    return lbl, stats, cent, keep

def nearest_in_zone(depth_proc, rsl, csl):
    z = depth_proc[rsl, csl]
    return robust_nearest(z)

# ===================== Main =========================
def main():
    cv2.setUseOptimized(True)

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = CAM_RES
    init.camera_fps = CAM_FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = UNITS
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    pc_mat = sl.Mat()

    frame_i = 0
    last_report = 0.0
    REPORT_HZ = 2.0

    plane_n = None
    plane_d = None
    height_proc = None

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001); continue

            # Depth always
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
            depth = depth_mat.get_data()
            H, W = depth.shape

            # GUI base
            vis = colorize_depth(depth)

            # Downscaled depth
            pw, ph = max(64, int(W*PROCESS_SCALE)), max(48, int(H*PROCESS_SCALE))
            depth_proc = cv2.resize(depth, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
            depth_proc[~(np.isfinite(depth_proc) & (depth_proc >= DEPTH_MIN) & (depth_proc <= DEPTH_MAX))] = np.nan

            # Refresh plane/height occasionally
            if (frame_i % PLANE_EVERY_N == 0) or (plane_n is None):
                zed.retrieve_measure(pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
                pc = pc_mat.get_data()
                # Downsample PC by striding to match (ph,pw)
                step_h = max(1, int(round(H / ph)))
                step_w = max(1, int(round(W / pw)))
                pc_proc = pc[::step_h, ::step_w, :][:ph, :pw, :]

                plane_n, plane_d = fit_floor_plane_ransac_downscaled(pc_proc)
                if plane_n is not None:
                    height_proc = height_above_plane_proc(plane_n, plane_d, pc_proc)
                else:
                    # Fallback: row-based approx (less accurate)
                    rr = np.linspace(0, 1, ph, dtype=np.float32).reshape(ph,1)
                    height_proc = (1 - rr) * CAMERA_HEIGHT_M
            elif height_proc is None:
                # ensure defined on first frames
                zed.retrieve_measure(pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
                pc = pc_mat.get_data()
                step_h = max(1, int(round(H / ph)))
                step_w = max(1, int(round(W / pw)))
                pc_proc = pc[::step_h, ::step_w, :][:ph, :pw, :]
                if plane_n is None:
                    plane_n, plane_d = fit_floor_plane_ransac_downscaled(pc_proc)
                if plane_n is not None:
                    height_proc = height_above_plane_proc(plane_n, plane_d, pc_proc)
                else:
                    rr = np.linspace(0, 1, ph, dtype=np.float32).reshape(ph,1)
                    height_proc = (1 - rr) * CAMERA_HEIGHT_M

            # Light smoothing
            if SMOOTH_GAUSS_K and SMOOTH_GAUSS_K >= 3:
                k = int(SMOOTH_GAUSS_K) | 1
                for arr in (depth_proc, height_proc):
                    tmp = arr.copy()
                    tmp[np.isnan(tmp)] = 0.0
                    keep = np.isfinite(arr).astype(np.uint8)
                    tmp = cv2.GaussianBlur(tmp, (k,k), 0)
                    keep_blur = cv2.GaussianBlur(keep, (k,k), 0)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        arr[...] = np.where(keep_blur > 0, tmp / np.maximum(keep_blur,1e-6), np.nan)

            # Zones
            sl_full = make_slices(H, W)
            sl_proc = make_slices(ph, pw)
            centers = sl_full["_centers"]

            labels = []

            # ===== TRIP HAZARDS (bottom thirds) =====
            trip_band = (height_proc >= TRIP_H_MIN) & (height_proc <= TRIP_H_MAX) & (depth_proc <= TRIP_DIST_MAX)
            trip_mask = np.where(np.isfinite(depth_proc) & trip_band, 255, 0).astype(np.uint8)
            lbl, stats, cent, keep = clean_and_cc(trip_mask, TRIP_MIN_AREA)

            for key in ("trip_left","trip_front","trip_right"):
                rsl, csl = sl_proc[key]
                zone_lbl = lbl[rsl, csl]
                zone_keep = [i for i in keep if np.any(zone_lbl == i)]
                if not zone_keep:
                    continue
                # compute coverage in zone
                zone_valid = np.isfinite(depth_proc[rsl, csl])
                zone_mask = (zone_lbl > 0)
                cover = float(np.count_nonzero(zone_mask & zone_valid)) / max(1, int(np.count_nonzero(zone_valid)))
                if cover < COVER_TRIP_MIN:
                    continue
                # nearest distance of those pixels
                near = nearest_in_zone(depth_proc, rsl, csl)
                if near is None:
                    continue
                cx, cy = centers[key]
                labels.append((f"TRIP {key.split('_')[1]} {format_imperial(near)}", cx, cy))

            # ===== BUMP HAZARDS (middle L/R) =====
            bump_band = (height_proc >= BUMP_H_MIN) & (height_proc <= BUMP_H_MAX) & (depth_proc <= BUMP_DIST_MAX)
            bump_mask = np.where(np.isfinite(depth_proc) & bump_band, 255, 0).astype(np.uint8)
            lbl, stats, cent, keep = clean_and_cc(bump_mask, BUMP_MIN_AREA)

            for key in ("bump_left","bump_right"):
                rsl, csl = sl_proc[key]
                zone_lbl = lbl[rsl, csl]
                zone_keep = [i for i in keep if np.any(zone_lbl == i)]
                if not zone_keep:
                    continue
                zone_valid = np.isfinite(depth_proc[rsl, csl])
                zone_mask = (zone_lbl > 0)
                cover = float(np.count_nonzero(zone_mask & zone_valid)) / max(1, int(np.count_nonzero(zone_valid)))
                if cover < COVER_BUMP_MIN:
                    continue
                near = nearest_in_zone(depth_proc, rsl, csl)
                if near is None:
                    continue
                cx, cy = centers[key]
                side = "left" if "left" in key else "right"
                labels.append((f"BUMP {side} {format_imperial(near)}", cx, cy))

            # ===== HEAD HAZARDS (top thirds) – OVERHANG ONLY =====
            head_band = (height_proc >= HEAD_H_MIN) & (height_proc <= HEAD_H_MAX) & (depth_proc <= HEAD_DIST_MAX)
            head_mask = np.where(np.isfinite(depth_proc) & head_band, 255, 0).astype(np.uint8)
            lbl_h, stats_h, cent_h, keep_h = clean_and_cc(head_mask, HEAD_MIN_AREA)

            # Make a torso mask around the prospective head distance to assert "overhang"
            torso_band = (height_proc >= BUMP_H_MIN) & (height_proc <= BUMP_H_MAX)

            def torso_cover_near(rsl, csl, head_near):
                if head_near is None:
                    return 1.0  # treat as not overhang
                # pixels at torso height and within (head_near + margin)
                near_mask = torso_band & (depth_proc <= (head_near + HEAD_TORSO_GAP_M - 0.10))
                m = np.where(np.isfinite(depth_proc) & near_mask, 1, 0).astype(np.uint8)
                z = m[rsl, csl]
                valid = np.isfinite(depth_proc[rsl, csl])
                denom = int(np.count_nonzero(valid))
                if denom == 0: return 1.0
                return float(np.count_nonzero(z & valid)) / float(denom)

            for key in ("head_left","head_center","head_right"):
                rsl, csl = sl_proc[key]
                zone_lbl = lbl_h[rsl, csl]
                zone_keep = [i for i in keep_h if np.any(zone_lbl == i)]
                if not zone_keep:
                    continue
                zone_valid = np.isfinite(depth_proc[rsl, csl])
                zone_mask = (zone_lbl > 0)
                cover = float(np.count_nonzero(zone_mask & zone_valid)) / max(1, int(np.count_nonzero(zone_valid)))
                if cover < COVER_HEAD_MIN:
                    continue

                # nearest head distance in this zone
                head_near = nearest_in_zone(depth_proc, rsl, csl)
                if head_near is None:
                    continue

                # torso must be CLEAR near that distance => overhang
                torso_cov = torso_cover_near(rsl, csl, head_near)
                # Also require torso nearest to be sufficiently farther than head
                torso_near = robust_nearest(depth_proc[rsl, csl][torso_band[rsl, csl]])
                far_enough = (torso_near is None) or (torso_near - head_near >= HEAD_TORSO_GAP_M)

                if torso_cov <= TORSO_CLEAR_COVER_MAX and far_enough:
                    cx, cy = centers[key]
                    zone = key.split('_')[1]
                    labels.append((f"HEAD {zone} {format_imperial(head_near)}", cx, cy))
                # else: likely a wall/flat face ⇒ suppress head

            # ===== WALLS (per vertical third) =====
            walls = []
            if frame_i % WALL_EVERY_N == 0:
                for col_key, label_key in (("col_left","wall_left"), ("col_center","wall_center"), ("col_right","wall_right")):
                    is_wall, rep = detect_wall_in_column(depth_proc, sl_proc[col_key])
                    if is_wall and rep is not None and rep <= WALL_NEAR_MAX_M:
                        cx, cy = centers[label_key]
                        walls.append((f"WALL ~{format_imperial(rep)}", cx, cy))

            # ===== Column conflict resolution =====
            def col_of_x(x):
                l_end = int(W * LEFT_FRAC)
                c_end = int(W * (LEFT_FRAC + CENT_FRAC))
                return 0 if x < l_end else (1 if x < c_end else 2)

            def parse_meters_from_text(txt):
                s = txt.replace('~','')
                toks = s.split()
                ft = None; inch = 0.0
                for i,t in enumerate(toks):
                    if t == 'ft' and i>0:
                        try: ft = float(toks[i-1])
                        except: pass
                    if t == 'in' and i>0:
                        try: inch = float(toks[i-1])
                        except: pass
                if ft is None and inch == 0.0:
                    return 1e9
                return ((ft or 0.0)*12.0 + inch) / 39.3701

            draw_list = labels[:]
            if walls:
                combined = walls + labels
                buckets = {0: [], 1: [], 2: []}
                for t, x, y in combined:
                    buckets[col_of_x(x)].append((t,x,y))
                draw_list = []
                for col, items in buckets.items():
                    if not items: continue
                    has_wall = any("WALL" in t for (t,_,_) in items)
                    if has_wall:
                        # keep the single closest item; walls lose if something is clearly closer
                        draw_list.append(min(items, key=lambda it: parse_meters_from_text(it[0])))
                    else:
                        draw_list.extend(items)

            # ---------- GUI ----------
            for text, cx, cy in draw_list:
                put_label(vis, text, cx, cy, scale=0.9, thickness=2)

            cv2.imshow("Blind-Assist (overhang-strict)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # ---------- Console (2 Hz) ----------
            now = time.time()
            if (now - last_report) >= (1.0 / REPORT_HZ):
                last_report = now
                if draw_list:
                    print(" | ".join([t for (t,_,_) in draw_list]))
                else:
                    print("Clear within configured ranges.")

            frame_i += 1

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        depth_mat.free(); pc_mat.free()
        zed.close()

if __name__ == "__main__":
    main()
