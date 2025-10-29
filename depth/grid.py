# zed_3x3_red_cells_6ft_rgb_proportioned_singlewin_scaled.py
import time
import numpy as np
import cv2
import pyzed.sl as sl

# ============== Config ==============
SHOW_WINDOW = True
WIN_NAME = "ZED 3x3 Assist (RGB <=6ft)"
DISPLAY_SCALE = 0.6          # 0.2–1.0; only affects the GUI, not detection

CAM_RES    = sl.RESOLUTION.HD720
CAM_FPS    = 30
DEPTH_MODE = sl.DEPTH_MODE.NEURAL

MAX_CONSIDER_FT = 6.0
DEPTH_MIN = 0.25                              # meters
DEPTH_MAX = MAX_CONSIDER_FT * 0.3048          # 1.8288 m
ROBUST_K  = 120

# Person-proportioned grid
ROW_FEET = (1.0, 4.0, 1.0)   # head : torso : feet (total ~6 ft)
COL_FEET = (1.5, 1.0, 1.5)   # left : body : right (total ~4 ft incl. buffers)

# Overlay tuning
ALPHA_MIN, ALPHA_MAX = 0.25, 0.95
CLOSENESS_POW = 1.25
GRID_LINE_COLOR = (255, 255, 255)
GRID_LINE_THICK = 2

# ============== Helpers ==============
def proportioned_edges(total_px: int, feet_triplet):
    w = np.array(feet_triplet, dtype=np.float32)
    w = np.maximum(w, 1e-6)
    frac = w / np.sum(w)
    e1 = int(round(total_px * frac[0]))
    e2 = int(round(total_px * (frac[0] + frac[1])))
    return [0, e1, e2, total_px]

def proportioned_grid_bounds(h, w):
    r_edges = proportioned_edges(h, ROW_FEET)
    c_edges = proportioned_edges(w, COL_FEET)
    for r in range(3):
        for c in range(3):
            yield r, c, r_edges[r], r_edges[r+1], c_edges[c], c_edges[c+1]

def cell_nearest(depth_m, r0, r1, c0, c1):
    cell = depth_m[r0:r1, c0:c1]
    if cell.size == 0:
        return None
    valid = np.isfinite(cell) & (cell > 0)
    if DEPTH_MIN is not None: valid &= (cell >= DEPTH_MIN)
    if DEPTH_MAX is not None: valid &= (cell <= DEPTH_MAX)
    if not np.any(valid): return None
    d = cell[valid].ravel()
    k = min(ROBUST_K, d.size)
    return float(np.median(np.partition(d, k - 1)[:k]))

def blend_cell_red(vis_bgr, r0, r1, c0, c1, alpha):
    if r1 <= r0 or c1 <= c0: return
    roi = vis_bgr[r0:r1, c0:c1]
    red = np.empty_like(roi); red[:] = (0, 0, 255)
    cv2.addWeighted(red, float(alpha), roi, 1.0 - float(alpha), 0.0, dst=roi)

# ============== Main ==============
def main():
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = CAM_RES
    init.camera_fps = CAM_FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = sl.UNIT.METER
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()
    image_mat = sl.Mat()

    if SHOW_WINDOW:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)  # resizable window

    disp_scale = DISPLAY_SCALE

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001); continue

            # RGB frame (BGRA -> BGR)
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame = image_mat.get_data()
            vis = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame.copy()
            H, W = vis.shape[:2]

            # Depth
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
            depth = depth_mat.get_data()

            # Overlay per cell
            for _, _, r0, r1, c0, c1 in proportioned_grid_bounds(H, W):
                nearest = cell_nearest(depth, r0, r1, c0, c1)
                if nearest is None: continue
                closeness = 1.0 - (nearest - DEPTH_MIN) / max(1e-6, (DEPTH_MAX - DEPTH_MIN))
                closeness = float(np.clip(closeness, 0.0, 1.0)) ** CLOSENESS_POW
                alpha = float(np.clip(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN)*closeness, 0.0, 1.0))
                blend_cell_red(vis, r0, r1, c0, c1, alpha)

            # Grid lines
            for _, _, r0, r1, c0, c1 in proportioned_grid_bounds(H, W):
                cv2.rectangle(vis, (c0, r0), (c1 - 1, r1 - 1), GRID_LINE_COLOR, GRID_LINE_THICK, cv2.LINE_AA)

            # ---- Display scaling (GUI only) ----
            if SHOW_WINDOW:
                s = max(0.1, min(1.5, float(disp_scale)))  # clamp for safety
                if abs(s - 1.0) > 1e-3:
                    disp = cv2.resize(vis, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
                else:
                    disp = vis
                cv2.imshow(WIN_NAME, disp)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC/q quit
                    break
                elif key in (ord('-'), ord('_')):  # smaller
                    disp_scale = max(0.2, disp_scale - 0.1)
                elif key in (ord('+'), ord('=')):  # larger
                    disp_scale = min(1.5, disp_scale + 0.1)
                elif key == ord('1'):
                    disp_scale = 0.5
                elif key == ord('2'):
                    disp_scale = 0.75
                elif key == ord('3'):
                    disp_scale = 1.0
            else:
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        if SHOW_WINDOW:
            cv2.destroyWindow(WIN_NAME)
        depth_mat.free(); image_mat.free()
        zed.close()

if __name__ == "__main__":
    main()
