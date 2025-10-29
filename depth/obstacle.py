# zed_depth_3d_boxes_with_dist_wall_hcov.py
import cv2
import numpy as np
import pyzed.sl as sl

# ---------- Tunables ----------
CAM_RES        = sl.RESOLUTION.HD720
DEPTH_MODE     = sl.DEPTH_MODE.NEURAL
CONF_TH        = 80
Z_MIN, Z_MAX   = 0.25, 8.0
MIN_AREA_FRAC  = 0.001
SUBSAMPLE_STEP = 2
REQUIRE_VALID_RATIO = 0.10      # for 3D validity

# Vertical screen coverage to call "wall"
WALL_VERT_RATIO = 0.90
WALL_RED        = (0, 0, 255)   # BGR
WALL_ALPHA      = 0.35

# Windows
VIEW_WIN = "ZED2i_Depth"
CTRL_WIN = "ZED2i_Controls"

# Sliders
INIT_VMIN = 180
INIT_SMAX = 255

FEET_PER_METER = 3.28084

# ---------- UI helpers ----------
def create_ui():
    cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEW_WIN, 1280, 720)

    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 420, 120)

    ctrl_img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(ctrl_img, "Adjust thresholds here", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow(CTRL_WIN, ctrl_img)

    cv2.createTrackbar("Vmin", CTRL_WIN, INIT_VMIN, 255, lambda v: None)
    cv2.createTrackbar("Smax", CTRL_WIN, INIT_SMAX, 255, lambda v: None)
    cv2.waitKey(1)

def get_intrinsics(zed):
    info = zed.get_camera_information()
    try:
        cam = info.camera_configuration.calibration_parameters.left_cam  # SDK 4.x
    except AttributeError:
        cam = info.calibration_parameters.left_cam                       # SDK 3.x
    return cam.fx, cam.fy, cam.cx, cam.cy

def project_points(pts_xyz, fx, fy, cx, cy):
    X, Y, Z = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    Z = np.where(Z == 0, 1e-6, Z)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)

def draw_box_wireframe(img, corners_2d, color=(0,255,0), thickness=2):
    c2d = np.asarray(corners_2d, dtype=np.float32)
    if not np.all(np.isfinite(c2d)): return
    c2d = c2d.astype(np.int32)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    h, w = img.shape[:2]
    for a,b in edges:
        p1, p2 = c2d[a], c2d[b]
        if (-200 <= p1[0] <= w+200 and -200 <= p1[1] <= h+200 and
            -200 <= p2[0] <= w+200 and -200 <= p2[1] <= h+200):
            cv2.line(img, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)

def draw_rotated_2d_box(img, contour, color=(255,255,0), thickness=2):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [box], True, color, thickness, cv2.LINE_AA)
    return rect, box  # rect center/size/angle, and polygon (4,1,2)

def hud(img, text, y, color=(0,255,0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def put_center_label(img, center_xy, text, bg_color, txt_color=(255,255,255)):
    """Centered text with a filled box, centered at center_xy."""
    x, y = int(center_xy[0]), int(center_xy[1])
    x = max(0, min(img.shape[1]-1, x))
    y = max(0, min(img.shape[0]-1, y))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.7, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x1 = x - (tw // 2) - 6
    y1 = y - (th // 2) - 6
    x2 = x + (tw // 2) + 6
    y2 = y + (th // 2) + 6
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x1+6, y2-6), font, scale, txt_color, thick, cv2.LINE_AA)

def fill_transparent_poly(img, poly_pts, color=WALL_RED, alpha=WALL_ALPHA):
    """poly_pts: (N,2) or (N,1,2) int32. Fills with transparency onto img."""
    overlay = img.copy()
    pts = poly_pts
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)
    pts = pts.astype(np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def vertical_coverage_ratio(poly_pts, frame_h):
    """Return vertical coverage ratio of polygon wrt frame height (0..1)."""
    pts = poly_pts
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)
    ys = pts[:, 1].astype(np.float32)
    # Clip to screen bounds before measuring
    ys = np.clip(ys, 0, frame_h - 1)
    if ys.size == 0:
        return 0.0
    span = float(ys.max() - ys.min())
    return span / float(frame_h) if frame_h > 0 else 0.0

def main():
    # ---- ZED init ----
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = CAM_RES
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = Z_MIN

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = CONF_TH
    try: runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    except AttributeError: pass

    fx, fy, cx, cy = get_intrinsics(zed)

    depth_col = sl.Mat()      # colorized depth (for view & HSV mask)
    xyz_mat   = sl.Mat()      # 3D XYZ in meters
    depth_m   = sl.Mat()      # float depth map in meters

    create_ui()
    print("[q] quit   [m] toggle mask   (sliders in 'ZED2i_Controls')")

    show_mask = False

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        # Frames/measures
        zed.retrieve_image(depth_col, sl.VIEW.DEPTH)   # BGRA8
        bgr = depth_col.get_data()[:, :, :3].copy()

        zed.retrieve_measure(xyz_mat,   sl.MEASURE.XYZ)
        xyz = xyz_mat.get_data().copy()
        if xyz.shape[2] >= 3:
            xyz = xyz[:, :, :3]

        zed.retrieve_measure(depth_m,   sl.MEASURE.DEPTH)
        Zmap = depth_m.get_data().copy()

        # Sliders
        try:
            vmin = cv2.getTrackbarPos("Vmin", CTRL_WIN)
            smax = cv2.getTrackbarPos("Smax", CTRL_WIN)
        except cv2.error:
            create_ui()
            vmin, smax = INIT_VMIN, INIT_SMAX

        # Mask (bright + optionally white)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[:,:,1], hsv[:,:,2]
        mask = ((V >= vmin) & (S <= smax)).astype(np.uint8) * 255

        # Morphology
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        min_area = int(MIN_AREA_FRAC * (w*h))

        out = bgr
        total, n3d, n2d, nwalls = 0, 0, 0, 0

        for c in contours:
            if cv2.contourArea(c) < min_area: continue
            total += 1

            # ROI for this contour
            x, y, bw, bh = cv2.boundingRect(c)
            x2, y2 = x+bw, y+bh
            x, y = max(0,x), max(0,y)
            x2, y2 = min(w,x2), min(h,y2)

            roi_mask = np.zeros((bh, bw), dtype=np.uint8)
            c_shift = c - [x, y]
            cv2.drawContours(roi_mask, [c_shift], -1, 255, thickness=cv2.FILLED)

            # Subsample for XYZ
            ys, xs = np.where(roi_mask[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP] > 0)
            ys = ys*SUBSAMPLE_STEP + y; xs = xs*SUBSAMPLE_STEP + x

            did_3d = False
            dist_ft = None
            center2d_for_label = None

            if xs.size > 0:
                pts = xyz[ys, xs, :]
                finite = np.isfinite(pts).all(axis=1)
                pts = pts[finite]
                if pts.shape[0] > 0:
                    Zv = pts[:,2]
                    inrange = (Zv > Z_MIN) & (Zv < Z_MAX)
                    valid_ratio = float(inrange.sum())/float(len(pts)) if len(pts) else 0.0
                    pts = pts[inrange]

                    if pts.shape[0] >= 30 and valid_ratio >= REQUIRE_VALID_RATIO:
                        # 3D AABB + distance at center
                        mins, maxs = pts.min(axis=0), pts.max(axis=0)
                        center3d   = 0.5*(mins + maxs)
                        dist_m     = float(np.linalg.norm(center3d))
                        dist_ft    = dist_m * FEET_PER_METER

                        # Corners + projection
                        xmin,ymin,zmin = mins; xmax,ymax,zmax = maxs
                        corners_3d = np.array([
                            [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
                            [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]
                        ], dtype=np.float32)
                        c2d = project_points(corners_3d, fx, fy, cx, cy)

                        # Use convex hull for drawing & vertical coverage
                        hull = cv2.convexHull(c2d.astype(np.float32))   # (k,1,2)
                        vratio = vertical_coverage_ratio(hull, h)

                        if vratio >= WALL_VERT_RATIO:
                            fill_transparent_poly(out, hull, WALL_RED, WALL_ALPHA)
                            nwalls += 1
                            center2d_for_label = project_points(center3d.reshape(1,3), fx, fy, cx, cy)[0]
                            put_center_label(out, center2d_for_label, f"WALL — {dist_ft:.1f} ft",
                                             bg_color=WALL_RED, txt_color=(255,255,255))
                        else:
                            draw_box_wireframe(out, c2d, (0,255,0), 2)
                            center2d_for_label = project_points(center3d.reshape(1,3), fx, fy, cx, cy)[0]
                            put_center_label(out, center2d_for_label, f"{dist_ft:.1f} ft",
                                             bg_color=(0,255,0), txt_color=(0,0,0))
                            n3d += 1
                        did_3d = True

            if not did_3d:
                # 2D fallback
                rect, box = draw_rotated_2d_box(out, c, color=(255,255,0), thickness=2)  # cyan
                # Distance: median depth inside contour on Z map
                full_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(full_mask, [c_shift], -1, 255, thickness=cv2.FILLED)
                z_roi = Zmap[y:y2, x:x2]
                z_vals = z_roi[full_mask > 0]
                if z_vals.size > 0:
                    z_vals = z_vals[np.isfinite(z_vals)]
                    z_vals = z_vals[(z_vals > Z_MIN) & (z_vals < Z_MAX)]
                if z_vals.size > 0:
                    dist_m = float(np.median(z_vals))
                    dist_ft = dist_m * FEET_PER_METER

                # Vertical coverage from rotated rect polygon
                vratio = vertical_coverage_ratio(box, h)

                if vratio >= WALL_VERT_RATIO:
                    # Fill polygon and label as WALL
                    fill_transparent_poly(out, box, WALL_RED, WALL_ALPHA)
                    center2d_for_label = rect[0]  # (cx,cy)
                    if dist_ft is not None:
                        put_center_label(out, center2d_for_label, f"WALL — {dist_ft:.1f} ft",
                                         bg_color=WALL_RED, txt_color=(255,255,255))
                    else:
                        put_center_label(out, center2d_for_label, "WALL — N/A",
                                         bg_color=WALL_RED, txt_color=(255,255,255))
                    nwalls += 1
                else:
                    center2d_for_label = rect[0]
                    if dist_ft is not None:
                        put_center_label(out, center2d_for_label, f"{dist_ft:.1f} ft",
                                         bg_color=(255,255,0), txt_color=(0,0,0))
                    else:
                        put_center_label(out, center2d_for_label, "N/A",
                                         bg_color=(255,255,0), txt_color=(0,0,0))
                    n2d += 1

        # HUD
        hud(out, f"Detections: {total} | 3D: {n3d} | 2D: {n2d} | WALLS: {nwalls}", 24)
        hud(out, "WALL criterion: vertical coverage >= 90% of frame height", 48, (255,255,0))

        # Show
        cv2.imshow(VIEW_WIN, out)
        if show_mask:
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): show_mask = not show_mask

    cv2.destroyAllWindows()
    depth_col.free(); xyz_mat.free(); depth_m.free()
    zed.close()

if __name__ == "__main__":
    main()
