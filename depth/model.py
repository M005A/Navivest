# model_hybrid_gui.py — YOLOv8 (CPU) + Depth proposals -> ZED custom boxes -> 3D tracking (GUI)
import time
import numpy as np
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

# ---------------- Config ----------------
# ZED
MAX_RANGE_M    = 2.4384   # 8 ft
RESOLUTION     = sl.RESOLUTION.VGA
FPS            = 30
DEPTH_MODE     = sl.DEPTH_MODE.NEURAL_LIGHT  # try sl.DEPTH_MODE.PERFORMANCE if tight

# YOLO (runs every N frames; CPU)
YOLO_MODEL     = "yolov8n.pt"
DEVICE         = "cpu"           # you have CPU-only torch
IMG_SIZE       = 416             # a bit larger helps on medium/large objects
CONF_TH        = 0.25
IOU_TH         = 0.45
MAX_DET        = 30
DETECT_EVERY   = 3               # run YOLO every N frames
CENTER_CROP_FR = 1.0             # 1.0 disables crop (helps with big objects at edges)

# Depth proposals (cheap, runs every frame)
BIN_INCHES     = 3.0             # depth bin quantization (smaller = finer)
MERGE_ADJ_BINS = 1               # merge +/- bins to reduce banding
BILAT_DIAM     = 7               # bilateral diameter (odd)
BILAT_SIGMA_M  = 0.08            # range sigma in meters
OPEN_K         = 3               # morphology open kernel
CLOSE_K        = 11              # morphology close kernel
JOIN_CLOSE_PX  = 22              # dilate to merge pieces into one large object
MIN_OBJ_AREA   = 700             # min component area (px) to keep

# Floor/wall rejection heuristics (for depth proposals)
MAX_PLANE_STD_M  = 0.025         # very flat = plane
PLANE_AREA_FRAC  = 0.35          # only remove very large flat regions
BORDER_TOUCH_FR  = 0.60          # tall/wide border-hugging => wall
FLOOR_BAND_FRAC  = 0.08          # bottom band's height fraction
FLOOR_MIN_W_FR   = 0.80          # very wide inside floor band => floor

# GUI
SHOW_PREVIEW   = True
BOX_COLOR      = (0, 200, 0)
LABEL_SCALE    = 0.55
LABEL_THICK    = 2
SIDE_THRESH_M  = 0.20            # LEFT/RIGHT cutoff by camera X

# --------------- Utils ----------------
def fmt_feet(m):
    if m is None or not np.isfinite(m) or m <= 0: return "--"
    inches = m * 39.3701
    if inches < 12: return f"{inches:.0f} in"
    ft = int(inches // 12); rem = inches % 12
    return f"{ft} ft {rem:.0f} in" if ft < 12 else f"{ft} ft"

def side_from_x(x_m, t=SIDE_THRESH_M):
    if x_m <= -t: return "LEFT"
    if x_m >=  t: return "RIGHT"
    return "MIDDLE"

def make_kernel(k):
    k = max(1, int(k));  k += (k % 2 == 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def bilateral_depth(d, ddiam, rsigma):
    if ddiam < 3: return d
    return cv2.bilateralFilter(d.astype(np.float32), ddiam, rsigma, rsigma)

def reject_plane_wall_floor(comp_mask, bbox, depth_m):
    H, W = depth_m.shape
    x, y, w, h = bbox
    area = int(comp_mask.sum())
    if area < MIN_OBJ_AREA: return True

    vals = depth_m[comp_mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0: return True

    d_std = float(np.std(vals))
    total = H * W

    if area > PLANE_AREA_FRAC * total and d_std < MAX_PLANE_STD_M:
        return True

    touches_left   = (x == 0)
    touches_right  = (x + w) >= W
    touches_top    = (y == 0)
    touches_bottom = (y + h) >= H
    border_like = False
    if touches_left or touches_right:
        border_like |= (h / H) > BORDER_TOUCH_FR
    if touches_top or touches_bottom:
        border_like |= (w / W) > BORDER_TOUCH_FR
    if border_like: return True

    in_floor_band = (y + h) > int((1.0 - FLOOR_BAND_FRAC) * H)
    if in_floor_band and (w / W) > FLOOR_MIN_W_FR:
        return True

    return False

def depth_to_boxes(depth_m):
    """Return a list of (x1,y1,x2,y2) boxes from depth segmentation, filtered for walls/floor."""
    H, W = depth_m.shape
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= MAX_RANGE_M)
    if not np.any(valid): return []

    d = depth_m.copy().astype(np.float32);  d[~valid] = 0.0
    d = bilateral_depth(d, BILAT_DIAM, BILAT_SIGMA_M)

    inches = d * 39.3701
    bins = np.floor_divide(inches.astype(np.int32), int(max(1, BIN_INCHES)))
    bins[(inches <= 0)] = -1

    occ = [b for b in np.unique(bins) if b >= 0]
    if not occ: return []

    base = np.zeros((H, W), dtype=np.uint8)
    for b in occ:
        m = (bins == b)
        if MERGE_ADJ_BINS > 0:
            for dlt in range(1, MERGE_ADJ_BINS + 1):
                m |= (bins == (b - dlt)) | (bins == (b + dlt))
        base |= m.astype(np.uint8)
    base = (base * 255).astype(np.uint8)

    if OPEN_K  >= 3: base = cv2.morphologyEx(base, cv2.MORPH_OPEN,  make_kernel(OPEN_K),  iterations=1)
    if CLOSE_K >= 3: base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, make_kernel(CLOSE_K), iterations=1)
    if JOIN_CLOSE_PX > 0: base = cv2.dilate(base, make_kernel(JOIN_CLOSE_PX), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        comp_mask = (labels == i)
        if reject_plane_wall_floor(comp_mask, (x, y, w, h), depth_m): continue
        boxes.append((x, y, x + w, y + h))
    return boxes

def nms_xyxy(boxes, scores, iou_th=0.5):
    """Simple NMS for (x1,y1,x2,y2) arrays. Returns indices to keep."""
    if len(boxes) == 0: return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1e-3) * (y2 - y1 + 1e-3)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return keep

def draw_labelled_box(img, bbox2d, text, color=BOX_COLOR):
    bb = np.asarray(bbox2d, dtype=np.int32)
    x1, y1 = bb[:, 0].min(), bb[:, 1].min()
    x2, y2 = bb[:, 0].max(), bb[:, 1].max()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
    tx, ty = x1, max(0, y1 - 6)
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 6, ty + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (tx + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (255, 255, 255), LABEL_THICK, cv2.LINE_AA)

# ---------------- Main ----------------
def main():
    # ZED
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTION
    init.camera_fps = FPS
    init.depth_mode = DEPTH_MODE
    init.coordinate_units = sl.UNIT.METER
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    if zed.enable_positional_tracking(sl.PositionalTrackingParameters()) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking"); return

    od_params = sl.ObjectDetectionParameters()
    od_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    od_params.enable_tracking = True
    if zed.enable_object_detection(od_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable object detection"); return
    od_rt = sl.ObjectDetectionRuntimeParameters()

    # YOLO
    yolo = YOLO(YOLO_MODEL)

    runtime   = sl.RuntimeParameters()
    img_left  = sl.Mat()
    depth_mat = sl.Mat()
    objects   = sl.Objects()

    frame_idx = 0
    t0 = time.time(); fps_smooth = None

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.001); continue

            # Get RGB + Depth
            zed.retrieve_image(img_left,  sl.VIEW.LEFT,  sl.MEM.CPU)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
            frame_bgr = img_left.get_data()[:, :, :3]
            depth     = depth_mat.get_data()
            H, W = frame_bgr.shape[:2]

            # -------- Depth proposals (every frame) --------
            depth_boxes = depth_to_boxes(depth)  # list of (x1,y1,x2,y2)

            # -------- YOLO (every N frames) --------
            yolo_boxes = []
            yolo_scores = []
            if (frame_idx % DETECT_EVERY) == 0:
                src = frame_bgr
                ox = oy = 0
                if 0.5 <= CENTER_CROP_FR < 1.0:
                    cw = int(W * CENTER_CROP_FR); ch = int(H * CENTER_CROP_FR)
                    ox = (W - cw) // 2; oy = (H - ch) // 2
                    src = frame_bgr[oy:oy+ch, ox:ox+cw]

                res = yolo.predict(
                    source=src, imgsz=IMG_SIZE, conf=CONF_TH, iou=IOU_TH,
                    max_det=MAX_DET, agnostic_nms=True, verbose=False, device=DEVICE
                )[0]

                for b in res.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    x1 += ox; x2 += ox; y1 += oy; y2 += oy
                    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
                    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
                    if x2 <= x1 or y2 <= y1: continue
                    yolo_boxes.append((x1, y1, x2, y2))
                    # give YOLO boxes a bit more weight than depth proposals
                    yolo_scores.append(float(b.conf[0]) + 0.2)

            # -------- Union + NMS --------
            union_boxes  = list(yolo_boxes) + list(depth_boxes)
            union_scores = list(yolo_scores) + [0.6] * len(depth_boxes)
            keep_idx = nms_xyxy(union_boxes, union_scores, iou_th=0.5)
            final_boxes = [union_boxes[i] for i in keep_idx]

            # Ingest combined boxes into ZED
            boxes_in = []
            for (x1, y1, x2, y2) in final_boxes:
                box = sl.CustomBoxObjectData()
                box.unique_object_id = sl.generate_unique_id()
                box.probability = 1.0
                box.label = 0
                box.bounding_box_2d = np.array(
                    [[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.int32
                )
                box.is_grounded = False
                boxes_in.append(box)
            if boxes_in:
                zed.ingest_custom_box_objects(boxes_in)

            # Retrieve tracked 3D objects & draw
            view = frame_bgr.copy()
            shown = 0
            if zed.retrieve_objects(objects, od_rt) == sl.ERROR_CODE.SUCCESS:
                for obj in objects.object_list:
                    pos = np.array(obj.position, dtype=np.float32)
                    if not np.all(np.isfinite(pos)): continue
                    dist = float(np.linalg.norm(pos))
                    if dist > MAX_RANGE_M: continue
                    side = side_from_x(float(pos[0]))

                    bb2d = getattr(obj, "bounding_box_2d", None)
                    if bb2d is None or len(bb2d) != 4: continue
                    draw_labelled_box(view, bb2d, f"{fmt_feet(dist)}  [{side}]")
                    shown += 1

            # FPS overlay
            now = time.time()
            dt = now - t0; t0 = now
            fps = 1.0 / dt if dt > 1e-6 else 0.0
            fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
            cv2.putText(view, f"FPS: {fps_smooth:.1f} | Objects: {shown}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(view, f"FPS: {fps_smooth:.1f} | Objects: {shown}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if SHOW_PREVIEW:
                cv2.imshow("Hybrid: YOLO + Depth -> ZED 3D", view)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if SHOW_PREVIEW: cv2.destroyAllWindows()
        depth_mat.free(); img_left.free()
        zed.disable_object_detection()
        zed.disable_positional_tracking()
        zed.close()

if __name__ == "__main__":
    main()
