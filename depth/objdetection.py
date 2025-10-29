import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

YOLO_WEIGHTS = "yolo11n.onnx"   # keep ONNX, or swap to "yolo11n.pt" if you have CUDA PyTorch
YOLO_IMGSZ   = 288              # 384–640; smaller = faster
YOLO_CONF    = 0.28
DETECT_EVERY = 1                # run YOLO every N frames and rely on ZED tracking in between

def draw_text(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def main():
    cv2.setUseOptimized(True)

    # --- ZED init ---
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE   # <— faster than NEURAL

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED"); return

    zed.enable_positional_tracking(sl.PositionalTrackingParameters())

    od_params = sl.ObjectDetectionParameters()
    od_params.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    od_params.enable_tracking = True
    if zed.enable_object_detection(od_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable OD"); return

    model   = YOLO(YOLO_WEIGHTS, task="detect")
    names   = model.names
    runtime = sl.ObjectDetectionRuntimeParameters()
    objects = sl.Objects()
    left    = sl.Mat()

    cv2.namedWindow("ZED + YOLO (fast)", cv2.WINDOW_NORMAL)
    frame_idx = 0

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left, sl.VIEW.LEFT)
            frame = left.get_data()

            # ZED returns BGRA; YOLO expects 3ch
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # drop alpha (BGRA->BGR)
            frame = np.ascontiguousarray(frame)

            # Run YOLO every Nth frame only
            if frame_idx % DETECT_EVERY == 0:
                res = model.predict(source=frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]
                to_ingest = []
                for b in res.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cls = int(b.cls[0]); conf = float(b.conf[0])

                    cbd = sl.CustomBoxObjectData()
                    cbd.unique_object_id = sl.generate_unique_id()
                    cbd.probability = conf
                    cbd.label = cls
                    # TL, TR, BR, BL with ints
                    cbd.bounding_box_2d = np.array(
                        [[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.int32
                    )
                    cbd.is_grounded = False
                    to_ingest.append(cbd)

                # Provide fresh detections to ZED (tracking will take it from here)
                zed.ingest_custom_box_objects(to_ingest)

            # Retrieve tracked objects (works on non-YOLO frames too)
            zed.retrieve_objects(objects, runtime)

            # Draw
            out = frame  # already BGR
            for obj in objects.object_list:
                tl = obj.bounding_box_2d[0]; br = obj.bounding_box_2d[2]
                cid = int(obj.raw_label)
                cname = (names[cid] if isinstance(names, list) and 0 <= cid < len(names)
                         else (names.get(cid, str(cid)) if hasattr(names, "get") else str(cid)))
                cv2.rectangle(out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0,255,0), 2)
                draw_text(out, f"{cname}  {obj.position[2]:.2f} m",
                          (int(tl[0]), max(0, int(tl[1])-6)))

            cv2.imshow("ZED + YOLO (fast)", out)
            frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
