import sys, time, threading, queue, json
import cv2
import numpy as np
import pyzed.sl as sl
from dataclasses import dataclass, asdict
import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GLib, GstRtspServer, GstRtsp

# ---- HTTP control panel ----
from flask import Flask, request, jsonify, Response

# ==================== Tunables ====================
CAM_RES        = sl.RESOLUTION.HD720
DEPTH_MODE     = sl.DEPTH_MODE.NEURAL
CONF_TH        = 80
Z_MIN, Z_MAX   = 0.25, 8.0
MIN_AREA_FRAC  = 0.001
SUBSAMPLE_STEP = 2
REQUIRE_VALID_RATIO = 0.10

WALL_VERT_RATIO = 0.90
WALL_RED        = (0, 0, 255)
WALL_ALPHA      = 0.35

INIT_VMIN = 135
INIT_SMAX = 255
FEET_PER_METER = 3.28084

# ---- Stream config ----
TARGET_FPS     = 30
BITRATE_KBPS   = 2200
RTP_MTU        = 1200
DOWNSCALE_TO   = None

RTSP_PORT      = "8554"
RTSP_MOUNT     = "/zed"

HTTP_PORT      = 8088

# ==================== Shared state ====================
frame_q = queue.Queue(maxsize=1)   # drop-old to keep latency low
running = True

# ==================== Live Settings (thread-safe) ====================
@dataclass
class Settings:
    # Detection / filtering
    CONF_TH: int = CONF_TH
    Z_MIN: float = Z_MIN
    Z_MAX: float = Z_MAX
    MIN_AREA_FRAC: float = MIN_AREA_FRAC
    SUBSAMPLE_STEP: int = SUBSAMPLE_STEP
    REQUIRE_VALID_RATIO: float = REQUIRE_VALID_RATIO

    # Wall classification + overlay color/alpha
    WALL_VERT_RATIO: float = WALL_VERT_RATIO
    WALL_B: int = WALL_RED[0]
    WALL_G: int = WALL_RED[1]
    WALL_R: int = WALL_RED[2]
    WALL_ALPHA: float = WALL_ALPHA

    # Color mask (HSV)
    VMIN: int = INIT_VMIN
    SMAX: int = INIT_SMAX

settings = Settings()
_settings_lock = threading.Lock()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def snapshot_settings():
    with _settings_lock:
        return Settings(**asdict(settings))

def update_settings_from_dict(d):
    with _settings_lock:
        if 'CONF_TH' in d:
            settings.CONF_TH = int(clamp(d['CONF_TH'], 0, 100))
        if 'Z_MIN' in d:
            settings.Z_MIN = float(clamp(d['Z_MIN'], 0.0, 50.0))
        if 'Z_MAX' in d:
            settings.Z_MAX = float(clamp(d['Z_MAX'], settings.Z_MIN + 0.01, 50.0))
        if 'MIN_AREA_FRAC' in d:
            settings.MIN_AREA_FRAC = float(clamp(d['MIN_AREA_FRAC'], 0.0, 0.5))
        if 'SUBSAMPLE_STEP' in d:
            settings.SUBSAMPLE_STEP = int(clamp(int(d['SUBSAMPLE_STEP']), 1, 8))
        if 'REQUIRE_VALID_RATIO' in d:
            settings.REQUIRE_VALID_RATIO = float(clamp(d['REQUIRE_VALID_RATIO'], 0.0, 1.0))
        if 'WALL_VERT_RATIO' in d:
            settings.WALL_VERT_RATIO = float(clamp(d['WALL_VERT_RATIO'], 0.0, 1.0))
        if 'WALL_B' in d:
            settings.WALL_B = int(clamp(d['WALL_B'], 0, 255))
        if 'WALL_G' in d:
            settings.WALL_G = int(clamp(d['WALL_G'], 0, 255))
        if 'WALL_R' in d:
            settings.WALL_R = int(clamp(d['WALL_R'], 0, 255))
        if 'WALL_ALPHA' in d:
            settings.WALL_ALPHA = float(clamp(d['WALL_ALPHA'], 0.0, 1.0))
        if 'VMIN' in d:
            settings.VMIN = int(clamp(d['VMIN'], 0, 255))
        if 'SMAX' in d:
            settings.SMAX = int(clamp(d['SMAX'], 0, 255))

# ==================== Helpers ====================
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
    return rect, box

def hud(img, text, y, color=(0,255,0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def put_center_label(img, center_xy, text, bg_color, txt_color=(255,255,255)):
    x, y = int(center_xy[0]), int(center_xy[1])
    x = max(0, min(img.shape[1]-1, x)); y = max(0, min(img.shape[0]-1, y))
    font = cv2.FONT_HERSHEY_SIMPLEX; scale, thick = 0.7, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1 = max(0, x - (tw // 2) - 6); y1 = max(0, y - (th // 2) - 6)
    x2 = min(img.shape[1]-1, x + (tw // 2) + 6); y2 = min(img.shape[0]-1, y + (th // 2) + 6)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(img, text, (x1+6, y2-6), font, scale, txt_color, thick, cv2.LINE_AA)

def fill_transparent_poly(img, poly_pts, color, alpha):
    overlay = img.copy()
    pts = poly_pts
    if pts.ndim == 3 and pts.shape[1] == 1: pts = pts.reshape(-1, 2)
    pts = pts.astype(np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def vertical_coverage_ratio(poly_pts, frame_h):
    pts = poly_pts
    if pts.ndim == 3 and pts.shape[1] == 1: pts = pts.reshape(-1, 2)
    ys = np.clip(pts[:, 1].astype(np.float32), 0, frame_h - 1)
    if ys.size == 0: return 0.0
    return float(ys.max() - ys.min()) / float(frame_h) if frame_h > 0 else 0.0

# ==================== ZED processing thread ====================
def zed_loop():
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
    depth_col = sl.Mat(); xyz_mat = sl.Mat(); depth_m = sl.Mat()

    tick_ns = int(1e9 / max(1, TARGET_FPS))
    next_t = time.time_ns()

    while running:
        s = snapshot_settings()
        conf_th         = int(s.CONF_TH)
        zmin, zmax      = float(s.Z_MIN), float(s.Z_MAX)
        min_area_frac   = float(s.MIN_AREA_FRAC)
        substep         = int(s.SUBSAMPLE_STEP)
        valid_ratio_req = float(s.REQUIRE_VALID_RATIO)
        wall_vert_ratio = float(s.WALL_VERT_RATIO)
        wall_color_bgr  = (int(s.WALL_B), int(s.WALL_G), int(s.WALL_R))
        wall_alpha      = float(s.WALL_ALPHA)
        vmin, smax      = int(s.VMIN), int(s.SMAX)

        runtime.confidence_threshold = conf_th

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(depth_col, sl.VIEW.DEPTH)
        bgr = depth_col.get_data()[:, :, :3].copy()

        zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
        xyz = xyz_mat.get_data().copy()
        if xyz.shape[2] >= 3: xyz = xyz[:, :, :3]

        zed.retrieve_measure(depth_m, sl.MEASURE.DEPTH)
        Zmap = depth_m.get_data().copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[:,:,1], hsv[:,:,2]
        mask = ((V >= vmin) & (S <= smax)).astype(np.uint8) * 255
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape[:2]
        min_area = int(min_area_frac * (w*h))

        out = bgr
        total, n3d, n2d, nwalls = 0, 0, 0, 0

        for c in contours:
            if cv2.contourArea(c) < min_area: continue
            total += 1

            x, y, bw, bh = cv2.boundingRect(c)
            x2, y2 = x+bw, y+bh
            x, y = max(0,x), max(0,y)
            x2, y2 = min(w,x2), min(h,y2)

            roi_mask = np.zeros((bh, bw), dtype=np.uint8)
            c_shift = c - [x, y]
            cv2.drawContours(roi_mask, [c_shift], -1, 255, thickness=cv2.FILLED)

            ys, xs = np.where(roi_mask[::substep, ::substep] > 0)
            ys = ys*substep + y; xs = xs*substep + x

            did_3d = False
            dist_ft = None

            if xs.size > 0:
                pts = xyz[ys, xs, :]
                finite = np.isfinite(pts).all(axis=1)
                pts = pts[finite]
                if pts.shape[0] > 0:
                    Zv = pts[:,2]
                    inrange = (Zv > zmin) & (Zv < zmax)
                    valid_ratio = float(inrange.sum())/float(len(pts)) if len(pts) else 0.0
                    pts = pts[inrange]

                    if pts.shape[0] >= 30 and valid_ratio >= valid_ratio_req:
                        mins, maxs = pts.min(axis=0), pts.max(axis=0)
                        center3d   = 0.5*(mins + maxs)
                        dist_m     = float(np.linalg.norm(center3d))
                        dist_ft    = dist_m * FEET_PER_METER

                        xmin,ymin,zmin = mins; xmax,ymax,zmax = maxs
                        corners_3d = np.array([
                            [xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin],
                            [xmin,ymin,zmax],[xmax,ymin,zmax],[xmax,ymax,zmax],[xmin,ymax,zmax]
                        ], dtype=np.float32)
                        c2d = project_points(corners_3d, fx, fy, cx, cy)
                        hull = cv2.convexHull(corners_3d[:, :2].astype(np.float32))  # not used; kept for clarity
                        vratio = vertical_coverage_ratio(cv2.convexHull(c2d.astype(np.float32)), h)

                        if vratio >= wall_vert_ratio:
                            fill_transparent_poly(out, cv2.convexHull(c2d.astype(np.float32)), color=wall_color_bgr, alpha=wall_alpha)
                            center2d = project_points(center3d.reshape(1,3), fx, fy, cx, cy)[0]
                            put_center_label(out, center2d,
                                f"WALL — {dist_ft:.1f} ft" if dist_ft is not None else "WALL — N/A",
                                bg_color=wall_color_bgr, txt_color=(255,255,255))
                            nwalls += 1
                        else:
                            draw_box_wireframe(out, c2d, (0,255,0), 2)
                            center2d = project_points(center3d.reshape(1,3), fx, fy, cx, cy)[0]
                            put_center_label(out, center2d, f"{dist_ft:.1f} ft",
                                             bg_color=(0,255,0), txt_color=(0,0,0))
                            n3d += 1
                        did_3d = True

            if not did_3d:
                rect, box = draw_rotated_2d_box(out, c, color=(255,255,0), thickness=2)
                full_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(full_mask, [c_shift], -1, 255, thickness=cv2.FILLED)
                z_roi = Zmap[y:y2, x:x2]
                z_vals = z_roi[full_mask > 0]
                if z_vals.size > 0:
                    z_vals = z_vals[np.isfinite(z_vals)]
                    z_vals = z_vals[(z_vals > zmin) & (z_vals < zmax)]
                if z_vals.size > 0:
                    dist_m = float(np.median(z_vals)); dist_ft = dist_m * FEET_PER_METER

                vratio = vertical_coverage_ratio(box, h)
                if vratio >= wall_vert_ratio:
                    fill_transparent_poly(out, box, color=wall_color_bgr, alpha=wall_alpha)
                    center2d = rect[0]
                    put_center_label(out, center2d,
                        f"WALL {dist_ft:.1f} ft" if dist_ft is not None else "WALL — N/A",
                        bg_color=wall_color_bgr, txt_color=(255,255,255))
                    nwalls += 1
                else:
                    center2d = rect[0]
                    put_center_label(out, center2d,
                        f"{dist_ft:.1f} ft" if dist_ft is not None else "N/A",
                        bg_color=(255,255,0), txt_color=(0,0,0))
                    n2d += 1

        hud(out, f"Detections: {total} | 3D: {n3d} | 2D: {n2d} | WALLS: {nwalls}", 24)
        hud(out, f"WALL criterion: vertical coverage >= {int(wall_vert_ratio*100)}%", 48, (255,255,0))

        if DOWNSCALE_TO is not None:
            out = cv2.resize(out, DOWNSCALE_TO)

        now = time.time_ns()
        if now < next_t: time.sleep((next_t - now) / 1e9)
        next_t = time.time_ns() + tick_ns

        if not frame_q.empty():
            try: frame_q.get_nowait()
            except queue.Empty: pass
        frame_q.put(out)

    depth_col.free(); xyz_mat.free(); depth_m.free()
    zed.close()

# ==================== RTSP factory (single, minimal, correct caps) ====================
class AppSrcFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps):
        super().__init__()
        self.w = width; self.h = height; self.fps = fps

        enc = (
          "videoconvert ! video/x-raw,format=I420 "
          f"! x264enc tune=zerolatency speed-preset=ultrafast bitrate={BITRATE_KBPS} "
          "bframes=0 key-int-max=30 byte-stream=true "
          "! h264parse "
          f"! rtph264pay name=pay0 pt=96 mtu={RTP_MTU}"
        )

        self.launch_str = (
            f'appsrc name=src is-live=true do-timestamp=true format=time '
            f'caps=video/x-raw,format=BGR,width={self.w},height={self.h},framerate={self.fps}/1 '
            f'! {enc}'
        )
        self.set_launch(self.launch_str)
        self.set_latency(0)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsrc.set_property("block", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)

        self.frame_count = 0
        self.frame_duration_ns = int(1e9 // self.fps)
        self.push_thread = threading.Thread(target=self.push_frames, daemon=True)
        self.push_thread.start()

    def push_frames(self):
        while running:
            frame = frame_q.get()
            if frame is None:
                time.sleep(0.001); continue

            h, w = frame.shape[:2]
            if (w != self.w) or (h != self.h):
                frame = cv2.resize(frame, (self.w, self.h))

            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            pts = self.frame_count * self.frame_duration_ns
            buf.pts = pts; buf.dts = pts; buf.duration = self.frame_duration_ns
            self.frame_count += 1

            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                pass

# ==================== HTTP Control Panel ====================
app = Flask(__name__)

@app.route("/")
def index():
    # Double braces {{ }} inside this f-string are intentional.
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>ZED RTSP Controls</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body{{{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:760px;margin:24px auto;padding:0 12px}}}}
 h1{{{{font-size:20px;margin:0 0 8px}}}}
 .row{{{{display:grid;grid-template-columns:220px 1fr 70px;gap:10px;align-items:center;margin:8px 0}}}}
 input[type=range]{{{{width:100%}}}}
 code{{{{background:#f2f2f2;padding:2px 6px;border-radius:6px}}}}
 .card{{{{border:1px solid #eee;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}}}
 .tiny{{{{opacity:.7;font-size:12px}}}}
 .swatch{{{{width:26px;height:18px;display:inline-block;border-radius:4px;border:1px solid #ccc;vertical-align:middle;margin-left:6px}}}}
</style>
</head>
<body>
  <h1>ZED Stream Controls</h1>
  <div class="card tiny">Open your stream with
    <code>ffplay -rtsp_transport tcp rtsp://&lt;JETSON_IP&gt;:{RTSP_PORT}{RTSP_MOUNT}</code>
  </div>
  <div id="controls"></div>
  <script>
  const spec = [
    ["CONF_TH", 0,100,1,"Confidence threshold"],
    ["Z_MIN", 0,10,0.01,"Z min (m)"],
    ["Z_MAX", 0,50,0.01,"Z max (m)"],
    ["MIN_AREA_FRAC", 0,0.1,0.0005,"Min area (fraction of frame)"],
    ["SUBSAMPLE_STEP", 1,8,1,"Subsample step"],
    ["REQUIRE_VALID_RATIO", 0,1,0.01,"Valid ratio (3D)"],
    ["WALL_VERT_RATIO", 0,1,0.01,"Wall vertical coverage"],
    ["VMIN", 0,255,1,"HSV V min"],
    ["SMAX", 0,255,1,"HSV S max"],
    ["WALL_R", 0,255,1,"Wall R"],
    ["WALL_G", 0,255,1,"Wall G"],
    ["WALL_B", 0,255,1,"Wall B"],
    ["WALL_ALPHA", 0,1,0.01,"Wall alpha"]
  ];
  const container = document.getElementById('controls');

  function row(label, key, min, max, step, val){{ 
    const id = 's_'+key;
    const div = document.createElement('div');
    div.className='card';
    div.innerHTML = `
      <div class="row">
        <div><b>${{label}}</b></div>
        <div><input type="range" id="${{id}}" min="${{min}}" max="${{max}}" step="${{step}}" value="${{val}}"></div>
        <div><span id="${{id}}_val"></span></div>
      </div>`;
    container.appendChild(div);
    const slider = div.querySelector('#'+id);
    const out = div.querySelector('#'+id+'_val');
    function fmtVal(v){{ return (Number(v).toFixed((step<1)?2:0)); }}
    function updateBadge(){{ 
      out.textContent = fmtVal(slider.value);
      if (key === "WALL_R" || key === "WALL_G" || key === "WALL_B") {{
        const r = document.getElementById('s_WALL_R')?.value || 0;
        const g = document.getElementById('s_WALL_G')?.value || 0;
        const b = document.getElementById('s_WALL_B')?.value || 0;
        out.innerHTML = fmtVal(slider.value) + ' <span class="swatch" style="background: rgb('+r+','+g+','+b+')"></span>';
      }}
    }}
    updateBadge();

    let debounce;
    slider.addEventListener('input', ()=>{{ 
      updateBadge();
      clearTimeout(debounce);
      debounce = setTimeout(()=>{{ 
        const payload = {{ [key]: Number(slider.value) }};
        fetch('/api/settings', {{
          method:'POST',
          headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify(payload)
        }});
      }}, 80);
    }});
  }}

  async function boot(){{ 
    const s = await (await fetch('/api/settings')).json();
    for (const [key,min,max,step,label] of spec){{ 
      row(label, key, min, max, step, s[key]);
    }}
  }}
  boot();
  </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        s = snapshot_settings()
        return jsonify(asdict(s))
    else:
        try:
            data = request.get_json(force=True, silent=True) or {}
            update_settings_from_dict(data)
            return jsonify({"ok": True, "settings": asdict(snapshot_settings())})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

def http_loop():
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)

def main():
    global running

    # Start ZED processing
    zt = threading.Thread(target=zed_loop, daemon=True); zt.start()

    # Start HTTP control panel
    ht = threading.Thread(target=http_loop, daemon=True); ht.start()

    print("Waiting for first frame...")
    first = None
    for _ in range(100):
        try: first = frame_q.get(timeout=0.1); break
        except queue.Empty: pass
    if first is None:
        print("No frames received. Exiting."); running = False; return

    h, w = first.shape[:2]
    if DOWNSCALE_TO is not None:
        w, h = DOWNSCALE_TO
    fps = TARGET_FPS

    frame_q.put(first)  # put back

    Gst.init(None)

    if Gst.ElementFactory.find("x264enc") is None:
        print("ERROR: x264enc not found. Install gstreamer1.0-plugins-ugly.")
        running = False; return

    server = GstRtspServer.RTSPServer()
    server.props.service = RTSP_PORT
    mounts = server.get_mount_points()

    factory = AppSrcFactory(width=w, height=h, fps=fps)
    factory.set_shared(True)
    # you can allow both TCP|UDP; TCP is most robust:
    factory.set_protocols(GstRtsp.RTSPLowerTrans.TCP)

    mounts.add_factory(RTSP_MOUNT, factory)
    server.attach(None)

    print(f"RTSP ready at rtsp://0.0.0.0:{RTSP_PORT}{RTSP_MOUNT}")
    print(f"Open controls at:  http://0.0.0.0:{HTTP_PORT}/  (use the Jetson IP)")

    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        time.sleep(0.5)

if __name__ == "__main__":
    main()
