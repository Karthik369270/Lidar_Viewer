#!/usr/bin/env python3
"""
app.py - LiDAR / camera synchronised viewer.

Serves a browser UI that shows a camera frame and the matching LiDAR
revolution side by side, with motion deskewing switchable on and off.

Run:
    python app.py
    then open http://127.0.0.1:5000

Drop a .pcap (LiDAR), an .mp4 (one lens) and optionally the IMU .csv onto
the page. Without the IMU the viewer still works, but deskewing is
unavailable - there is no velocity to compensate with.

Two things this is built to make visible:

  Motion distortion. A VLP-16 revolution takes 99.5 ms. At 19 km/h the
  vehicle moves 0.53 m during one sweep, so a revolution is not a snapshot
  but a shear - points at different bearings are measured from different
  positions. Projected into an image this looks like a time offset that
  varies frame to frame, which is easy to mistake for a clock problem.
  The deskew toggle shows the size of the effect directly.

  Timing. The camera lag is adjustable from the UI because the value
  measured for the 21 July clip (-3.50 s) rests on a clip start time that
  was inferred rather than read from a manifest. Sliding it and watching
  when moving objects line up is a more direct check than any correlation.
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
import threading
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="static")

# --------------------------------------------------------------------------
# VLP-16 decoding
# --------------------------------------------------------------------------

VERT_DEG = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                     -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float64)
VERT_RAD = np.radians(VERT_DEG)
RING_OF = np.argsort(np.argsort(VERT_DEG))

BLOCK, N_BLOCKS, PACKET = 100, 12, 1206
DIST_RES = 0.002
T_CHANNEL_US, T_SEQUENCE_US = 2.304, 55.296


def read_packets(path):
    """(sensor_times, payloads) for 1206-byte data packets."""
    import dpkt
    times, payloads = [], []
    with open(path, "rb") as fh:
        for _, buf in dpkt.pcap.Reader(fh):
            try:
                data = bytes(dpkt.ethernet.Ethernet(buf).data.data.data)
            except Exception:
                continue
            if len(data) != PACKET:
                continue
            times.append(int.from_bytes(data[1200:1204], "little") / 1e6)
            payloads.append(data)
    return np.asarray(times), payloads


def first_azimuth(p):
    for b in range(N_BLOCKS):
        o = b * BLOCK
        if p[o] == 0xFF and p[o + 1] == 0xEE:
            return int.from_bytes(p[o + 2:o + 4], "little") / 100.0
    return -1.0


def decode_packet(payload, packet_time=0.0):
    """
    (n, 6): x, y, z, intensity, ring, t_seconds.

    Azimuth is interpolated across the two firing sequences in each block.
    Sharing one azimuth between them produces a visible stair-step.

    packet_time must be supplied as the packet's offset from the start of
    the revolution. Left at its default the time column is per-packet and
    spans only ~1.3 ms, which silently makes any motion compensation built
    on it about 75 times too small.
    """
    az = np.empty(N_BLOCKS)
    ok = np.zeros(N_BLOCKS, bool)
    for b in range(N_BLOCKS):
        o = b * BLOCK
        if payload[o] == 0xFF and payload[o + 1] == 0xEE:
            az[b] = int.from_bytes(payload[o + 2:o + 4], "little") / 100.0
            ok[b] = True
    if not ok.any():
        return np.empty((0, 6))

    out = np.empty((N_BLOCKS * 32, 6))
    n = 0
    for b in range(N_BLOCKS):
        if not ok[b]:
            continue
        if b + 1 < N_BLOCKS and ok[b + 1]:
            step = (az[b + 1] - az[b]) % 360.0
        elif b > 0 and ok[b - 1]:
            step = (az[b] - az[b - 1]) % 360.0
        else:
            step = 0.2
        if not (0 < step < 20):
            step = 0.2

        raw = np.frombuffer(payload, np.uint8, 96, b * BLOCK + 4)
        dist = (raw[0::3].astype(np.uint16)
                | (raw[1::3].astype(np.uint16) << 8)) * DIST_RES
        refl = raw[2::3].astype(np.float64)

        for seq in (0, 1):
            for ch in range(16):
                idx = seq * 16 + ch
                d = dist[idx]
                if d <= 0.0:
                    continue
                frac = ((T_CHANNEL_US * ch + T_SEQUENCE_US * seq)
                        / (2 * T_SEQUENCE_US))
                a = np.radians((az[b] + step * frac) % 360.0)
                w = d * np.cos(VERT_RAD[ch])
                out[n] = (w * np.sin(a), w * np.cos(a),
                          d * np.sin(VERT_RAD[ch]), refl[idx], RING_OF[ch],
                          packet_time + (b * 2 * T_SEQUENCE_US
                                         + seq * T_SEQUENCE_US
                                         + ch * T_CHANNEL_US) * 1e-6)
                n += 1
    return out[:n]


# --------------------------------------------------------------------------
# Deskew
# --------------------------------------------------------------------------

def deskew(points, velocity, yaw_rate, ref="end"):
    """
    Remove within-revolution motion distortion.

    Each point carries its time offset within the sweep. Undo the heading
    change and the translation accumulated by that time, referred to the
    end of the revolution (which is the instant the revolution timestamp
    corresponds to when pairing with camera frames).
    """
    out = points.copy()
    if velocity is None and not yaw_rate:
        return out

    dt = points[:, 5]
    tau = dt - (dt.max() if ref == "end"
                else dt.min() if ref == "start"
                else 0.5 * (dt.min() + dt.max()))
    xyz = points[:, :3].copy()

    if yaw_rate:
        a = -yaw_rate * tau
        ca, sa = np.cos(a), np.sin(a)
        x, y = xyz[:, 0].copy(), xyz[:, 1].copy()
        xyz[:, 0] = ca * x - sa * y
        xyz[:, 1] = sa * x + ca * y

    if velocity is not None:
        xyz -= tau[:, None] * np.asarray(velocity, float)[None, :]

    out[:, :3] = xyz
    return out


def project_omni(Xc, K, D, xi, scale=1.0):
    """Camera-frame points -> pixels, unified (omnidirectional) model."""
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    s_ = Xc / np.maximum(n, 1e-9)
    den = s_[:, 2] + xi
    ok = den > 1e-3
    safe = np.where(ok, den, 1.0)
    x, y = s_[:, 0] / safe, s_[:, 1] / safe
    r2 = x * x + y * y
    k1, k2, p1, p2 = D
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return (np.c_[(K[0, 0] * xd + K[0, 2]) * scale,
                  (K[1, 1] * yd + K[1, 2]) * scale], ok)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rev_t = None          # revolution times, lidar clock
        self.rev_p = []            # list of (n,6) arrays
        self.video_path = None
        self.imu = None            # DataFrame
        self.status = "idle"
        self.progress = 0.0
        self.error = None
        self.fps = 30.0
        self.clip_t0 = None        # host time at clip start
        self.yaw_offset_deg = 94.0 # world->body rotation, measured
        self.T = None              # 4x4 lidar -> camera
        self.K = None              # omni intrinsics at calib resolution
        self.D = None
        self.xi = None
        self.calib_w = None
        self.tmp = Path(tempfile.mkdtemp(prefix="lidarview_"))


S = Session()


def body_velocity(t_host, window=0.15):
    """Forward/lateral velocity in the vehicle frame, from the IMU filter.

    filter_vel_* is a WORLD-frame velocity: during straight driving its
    direction differs from yaw_deg by a near-constant offset (measured at
    94 deg on the 21 July clip), which is the signature of a world frame
    rather than a body one. It is rotated here by yaw + that offset.
    filter_twist_* is not usable as a substitute - on the same clip its
    magnitude was a third of the true speed.
    """
    if S.imu is None:
        return None
    df = S.imu
    m = ((df["t_unix"] >= t_host - window) & (df["t_unix"] <= t_host + window))
    if m.sum() < 3:
        return None
    w = df.loc[m]
    yaw = np.radians(w["yaw_deg"].to_numpy(float)
                     + S.yaw_offset_deg)
    vx = w["filter_vel_x"].to_numpy(float)
    vy = w["filter_vel_y"].to_numpy(float)
    good = np.isfinite(yaw) & np.isfinite(vx) & np.isfinite(vy)
    if good.sum() < 3:
        return None
    yaw, vx, vy = yaw[good], vx[good], vy[good]
    fwd = float((vx * np.cos(yaw) + vy * np.sin(yaw)).mean())
    lat = float((-vx * np.sin(yaw) + vy * np.cos(yaw)).mean())
    return np.array([fwd, lat, 0.0])


def yaw_rate(t_host, window=0.15):
    if S.imu is None or "rate_of_turn_deg_s" not in S.imu:
        return 0.0
    df = S.imu
    m = ((df["t_unix"] >= t_host - window) & (df["t_unix"] <= t_host + window))
    if m.sum() < 3:
        return 0.0
    r = df.loc[m, "rate_of_turn_deg_s"].to_numpy(float)
    r = r[np.isfinite(r)]
    return float(np.radians(r.mean())) if len(r) else 0.0


def decode_worker(pcap_path):
    try:
        S.status = "reading packets"
        times, payloads = read_packets(pcap_path)
        if not payloads:
            S.error = "no 1206-byte data packets found in this pcap"
            S.status = "error"
            return

        S.status = "finding revolutions"
        az = np.array([first_azimuth(p) for p in payloads])
        wraps = np.flatnonzero((az[1:] < az[:-1]) & (az[1:] >= 0)
                               & (az[:-1] >= 0)) + 1
        bounds = np.concatenate([[0], wraps, [len(payloads)]]).astype(int)

        S.status = "decoding"
        rev_t, rev_p = [], []
        total = len(bounds) - 1
        for k in range(total):
            lo, hi = bounds[k], bounds[k + 1]
            if hi <= lo:
                continue
            t0 = times[lo]
            pts = np.vstack([decode_packet(p, packet_time=t - t0)
                             for t, p in zip(times[lo:hi], payloads[lo:hi])])
            if len(pts) < 1000:
                continue
            rev_t.append(t0)
            rev_p.append(pts.astype(np.float32))
            S.progress = (k + 1) / total

        S.rev_t = np.asarray(rev_t)
        S.rev_p = rev_p
        S.status = "ready"
    except Exception as e:                                # noqa: BLE001
        S.error = f"{type(e).__name__}: {e}"
        S.status = "error"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    S.reset()
    pcap = request.files.get("pcap")
    video = request.files.get("video")
    imu = request.files.get("imu")

    if pcap is None or video is None:
        return jsonify(error="need both a .pcap and an .mp4"), 400

    pcap_path = S.tmp / "scan.pcap"
    pcap.save(pcap_path)
    S.video_path = S.tmp / "video.mp4"
    video.save(S.video_path)

    if imu is not None and imu.filename:
        try:
            import pandas as pd
            imu_path = S.tmp / "imu.csv"
            imu.save(imu_path)
            S.imu = pd.read_csv(imu_path, comment="#", low_memory=False)
            if "t_unix" in S.imu:
                S.clip_t0 = float(S.imu["t_unix"].iloc[0])
        except Exception as e:                            # noqa: BLE001
            S.imu = None
            print("IMU load failed:", e)

    ext = request.files.get("extrinsic")
    omni = request.files.get("omni")
    if ext is not None and ext.filename:
        try:
            d = json.load(io.BytesIO(ext.read()))
            T = d.get("T_lidar_to_cam") or d.get("T")
            S.T = np.array(T, dtype=float)
        except Exception as e:                            # noqa: BLE001
            print("extrinsic load failed:", e)
    if omni is not None and omni.filename:
        try:
            d = json.load(io.BytesIO(omni.read()))
            S.K = np.array(d["K"], dtype=float)
            dist = np.ravel(np.array(d.get("D", [0, 0, 0, 0]), dtype=float))
            S.D = np.concatenate([dist, np.zeros(4)])[:4]
            S.xi = float(d["xi"])
            sz = d.get("image_size")
            S.calib_w = (int(sz["w"]) if isinstance(sz, dict)
                         else int(sz[0]) if sz else None)
        except Exception as e:                            # noqa: BLE001
            print("omni load failed:", e)

    threading.Thread(target=decode_worker, args=(pcap_path,),
                     daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/status")
def status():
    return jsonify(status=S.status, progress=S.progress, error=S.error,
                   revolutions=len(S.rev_p), has_imu=S.imu is not None,
                   clip_t0=S.clip_t0,
                   has_calib=(S.T is not None and S.K is not None))


@app.route("/api/meta")
def meta():
    if S.status != "ready":
        return jsonify(error="not ready"), 409
    t_rel = (S.rev_t - S.rev_t[0]).tolist()

    speeds, rates = [], []
    for i in range(len(S.rev_p)):
        if S.imu is None or S.clip_t0 is None:
            speeds.append(None); rates.append(None); continue
        th = S.clip_t0 + t_rel[i]
        v = body_velocity(th)
        speeds.append(None if v is None else float(np.linalg.norm(v)))
        rates.append(float(np.degrees(yaw_rate(th))))

    return jsonify(revolutions=len(S.rev_p), t_rel=t_rel,
                   period=float(np.median(np.diff(S.rev_t))),
                   speeds=speeds, yaw_rates=rates,
                   has_imu=S.imu is not None, clip_t0=S.clip_t0,
                   yaw_offset_deg=S.yaw_offset_deg,
                   has_calib=(S.T is not None and S.K is not None),
                   translation_norm=(None if S.T is None
                                     else float(np.linalg.norm(S.T[:3, 3]))))


@app.route("/api/cloud/<int:idx>")
def cloud(idx):
    """Binary: float32 x,y,z,range interleaved. Optional ?deskew=1."""
    if S.status != "ready" or idx >= len(S.rev_p):
        return jsonify(error="unavailable"), 404

    pts = S.rev_p[idx]
    want = request.args.get("deskew") == "1"
    applied, smear, heading = False, 0.0, 0.0

    if want and S.imu is not None and S.clip_t0 is not None:
        th = S.clip_t0 + float(S.rev_t[idx] - S.rev_t[0])
        v, wz = body_velocity(th), yaw_rate(th)
        if v is not None or wz:
            span = float(pts[:, 5].max() - pts[:, 5].min())
            smear = float(np.linalg.norm(v) * span) if v is not None else 0.0
            heading = float(np.degrees(abs(wz) * span))
            pts = deskew(pts.astype(np.float64), v, wz).astype(np.float32)
            applied = True

    step = max(1, len(pts) // 60000)          # keep the browser responsive
    xyz = pts[::step, :3]
    rng = np.linalg.norm(xyz, axis=1).astype(np.float32)
    buf = np.empty((len(xyz), 4), np.float32)
    buf[:, :3] = xyz
    buf[:, 3] = rng

    return Response(
        buf.tobytes(),
        mimetype="application/octet-stream",
        headers={"X-Points": str(len(xyz)),
                 "X-Deskew-Applied": "1" if applied else "0",
                 "X-Smear-M": f"{smear:.4f}",
                 "X-Heading-Deg": f"{heading:.3f}"})


@app.route("/api/project/<int:idx>")
def project(idx):
    """
    Projected pixels for one revolution.

    Binary: float32 u, v, range interleaved, for points that fall inside the
    frame. Query: deskew=0|1, w, h (video pixel size), min_range, max_range.

    The extrinsic this uses came from a board session where every capture sat
    at one depth, so a rotation error and a translation error can cancel at
    that depth and diverge away from it. Expect near-field points to look
    correct and far-field points not to. That is what the range filter is for.
    """
    if S.status != "ready" or idx >= len(S.rev_p):
        return jsonify(error="unavailable"), 404
    if S.T is None or S.K is None:
        return jsonify(error="no calibration loaded"), 409

    w = int(request.args.get("w", 1440))
    h = int(request.args.get("h", 1920))
    lo = float(request.args.get("min_range", 2.0))
    hi = float(request.args.get("max_range", 40.0))
    scale = (w / S.calib_w) if S.calib_w else 1.0

    pts = S.rev_p[idx]
    want = request.args.get("deskew") == "1"
    applied = False
    if want and S.imu is not None and S.clip_t0 is not None:
        th = S.clip_t0 + float(S.rev_t[idx] - S.rev_t[0])
        v, wz = body_velocity(th), yaw_rate(th)
        if v is not None or wz:
            pts = deskew(pts.astype(np.float64), v, wz).astype(np.float32)
            applied = True

    xyz = pts[:, :3].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    keep = (rng > lo) & (rng < hi)
    xyz, rng = xyz[keep], rng[keep]
    if len(xyz) == 0:
        return Response(b"", mimetype="application/octet-stream",
                        headers={"X-Points": "0"})

    Xc = xyz @ S.T[:3, :3].T + S.T[:3, 3]
    uv, ok = project_omni(Xc, S.K, S.D, S.xi, scale=scale)

    # Reject rays from behind the lens. With xi above 1 the unified model
    # maps the whole sphere, so the denominator stays positive for rear-
    # facing rays and they fold into the frame as ghost scan lines that
    # look like real returns. Filter on the ray angle instead.
    nrm = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9)
    theta = np.degrees(np.arccos(np.clip(nrm[:, 2], -1.0, 1.0)))
    fov = float(request.args.get("max_theta", 100.0))

    m = (ok & (theta < fov)
         & (uv[:, 0] >= 0) & (uv[:, 0] < w)
         & (uv[:, 1] >= 0) & (uv[:, 1] < h))

    step = max(1, int(m.sum()) // 40000)
    out = np.empty((int(m.sum()), 3), np.float32)
    out[:, :2] = uv[m]
    out[:, 2] = rng[m]
    out = out[::step]

    return Response(out.tobytes(), mimetype="application/octet-stream",
                    headers={"X-Points": str(len(out)),
                             "X-Deskew-Applied": "1" if applied else "0"})


@app.route("/api/timing/<int:idx>")
def timing(idx):
    """
    The actual LiDAR-to-camera gap for one revolution, broken into parts.

    Three separate things contribute, and they are worth seeing apart:

      lag            the fixed pipeline delay, currently a slider value
      quantisation   the camera runs at 30 fps and the LiDAR at ~10 Hz, so
                     the nearest frame is up to half a frame away. This is
                     unavoidable without interpolation and it varies from
                     revolution to revolution, which on its own can look
                     like a drifting offset.
      smear          the vehicle's own motion during the 99.5 ms sweep. Not
                     a time offset at all, but it produces bearing-dependent
                     misalignment that is easily mistaken for one.
    """
    if S.status != "ready" or idx >= len(S.rev_p):
        return jsonify(error="unavailable"), 404

    lag = float(request.args.get("lag", -3.50))
    fps = float(request.args.get("fps", 30.0))
    t_rel = float(S.rev_t[idx] - S.rev_t[0])
    want = t_rel - lag                       # ideal video time
    frame = int(round(want * fps))
    got = frame / fps
    quant = got - want                       # seconds, signed

    pts = S.rev_p[idx]
    span = float(pts[:, 5].max() - pts[:, 5].min())

    speed = None
    if S.imu is not None and S.clip_t0 is not None:
        v = body_velocity(S.clip_t0 + t_rel)
        if v is not None:
            speed = float(np.linalg.norm(v))

    return jsonify(
        revolution=idx,
        lidar_t_rel=t_rel,
        lidar_clock=float(S.rev_t[idx]),
        video_time_wanted=want,
        video_frame=frame,
        video_time_got=got,
        quantisation_s=quant,
        quantisation_m=(None if speed is None else abs(quant) * speed),
        sweep_span_s=span,
        smear_m=(None if speed is None else speed * span),
        speed_m_s=speed,
        applied_lag_s=lag)


@app.route("/api/video")
def video():
    if S.video_path is None:
        return jsonify(error="no video"), 404
    return send_file(S.video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/config", methods=["POST"])
def config():
    d = request.get_json(force=True)
    if "yaw_offset_deg" in d:
        S.yaw_offset_deg = float(d["yaw_offset_deg"])
    if "clip_t0" in d and d["clip_t0"]:
        S.clip_t0 = float(d["clip_t0"])
    return jsonify(ok=True, yaw_offset_deg=S.yaw_offset_deg,
                   clip_t0=S.clip_t0)


if __name__ == "__main__":
    print("\n  LiDAR / camera viewer   ->  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, threaded=True)
