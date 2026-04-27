#!/usr/bin/env python3
"""
predator_vision.py
==================

Predator-vision motion HUD for a handheld walking camera.

Imagine you're walking through a forest with a camera in your hand,
streaming to a laptop in your backpack, watching the output on a small
screen clipped to your chest. Your own walking dominates every frame —
trees swing across the view, the ground bobs up and down — but you want
to see only what's *actually* moving in the world: a deer in the trees,
a drone in the sky, a person crossing 80 meters out.

That's what this script does. It runs three filters in sequence, each
fixing what the one before couldn't:

  1. **Spatial stabilization** — Shi-Tomasi corners + Lucas-Kanade
     optical flow + RANSAC affine fit cancels camera shake and panning.
     RANSAC's bouncer keeps the static background and discards points
     that moved on their own (the targets you want).

  2. **Time-shifted differencing** — compares the current frame to one
     from K frames ago instead of just the previous frame. Slow movers
     (a deer 80 m away moves <1 px/frame; over 8 frames it moves ~6 px,
     enough to register clearly) become visible.

  3. **Adaptive noise floor + morphology** — what survives stabilization
     still has a low-amplitude pepper from sub-pixel registration error
     and parallax (near vs far). A running median floor + open/close
     morphology reduces it to zero almost everywhere except where there
     are real coherent moving blobs.

The output is a fullscreen HUD: live camera on the left, motion mask on
the right, blob overlays on the live view in a Predator-thermal palette.
A footer shows FPS, RANSAC inliers, blob count, and the current mode.

Hotkeys (press while focused on the window):
  q / ESC   quit
  SPACE     pause/resume
  m         cycle output mode: SIDE_BY_SIDE → OVERLAY → MOTION_ONLY → RAW
  s         toggle stabilization on/off  (compare with-vs-without live)
  +  /  -   increase / decrease time-shift K (1..30)
  [  /  ]   decrease / increase noise floor (more / less sensitive)
  r         start/stop recording an MP4 of whatever's on screen
  h         toggle this help overlay

Usage:
  python predator_vision.py                 # default webcam (index 0)
  python predator_vision.py --source 1      # different camera index
  python predator_vision.py --source rtsp://10.0.0.5:554/stream
  python predator_vision.py --source walk.mp4
  python predator_vision.py --width 1280 --height 720
  python predator_vision.py --record session.mp4   # start recording on launch

No external dependencies beyond opencv-python and numpy.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ─── tunables ────────────────────────────────────────────────────────────
# All of these are interactively adjustable at runtime; values here are
# the starting points. Picked to work well on a 640×480 @ 30 fps webcam
# while walking outdoors. If you have a 1080p stream and a beefy machine,
# bump PROCESS_W / PROCESS_H — ray-cast time scales with pixel count, but
# stabilization quality also improves with more corners.

@dataclass
class Config:
    # --- input ---
    source: object = 0                    # int / str passed to VideoCapture
    request_w: int = 1280                 # ask camera for this resolution
    request_h: int = 720
    process_w: int = 640                  # downsample TO this for processing
    process_h: int = 360                  # (display is upsampled back to native)

    # --- temporal differencing ---
    time_shift: int = 8                   # K, frames between A and B
    time_shift_min: int = 1
    time_shift_max: int = 30

    # --- stabilization (Shi-Tomasi + LK + RANSAC) ---
    stabilize: bool = True
    feature_params: dict = field(default_factory=lambda: dict(
        maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3))
    lk_winsize: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    ransac_min_inliers: int = 12          # below this, fall back unstabilized

    # --- noise floor & blob filtering ---
    motion_floor: int = 18                # zero out diffs below this
    motion_floor_min: int = 5
    motion_floor_max: int = 80
    morph_open: int = 3                   # erode kernel size (rejects pepper)
    morph_close: int = 7                  # dilate kernel size (joins blobs)
    blob_min_area_px: int = 25            # final reject threshold on contour area

    # --- display ---
    mode: str = "SIDE_BY_SIDE"            # SIDE_BY_SIDE | OVERLAY | MOTION_ONLY | RAW
    show_help: bool = True                # show hotkey legend on screen
    window_name: str = "predator-vision"

    # --- recording ---
    record_path: Optional[str] = None
    record_fps: int = 24


# ─── stabilizer ──────────────────────────────────────────────────────────
# Tracks corners between two frames and solves for the rigid 2D
# transform that aligns the static background. RANSAC throws out the
# corners that don't fit the consensus motion — those are the targets.

class Stabilizer:
    """Aligns frame B onto frame A's coordinate system using
    Shi-Tomasi + Lucas-Kanade + RANSAC partial-affine fit.

    Returns the warped B. Sets `last_inliers` and `last_M` for HUD."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_inliers: int = 0
        self.last_M: Optional[np.ndarray] = None
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             10, 0.03)

    def align(self, gray_A: np.ndarray, gray_B: np.ndarray) -> np.ndarray:
        # 1) corners in A
        p0 = cv2.goodFeaturesToTrack(gray_A, mask=None,
                                     **self.cfg.feature_params)
        if p0 is None or len(p0) < 6:
            self.last_M = None; self.last_inliers = 0
            return gray_B

        # 2) track each corner into B
        p1, status, _err = cv2.calcOpticalFlowPyrLK(
            gray_A, gray_B, p0, None,
            winSize=self.cfg.lk_winsize,
            maxLevel=self.cfg.lk_max_level,
            criteria=self._lk_criteria)
        if p1 is None or status is None:
            self.last_M = None; self.last_inliers = 0
            return gray_B
        good_old = p0[status.flatten() == 1]
        good_new = p1[status.flatten() == 1]
        if len(good_new) < 6:
            self.last_M = None; self.last_inliers = 0
            return gray_B

        # 3) RANSAC partial-affine: source=B-points, dest=A-points
        # i.e. "find the warp that takes B to A"
        M, inliers = cv2.estimateAffinePartial2D(
            good_new, good_old, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
        n_in = int(inliers.sum()) if inliers is not None else 0
        self.last_M = M; self.last_inliers = n_in

        if M is None or n_in < self.cfg.ransac_min_inliers:
            return gray_B

        # 4) warp B to align with A
        h, w = gray_A.shape
        return cv2.warpAffine(gray_B, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


# ─── motion extractor ────────────────────────────────────────────────────
# Stabilize, time-shift diff, threshold, denoise, find blobs.

class MotionExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stab = Stabilizer(cfg)
        self._gray_history: deque = deque(maxlen=cfg.time_shift + 1)
        self._open_k = self._kernel(cfg.morph_open)
        self._close_k = self._kernel(cfg.morph_close)

    @staticmethod
    def _kernel(sz: int) -> np.ndarray:
        sz = max(1, sz)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))

    def update_kernels(self):
        self._open_k = self._kernel(self.cfg.morph_open)
        self._close_k = self._kernel(self.cfg.morph_close)

    def push(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Returns dict with keys mask, blobs, inliers — or None until
        the frame buffer is primed."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adjust deque depth if K changed
        target = self.cfg.time_shift + 1
        if self._gray_history.maxlen != target:
            self._gray_history = deque(self._gray_history, maxlen=target)

        self._gray_history.append(gray)
        if len(self._gray_history) < target:
            return None

        gray_old = self._gray_history[0]   # K frames ago
        gray_new = self._gray_history[-1]  # current

        if self.cfg.stabilize:
            # Warp the OLD frame onto the NEW one's coordinate system.
            # That way overlays line up with the user's current view.
            ref = self.stab.align(gray_new, gray_old)
            inliers = self.stab.last_inliers
        else:
            ref = gray_old
            inliers = -1   # sentinel: stabilization off

        # Absolute difference, threshold, morphology.
        diff = cv2.absdiff(gray_new, ref)
        _, mask = cv2.threshold(diff, self.cfg.motion_floor, 255,
                                cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k)

        # Connected components → blobs (x, y, w, h, area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.cfg.blob_min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2; cy = y + h // 2
            blobs.append(dict(x=x, y=y, w=w, h=h, area=area, cx=cx, cy=cy))
        blobs.sort(key=lambda b: -b["area"])

        return dict(mask=mask, blobs=blobs, inliers=inliers,
                    diff_raw=diff)


# ─── HUD ─────────────────────────────────────────────────────────────────
# Composites the live frame, the motion mask, and HUD text into the
# final image to display / record.

# Predator thermal palette: cool dark→cyan→hot orange→white
def _build_predator_lut() -> np.ndarray:
    stops = np.array([
        [  0,   0,   0],     # 0:    pure black
        [ 12,   0,  60],     # 32:   deep purple
        [120,   0, 100],     # 64:   magenta
        [255,  20,  20],     # 128:  red-orange
        [255, 160,   0],     # 192:  amber
        [255, 255, 200],     # 255:  white-yellow
    ], dtype=np.float32)
    xs = np.linspace(0, 255, len(stops))
    lut = np.empty((256, 1, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, 0, ch] = np.interp(np.arange(256), xs, stops[:, ch]).astype(np.uint8)
    # cv2 expects BGR
    return lut[..., ::-1].copy()


_PREDATOR_LUT = _build_predator_lut()


def colorize_motion(diff_uint8: np.ndarray) -> np.ndarray:
    """Predator-thermal colorization of a (H, W) uint8 motion image."""
    return cv2.applyColorMap(diff_uint8, _PREDATOR_LUT)


def draw_blobs(canvas: np.ndarray, blobs: list,
               scale: Tuple[float, float] = (1.0, 1.0)) -> None:
    """Draw bounding boxes + crosshairs on canvas. scale lets us draw
    blobs detected at process resolution onto a display at full res."""
    sx, sy = scale
    for i, b in enumerate(blobs):
        x = int(b["x"] * sx); y = int(b["y"] * sy)
        w = int(b["w"] * sx); h = int(b["h"] * sy)
        cx = int(b["cx"] * sx); cy = int(b["cy"] * sy)
        # priority: bigger blobs hotter
        col = (0, 200, 255) if i == 0 else (0, 140, 220)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2, cv2.LINE_AA)
        # crosshair
        cv2.drawMarker(canvas, (cx, cy), col, cv2.MARKER_CROSS, 18, 2,
                       cv2.LINE_AA)
        # label
        lbl = f"#{i}  {b['area']}px"
        cv2.putText(canvas, lbl, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def draw_text_block(img: np.ndarray, lines: list, anchor: str = "tl",
                    pad: int = 8, fg=(220, 220, 220),
                    bg=(0, 0, 0)) -> None:
    """Draws a multi-line text block with semi-transparent background."""
    if not lines:
        return
    h, w = img.shape[:2]
    line_h = 22
    block_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                  for s in lines) + 2 * pad
    block_h = line_h * len(lines) + 2 * pad
    if anchor == "tl":   x0, y0 = 10, 10
    elif anchor == "tr": x0, y0 = w - block_w - 10, 10
    elif anchor == "bl": x0, y0 = 10, h - block_h - 10
    else:                x0, y0 = w - block_w - 10, h - block_h - 10
    # translucent bg
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + block_w, y0 + block_h),
                  bg, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + block_w, y0 + block_h),
                  (80, 80, 80), 1)
    for i, s in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 6
        cv2.putText(img, s, (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)


def compose(frame_bgr: np.ndarray, motion: dict, cfg: Config,
            stats: dict) -> np.ndarray:
    """Build the final display image based on cfg.mode."""
    H, W = frame_bgr.shape[:2]
    mask = motion["mask"]
    diff_raw = motion["diff_raw"]
    blobs = motion["blobs"]

    # Process resolution can differ from display; compute scale factors
    pH, pW = mask.shape
    sx = W / pW; sy = H / pH

    # Pre-build the colorized motion view at PROCESS resolution
    motion_color = colorize_motion(diff_raw)
    motion_color_full = cv2.resize(motion_color, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
    # Draw blobs onto the motion view too
    draw_blobs(motion_color_full, blobs, scale=(sx, sy))

    if cfg.mode == "RAW":
        canvas = frame_bgr.copy()

    elif cfg.mode == "MOTION_ONLY":
        canvas = motion_color_full

    elif cfg.mode == "OVERLAY":
        # Predator-vision: live frame with motion blobs glowing on top.
        canvas = frame_bgr.copy()
        # Translucent thermal motion blended on
        mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        # only blend where mask is hot
        hot = mask_full > 0
        glow = cv2.addWeighted(canvas, 0.5, motion_color_full, 0.7, 0)
        canvas[hot] = glow[hot]
        draw_blobs(canvas, blobs, scale=(sx, sy))

    else:  # SIDE_BY_SIDE — default
        # Left: live + bbox overlays. Right: thermal motion view.
        left = frame_bgr.copy()
        draw_blobs(left, blobs, scale=(sx, sy))
        right = motion_color_full
        canvas = np.hstack([left, right])
        # Divider
        cv2.line(canvas, (W, 0), (W, H), (60, 60, 60), 1)

    # ── HUD text overlays ──────────────────────────────────────────
    inliers = motion["inliers"]
    stab_str = (f"STAB ON  inliers={inliers:3d}"
                if inliers >= 0 else "STAB OFF")
    hud = [
        f"PREDATOR-VISION   mode={cfg.mode}",
        f"{stab_str}   K={cfg.time_shift:2d}   floor={cfg.motion_floor}",
        f"FPS {stats.get('fps', 0):5.1f}   blobs={len(blobs):2d}   "
        f"proc={pW}x{pH}",
    ]
    if stats.get("recording"):
        hud.append(f"REC  {stats['rec_seconds']:5.1f}s   "
                   f"{stats.get('rec_path','')}")
    if stats.get("paused"):
        hud.append("** PAUSED **")
    draw_text_block(canvas, hud, anchor="tl")

    if cfg.show_help:
        help_lines = [
            "q/ESC quit   SPC pause   m mode",
            "s stabilize   +/- shift K   [/] floor",
            "r record     h hide help",
        ]
        draw_text_block(canvas, help_lines, anchor="br",
                        fg=(180, 220, 255))

    return canvas


# ─── camera reader ───────────────────────────────────────────────────────

def open_capture(cfg: Config) -> cv2.VideoCapture:
    src = cfg.source
    # Allow numeric strings ("0", "1") to be treated as device indices
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open source: {cfg.source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.request_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.request_h)
    # MJPG often gives higher FPS on USB webcams; harmless for files.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


# ─── main loop ───────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    cap = open_capture(cfg)
    extractor = MotionExtractor(cfg)

    # Resolve actual capture dimensions for the writer
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.request_w
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.request_h
    print(f"[predator] source={cfg.source!r}  "
          f"capture={actual_w}x{actual_h}  process={cfg.process_w}x{cfg.process_h}")

    # If the user passed --record, set up the writer lazily on first
    # composed frame (we need its actual dimensions for SIDE_BY_SIDE)
    writer: Optional[cv2.VideoWriter] = None
    rec_started_at: Optional[float] = None
    rec_path: Optional[str] = cfg.record_path

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_name,
                     min(1600, actual_w * 2 if cfg.mode == "SIDE_BY_SIDE"
                         else actual_w), actual_h)

    paused = False
    last_frame_bgr: Optional[np.ndarray] = None
    fps_smoothed = 0.0
    t_prev = time.time()
    frame_idx = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[predator] source ended")
                break
            last_frame_bgr = frame
            frame_idx += 1
        else:
            frame = last_frame_bgr
            if frame is None:
                # paused before we got any frames; just spin
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    break
                continue

        # Process at reduced resolution
        small = cv2.resize(frame, (cfg.process_w, cfg.process_h),
                           interpolation=cv2.INTER_AREA)
        motion = extractor.push(small)

        # FPS smoothing (EMA)
        t_now = time.time()
        dt = max(1e-3, t_now - t_prev); t_prev = t_now
        inst_fps = 1.0 / dt
        fps_smoothed = (0.9 * fps_smoothed + 0.1 * inst_fps
                        if fps_smoothed else inst_fps)

        if motion is None:
            # Buffer still priming; just show the live frame
            display = frame.copy()
            draw_text_block(display,
                            [f"priming time-shift buffer "
                             f"({frame_idx}/{cfg.time_shift + 1})"],
                            anchor="tl")
        else:
            stats = dict(fps=fps_smoothed,
                         paused=paused,
                         recording=writer is not None,
                         rec_seconds=(time.time() - rec_started_at
                                      if rec_started_at else 0.0),
                         rec_path=rec_path or "")
            display = compose(frame, motion, cfg, stats)

        # Lazy-create the writer once we know the display size
        if rec_path and writer is None:
            h, w = display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(rec_path, fourcc, cfg.record_fps, (w, h))
            rec_started_at = time.time()
            print(f"[predator] recording to {rec_path} ({w}x{h}@{cfg.record_fps})")

        if writer is not None:
            writer.write(display)

        cv2.imshow(cfg.window_name, display)

        # ── handle keys ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):                   # quit
            break
        elif key == ord(" "):                       # pause
            paused = not paused
        elif key == ord("m"):                       # cycle modes
            modes = ["SIDE_BY_SIDE", "OVERLAY", "MOTION_ONLY", "RAW"]
            cfg.mode = modes[(modes.index(cfg.mode) + 1) % len(modes)]
        elif key == ord("s"):                       # toggle stabilization
            cfg.stabilize = not cfg.stabilize
        elif key in (ord("+"), ord("=")):           # K up
            cfg.time_shift = min(cfg.time_shift_max, cfg.time_shift + 1)
        elif key == ord("-"):                       # K down
            cfg.time_shift = max(cfg.time_shift_min, cfg.time_shift - 1)
        elif key == ord("]"):                       # less sensitive
            cfg.motion_floor = min(cfg.motion_floor_max, cfg.motion_floor + 2)
        elif key == ord("["):                       # more sensitive
            cfg.motion_floor = max(cfg.motion_floor_min, cfg.motion_floor - 2)
        elif key == ord("h"):                       # toggle help
            cfg.show_help = not cfg.show_help
        elif key == ord("r"):                       # toggle recording
            if writer is None:
                rec_path = (cfg.record_path
                            or f"predator_{datetime.now():%Y%m%d_%H%M%S}.mp4")
                # Lazily created above on next frame
            else:
                writer.release(); writer = None
                print(f"[predator] stopped recording → {rec_path}")
                rec_path = None; rec_started_at = None

    # ── shutdown ────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[predator] saved recording → {rec_path}")
    cv2.destroyAllWindows()


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 2)[1],   # one-paragraph summary
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=0,
                    help="camera index, RTSP URL, or video file path")
    ap.add_argument("--width",  type=int, default=1280, dest="request_w")
    ap.add_argument("--height", type=int, default=720,  dest="request_h")
    ap.add_argument("--proc-width",  type=int, default=640, dest="process_w")
    ap.add_argument("--proc-height", type=int, default=360, dest="process_h")
    ap.add_argument("--shift", type=int, default=8, dest="time_shift")
    ap.add_argument("--floor", type=int, default=18, dest="motion_floor")
    ap.add_argument("--no-stab", dest="stabilize", action="store_false")
    ap.add_argument("--mode", default="SIDE_BY_SIDE",
                    choices=("SIDE_BY_SIDE", "OVERLAY", "MOTION_ONLY", "RAW"))
    ap.add_argument("--record", default=None, dest="record_path",
                    help="path to MP4; recording starts immediately")
    args = ap.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\n[predator] interrupted")
        sys.exit(0)
