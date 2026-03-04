"""
BFMC Hybrid Pilot - Version 2 (with Calibration)
================================================
Modifications:
  - ADDED: Calibration warmup phase to let camera exposure settle.
  - TWEAK: Adjusted offsets to keep car away from the right lane edge.
  - TWEAK: Increased right-edge guard distance for safety.
  - FIX: Added a 5-second auto-start to calibration for headless setups.
  - FIX: Added terminal warnings when car stops due to lost lane.
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys

# ---------------------------------------------------------------------------
# Serial handler - graceful fallback
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, "..")          # allow running from the sub-folder
    from serial_handler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    print("WARNING: serial_handler not found - running in simulation mode")

    class STM32_SerialHandler:
        def connect(self):      return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self):   pass

# ---------------------------------------------------------------------------
# Camera - graceful fallback
# ---------------------------------------------------------------------------
_CAM_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    print("WARNING: picamera2 not found - camera disabled")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# PHYSICAL CONSTANTS  (measure your car carefully)
# ===========================================================================
WHEELBASE_M          = 0.23    # front-to-rear axle distance (m)
LANE_WIDTH_M         = 0.35    # one-lane physical width (m)

# ===========================================================================
# CAMERA - Bird's Eye View calibration
# SRC: 4 points in raw camera image  [TL, TR, BL, BR]
# DST: corresponding points in BEV   [TL, TR, BL, BR]
# ===========================================================================
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# RIGHT-LANE OFFSET
# Target pixel = image_centre + RIGHT_LANE_OFFSET_PX
# Positive = shift right. Reduced to keep away from right lane divider.
# ===========================================================================
RIGHT_LANE_OFFSET_PX = 30  # Was 70. Decreased to avoid right edge.

# Offset applied only in DUAL mode (both lines found).
# Negative value gives a slight left bias to avoid the right edge.
DUAL_OFFSET_PX = -10 

# ===========================================================================
# TIMING
# ===========================================================================
TARGET_FPS    = 30
FRAME_PERIOD  = 1.0 / TARGET_FPS   # seconds

# ===========================================================================
# LOST-LANE GRACE PERIOD
# Car holds last steering and slows down for this many frames before stopping.
# ===========================================================================
LOST_GRACE_FRAMES = 8


# ===========================================================================
# HYBRID LANE TRACKER
# ===========================================================================
class HybridLaneTracker:

    NWINDOWS         = 9
    SW_MARGIN        = 60     # sliding-window half-width (px)
    MINPIX           = 50     # min pixels to recenter a window
    POLY_MARGIN_BASE = 60     # normal search band around polynomial (px)
    POLY_MARGIN_CURV = 120    # wider band when curvature is high
    MIN_PIX_OK       = 200    # min pixels to declare a line "found"
    EMA_ALPHA        = 0.50   # polynomial EMA weight
    STALE_FIT_FRAMES = 5

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None   # EMA-smoothed left polynomial
        self.sr         = None   # EMA-smoothed right polynomial
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def update(self, warped_binary):
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy)
            mode_label  = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit  = fl
            self.sl        = self._ema(self.sl, fl)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            self.sr         = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        # Sanity: if both found, check lane width is plausible
        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                # Drop the less-confident line
                if self.left_conf < self.right_conf:
                    self.left_fit  = None
                    self.sl        = None
                    self.left_stale = self.STALE_FIT_FRAMES  # force discard next frame
                    has_l          = False
                else:
                    self.right_fit  = None
                    self.sr         = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r           = False

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0,
                     nav_state="NORMAL"):
        """
        Compute the BEV x-pixel the car should steer toward.

        RIGHT-LANE CONVENTION:
          left_fit  = white centre divider
          right_fit = white outer edge
        """
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit):
            return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None:
                return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None:
                return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr is not None:
                return ev(sr) - hw + extra_offset_px, "JCT_EDGE"
            if sl is not None:
                return ev(sl) + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        # NORMAL right-lane driving
        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "DUAL"

        # Ghost-line extrapolation
        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "GHOST_L"
        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + DUAL_OFFSET_PX, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None:
            return 0.0
        a, b = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def reset(self):
        self.mode      = "SEARCH"
        self.left_fit  = None
        self.right_fit = None
        self.sl        = None
        self.sr        = None

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------
    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)

        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN

        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float),
                                   np.ones(20) / 20, mode='same')
            p1 = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1-40):min(self.w, p1+40)] = 0
            p2 = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh

            xl0 = max(0, lx - self.SW_MARGIN)
            xl1 = min(self.w, lx + self.SW_MARGIN)
            xr0 = max(0, rx - self.SW_MARGIN)
            xr1 = min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]

            li.append(gl)
            ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(li)
        ri = np.concatenate(ri)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)

        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 80 < w < 560

    def _ema(self, prev, new):
        if prev is None:
            return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


# ===========================================================================
# JUNCTION DETECTOR
# ===========================================================================
class JunctionDetector:

    ENTRY_FRAMES       = 5
    EXIT_FRAMES        = 8
    CROSS_ENERGY_RATIO = 1.4
    WIDTH_RATIO_HIGH   = 1.6
    MIN_BOT_ENERGY     = 500

    def __init__(self):
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0

    def update(self, warped_binary, left_conf, right_conf,
               left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape

        both_lost = (left_conf < 200) and (right_conf < 200)
        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        hist_bot = float(np.sum(warped_binary[h // 2:, :]))

        cross_energy = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_energy = (hist_top / hist_bot) > self.CROSS_ENERGY_RATIO

        wide_lane = False
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit,  h - 50)
            rx = np.polyval(right_fit, h - 50)
            if (rx - lx) > lane_width_px * self.WIDTH_RATIO_HIGH:
                wide_lane = True

        evidence = both_lost or cross_energy or wide_lane

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state         = "JUNCTION"
                self.exit_count    = 0
                self.frames_in_jct = 0
                log.info("Junction ENTERED")

        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15:
                self.state       = "NORMAL"
                self.entry_count = 0
                log.info("Junction EXITED")

        return self.state


# ===========================================================================
# ROUNDABOUT NAVIGATOR
# ===========================================================================
class RoundaboutNavigator:

    ENTRY_WIDTH_RATIO  = 0.60
    EXIT_WIDTH_RATIO   = 0.82
    MIN_CIRCLE_FRAMES  = 25
    MAX_CIRCLE_FRAMES  = 120
    SPEED_SCALE        = 0.50
    LOOKAHEAD_SCALE    = 0.55

    def __init__(self):
        self.state  = "NORMAL"
        self.frames = 0

    def update(self, left_fit, right_fit, lane_width_px, img_h=480):
        y = img_h - 50

        if left_fit is not None and right_fit is not None:
            lx    = np.polyval(left_fit,  y)
            rx    = np.polyval(right_fit, y)
            ratio = (rx - lx) / max(float(lane_width_px), 1.0)

            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.state  = "ROUNDABOUT"
                    self.frames = 0
                    log.info("Roundabout ENTRY detected")

            elif self.state == "ROUNDABOUT":
                self.frames += 1
                normal_exit = (self.frames > self.MIN_CIRCLE_FRAMES
                               and ratio > self.EXIT_WIDTH_RATIO)
                timeout_exit = self.frames > self.MAX_CIRCLE_FRAMES
                if normal_exit or timeout_exit:
                    reason = "timeout" if timeout_exit else "width ratio"
                    self.state  = "NORMAL"
                    self.frames = 0
                    log.info(f"Roundabout EXIT detected ({reason})")

        elif self.state == "ROUNDABOUT":
            self.frames += 1
            if self.frames > self.MAX_CIRCLE_FRAMES:
                self.state  = "NORMAL"
                self.frames = 0
                log.info("Roundabout EXIT (timeout, one line lost)")

        return self.state


# ===========================================================================
# DIVIDER GUARD  (hard safety layer — runs every frame)
# ===========================================================================
class DividerGuard:

    DIVIDER_SAFE_PX = 55    # min gap: car centre → centre divider
    EDGE_SAFE_PX    = 75    # INCREASED: Keep further away from right outer edge
    GAIN            = 0.09  # proportional correction gain
    MAX_CORR        = 8.0   # max correction per trigger (degrees)
    DEADBAND_PX     = 5

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction  = 0.0
        speed_scale = 1.0
        triggered   = False

        # --- Centre divider (left_fit) ---
        div_corr = 0.0
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x          # positive = car is right of divider
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)   # steer right (+)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        # --- Outer edge (right_fit) ---
        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x        # positive = car is left of edge
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)  # steer left (-)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ===========================================================================
# MAIN PILOT
# ===========================================================================
class BFMC_Pilot:

    STEER_EMA_SLOW = 0.25
    STEER_EMA_FAST = 0.50
    GUARD_EMA      = 0.55
    MAX_STEER      = 30.0
    MAX_STEER_RATE = 5.0

    HIGH_CURV_THRESH = 0.003
    MED_CURV_THRESH  = 0.0015
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode

        # Serial
        self.handler   = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()

        # Camera
        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(
                    main={"size": (640, 480), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.warning(f"Camera init failed: {e} — using blank frames")

        # Perspective transform
        self.M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Sub-systems
        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.rbt     = RoundaboutNavigator()
        self.jct     = JunctionDetector()
        self.guard   = DividerGuard()

        # State
        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0 
        self.last_target   = 320.0 + RIGHT_LANE_OFFSET_PX
        self.lost_frames   = 0

        # FPS tracking
        self._fps_t = time.time()
        self._fps   = 0.0

        # UI window
        cv2.namedWindow("BFMC_v2")
        cv2.createTrackbar("Look Ahead",    "BFMC_v2", 150, 300, lambda x: None)
        cv2.createTrackbar("Lane Width PX", "BFMC_v2", 280, 400, lambda x: None)
        cv2.createTrackbar("Fine Offset",   "BFMC_v2",  50, 100, lambda x: None)
        cv2.createTrackbar("Base Speed",    "BFMC_v2",  50, 150, lambda x: None)

    # ------------------------------------------------------------------
    # IMAGE PROCESSING
    # ------------------------------------------------------------------
    def _get_bev(self, frame):
        warped_colour = cv2.warpPerspective(frame, self.M, (640, 480))
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        L   = self.clahe.apply(hls[:, :, 1])

        binary = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ------------------------------------------------------------------
    # STEERING
    # ------------------------------------------------------------------
    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / LANE_WIDTH_M
        dx    = target_x - 320.0
        dy    = max(float(look_ahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    # ------------------------------------------------------------------
    # VISUALISATION
    # ------------------------------------------------------------------
    def _draw_poly(self, img, fit, colour):
        if fit is None:
            return
        ploty = np.linspace(0, 479, 240).astype(np.float32)
        xs    = np.polyval(fit, ploty).astype(np.float32)
        pts = np.stack([xs, ploty], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(img, [pts], isClosed=False, color=colour, thickness=3)

    def _update_fps(self):
        now = time.time()
        dt  = now - self._fps_t
        self._fps_t = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

    # ------------------------------------------------------------------
    # CALIBRATION PHASE
    # ------------------------------------------------------------------
    def calibrate(self):
        print("\n=== CALIBRATION WARM-UP ===")
        print("1. Place the car in the lane.")
        print("2. Let the camera auto-exposure adjust for a few seconds.")
        print("3. Ensure the lane lines are highlighted in Yellow and Cyan.")
        print("4. Auto-starting in 5 seconds (or press 'C' to start now)...\n")

        start_t = time.time()
        while True:
            if self.cam_ok:
                frame = self.picam2.capture_array()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "SIM MODE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            warped = self._get_bev(frame)
            
            # Update tracker to build up EMA memory of polynomials before moving
            sl, sr, dbg, _ = self.tracker.update(warped)
            
            elapsed_time = time.time() - start_t
            remaining_time = max(0, 5.0 - elapsed_time)

            cv2.putText(dbg, "--- CALIBRATION WARM-UP ---", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(dbg, f"Auto-starting in {remaining_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(dbg, "Press 'C' to Start now", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            self._draw_poly(dbg, sl, (255, 220, 0))    # divider
            self._draw_poly(dbg, sr, (0,   200, 255))  # edge

            cv2.imshow("BFMC_v2", dbg)

            key = cv2.waitKey(30) & 0xFF
            
            # Auto-proceed after 5 seconds OR if 'C' is manually pressed
            if key == ord('c') or remaining_time == 0:
                print("\n--> Calibration complete. Proceeding to autonomous mode...")
                break
            elif key == ord('q'):
                self.stop()
                sys.exit(0)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        print("BFMC Pilot v2: STARTING — RIGHT LANE DRIVING MODE")
        if self.sim_mode:
            print("  [SIMULATION MODE — no camera, no serial]")
        if not self.connected and not self.sim_mode:
            log.warning("Serial connection is FALSE - Check wiring to STM32!")

        # 1. Run calibration first!
        self.calibrate()

        # 2. Proceed to main loop
        try:
            while True:
                t_frame_start = time.time()

                # --- Read trackbars ---
                look_ahead    = cv2.getTrackbarPos("Look Ahead",    "BFMC_v2")
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_v2")
                fine_offset   = cv2.getTrackbarPos("Fine Offset",   "BFMC_v2")
                base_speed    = cv2.getTrackbarPos("Base Speed",    "BFMC_v2")

                fine_px      = (fine_offset - 50) * 2
                total_offset = RIGHT_LANE_OFFSET_PX + fine_px

                # --- Capture frame ---
                if self.cam_ok:
                    frame = self.picam2.capture_array()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                warped = self._get_bev(frame)

                # --- Lane tracking ---
                sl, sr, dbg, detect_mode = self.tracker.update(warped)

                # --- State machines ---
                jct_state = self.jct.update(
                    warped,
                    self.tracker.left_conf,
                    self.tracker.right_conf,
                    self.tracker.left_fit,
                    self.tracker.right_fit,
                    lane_width_px)

                rbt_state = self.rbt.update(
                    self.tracker.left_fit,
                    self.tracker.right_fit,
                    lane_width_px)

                nav_state = rbt_state if rbt_state == "ROUNDABOUT" else jct_state

                # --- Effective look-ahead ---
                curvature_pre = self.tracker.get_curvature(self.tracker.h // 2)
                if nav_state == "ROUNDABOUT":
                    eff_la = int(look_ahead * self.rbt.LOOKAHEAD_SCALE)
                elif nav_state == "JUNCTION":
                    eff_la = int(look_ahead * 0.75)
                elif curvature_pre > self.HIGH_CURV_THRESH:
                    eff_la = int(look_ahead * 0.60)
                elif curvature_pre > self.MED_CURV_THRESH:
                    eff_la = int(look_ahead * 0.80)
                else:
                    eff_la = look_ahead

                eff_la = max(60, eff_la)
                y_eval = max(0, 480 - eff_la)

                # --- Target x ---
                target_x, anchor = self.tracker.get_target_x(
                    y_eval, lane_width_px, total_offset, nav_state)

                lost = target_x is None
                if lost:
                    self.lost_frames += 1
                    target_x = self.last_target
                else:
                    self.lost_frames  = 0
                    self.last_target  = target_x

                # --- Pure pursuit ---
                raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)

                steer_delta_abs = abs(raw_steer - self.smooth_steer)
                alpha = (self.STEER_EMA_FAST if steer_delta_abs > 8.0
                         else self.STEER_EMA_SLOW)
                self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
                steer_angle = self.smooth_steer

                rate_delta = steer_angle - self.prev_steer
                rate_delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, rate_delta))
                steer_angle    = self.prev_steer + rate_delta
                self.prev_steer = steer_angle

                # --- Divider + edge guard ---
                raw_steer_guarded, guard_spd, guard_on = self.guard.apply(
                    steer_angle,
                    self.tracker.left_fit,
                    self.tracker.right_fit,
                    y_eval=y_eval)

                guard_delta       = raw_steer_guarded - steer_angle
                self.smooth_guard = (self.GUARD_EMA * guard_delta
                                     + (1.0 - self.GUARD_EMA) * self.smooth_guard)
                steer_angle       = steer_angle + self.smooth_guard

                # --- Speed policy ---
                curvature = self.tracker.get_curvature(y_eval)

                if self.lost_frames > LOST_GRACE_FRAMES:
                    speed = 0.0
                    # Log to the terminal so the user knows exactly WHY it's not moving
                    if self.lost_frames % 30 == 0: 
                        log.warning("CAR STOPPED: Lane tracking completely lost.")
                elif base_speed == 0:
                    speed = 0.0
                elif nav_state == "ROUNDABOUT":
                    speed = base_speed * self.rbt.SPEED_SCALE
                elif nav_state == "JUNCTION":
                    speed = base_speed * 0.55
                elif curvature > self.HIGH_CURV_THRESH:
                    speed = base_speed * self.HIGH_CURV_SCALE
                elif curvature > self.MED_CURV_THRESH:
                    speed = base_speed * self.MED_CURV_SCALE
                elif abs(steer_angle) > 18:
                    speed = base_speed * 0.60
                elif abs(steer_angle) > 10:
                    speed = base_speed * 0.80
                else:
                    speed = float(base_speed)

                if 0 < self.lost_frames <= LOST_GRACE_FRAMES:
                    speed *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)

                if guard_on:
                    speed *= guard_spd

                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

                # --- Actuate ---
                if self.connected:
                    try:
                        # Safely cast to float for the serial handler
                        self.handler.set_speed(float(speed))
                        self.handler.set_steering(float(steer_angle))
                    except Exception as e:
                        log.error(f"Serial Handler Error: {e}")

                # --- Visualisation ---
                self._draw_poly(dbg, sl, (255, 220, 0))    # divider  = yellow
                self._draw_poly(dbg, sr, (0,   200, 255))  # edge     = cyan

                cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                cv2.line(dbg, (int(target_x), y_eval), (320, 470), (0, 255, 0), 2)
                cv2.line(dbg, (320, 450), (320, 480), (0, 0, 255), 3)

                ref_x = 320 + int(total_offset)
                for y_tick in range(0, 480, 20):
                    cv2.line(dbg, (ref_x, y_tick), (ref_x, y_tick + 10),
                             (100, 100, 100), 1)

                if guard_on:
                    cv2.putText(dbg, "! GUARD !", (230, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if lost:
                    grace_label = (f"LOST ({self.lost_frames}/{LOST_GRACE_FRAMES})"
                                   if self.lost_frames <= LOST_GRACE_FRAMES
                                   else "LOST - STOPPED")
                    cv2.putText(dbg, grace_label, (130, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                self._update_fps()
                line1 = f"{detect_mode} | {anchor} | {nav_state} | {self._fps:.0f}fps"
                line2 = (f"Steer:{steer_angle:.1f}  Speed:{speed:.0f}"
                         f"  Curv:{curvature:.4f}  Off:{int(total_offset)}")

                cv2.putText(dbg, line1, (10,  26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(dbg, line2, (10, 462),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                cv2.imshow("BFMC_v2", dbg)

                elapsed = time.time() - t_frame_start
                wait_ms = max(1, int((FRAME_PERIOD - elapsed) * 1000))
                if cv2.waitKey(wait_ms) == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self.connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        if self.cam_ok:
            self.picam2.stop()
        cv2.destroyAllWindows()
        print("BFMC Pilot v2: STOPPED")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Hybrid Pilot v2")
    parser.add_argument("--sim", action="store_true",
                        help="Simulation mode: no camera, no serial output")
    args = parser.parse_args()

    pilot = BFMC_Pilot(sim_mode=args.sim)
    pilot.run()