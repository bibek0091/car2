"""
lane_follower.py — Standalone BFMC Lane Detection and Following
===============================================================
A consolidated script extracted directly from perception.py, 
control.py, hardware_io.py, and the main pilot loop. 

Run with:
  python lane_follower.py --sim                  (for simulation mode)
  python lane_follower.py --sim-video path.mp4   (to test on a video file)
"""

import sys
import cv2
import numpy as np
import math
import os
import time
import logging
import threading
import queue
import argparse
from dataclasses import dataclass
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lane_follower")

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_FPS   = 30
FRAME_PERIOD = 1.0 / TARGET_FPS
PWM_DEADBAND = 14.0

C_GREEN = ( 50, 220,  50)
C_AMBER = ( 50, 190, 255)
C_RED   = ( 50,  50, 230)
C_WHITE = (230, 230, 230)

# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE I/O (From hardware_io.py)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from STM32_SerialHandler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    log.warning("STM32_SerialHandler not found. Using simulation mode for STM32.")

    class STM32_SerialHandler:
        def connect(self):      return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self):   pass

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False
    log.warning("picamera2 not found. Using simulation mode for Camera.")

_CV2_AVAILABLE = True


class HardwareIO:
    def __init__(self, sim_mode=False, sim_video=None):
        _no_hw = (not _SERIAL_AVAILABLE and not _CAM_AVAILABLE)
        if _no_hw and not sim_mode:
            log.warning("No hardware drivers found — entering simulation mode.")
            sim_mode = True

        self.sim_mode  = sim_mode
        self.sim_video = sim_video
        self.camera    = None
        self.video_cap = None
        self.serial    = STM32_SerialHandler()

        self.DEADBAND_PWM  = 12.0
        self.SPEED_CALIB   = 0.00568   # m/s per PWM unit above deadband
        self.MAX_SPEED_MS  = 0.50

        self._vel_filtered    = 0.0
        self._sim_yaw         = 0.0
        self._last_sim_time   = time.time()
        self._last_cmd_speed  = 0.0
        self._last_cmd_steer  = 0.0
        self._encoder_fail_count = 0
        self._ENCODER_FAIL_LIMIT = 30

        self._frame_queue = queue.Queue(maxsize=1)
        self._running     = True

        if not self.sim_mode and _SERIAL_AVAILABLE:
            connected = self.serial.connect()
            if not connected:
                log.error("Failed to connect to STM32. Motor commands will be ignored.")

        if self.sim_video and _CV2_AVAILABLE:
            self.video_cap = cv2.VideoCapture(self.sim_video)
            log.info(f"Loaded simulation video: {self.sim_video}")
            threading.Thread(target=self._video_worker, daemon=True,
                             name="video_worker").start()
        elif not self.sim_mode and _CAM_AVAILABLE:
            try:
                self.camera = Picamera2()
                cfg = self.camera.create_video_configuration(
                    main={"size": (1280, 720), "format": "XRGB8888"},
                    controls={
                        "AwbEnable":   False,
                        "ColourGains": (3.5, 1.2),
                        "AeEnable":    True,
                        "Saturation":  1.4,
                        "Sharpness":   1.2,
                    }
                )
                self.camera.configure(cfg)
                self.camera.start()
                log.info("PiCamera2 initialized with manual ColourGains.")
                threading.Thread(target=self._camera_worker, daemon=True,
                                 name="camera_worker").start()
            except Exception as e:
                log.error(f"PiCamera2 init error: {e}")
                self.camera = None

    def _push_frame(self, frame):
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(frame)

    def _camera_worker(self):
        while self._running:
            try:
                frame = self.camera.capture_array()
                if frame is not None and _CV2_AVAILABLE:
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self._push_frame(cv2.resize(frame, (640, 480)))
            except Exception as e:
                log.warning(f"Camera worker error: {e}")
                time.sleep(0.033)

    def _video_worker(self):
        while self._running:
            if not _CV2_AVAILABLE or self.video_cap is None:
                time.sleep(0.033)
                continue
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            if ret:
                self._push_frame(cv2.resize(frame, (640, 480)))
            time.sleep(0.033)

    def read_camera(self):
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_sim_heading_deg(self):
        if self.sim_mode:
            now = time.time()
            dt_actual = max(0.001, min(now - self._last_sim_time, 0.10))
            self._last_sim_time = now
            v = max(0.0, (self._last_cmd_speed - self.DEADBAND_PWM) * self.SPEED_CALIB)
            yaw_rate = 0.0
            if v > 0.05:
                steer_rad = math.radians(max(-45.0, min(45.0, self._last_cmd_steer)))
                yaw_rate  = (v / 0.23) * math.tan(steer_rad)
            self._sim_yaw += yaw_rate * dt_actual
            return math.degrees(self._sim_yaw)
        return 0.0

    def set_steering(self, steer_angle_deg):
        self._last_cmd_steer = max(-45.0, min(45.0, steer_angle_deg))
        if self.sim_mode:
            return
        self.serial.set_steering(self._last_cmd_steer)

    def set_speed(self, speed_pwm):
        speed_pwm = max(0.0, min(100.0, speed_pwm))
        self._last_cmd_speed = speed_pwm
        if self.sim_mode:
            return
        if speed_pwm == 0.0:
            speed_mm_s = 0.0
        else:
            speed_ms   = max(0.0, (speed_pwm - self.DEADBAND_PWM) * self.SPEED_CALIB)
            raw_mm_s   = speed_ms * 1000.0
            speed_mm_s = min(500.0, raw_mm_s)
        self.serial.set_speed(speed_mm_s)

    def get_velocity_ms(self):
        if self.sim_mode:
            raw = max(0.0, (self._last_cmd_speed - self.DEADBAND_PWM) * self.SPEED_CALIB)
            self._encoder_fail_count = 0
        else:
            try:
                if hasattr(self.serial, 'get_feedback'):
                    raw_mms = self.serial.get_feedback()[0]
                else:
                    import contextlib
                    with getattr(self.serial, 'feedback_lock', contextlib.nullcontext()):
                        raw_mms = getattr(self.serial, '_feedback_speed', 0.0)
                raw = max(0.0, raw_mms / 1000.0)
                self._encoder_fail_count = 0
            except Exception as e:
                self._encoder_fail_count += 1
                if self._encoder_fail_count >= self._ENCODER_FAIL_LIMIT:
                    self._encoder_fail_count = 0
                raw = 0.0

        self._vel_filtered = 0.80 * self._vel_filtered + 0.20 * raw
        return self._vel_filtered

    def shutdown(self):
        self._running = False
        self.set_speed(0)
        time.sleep(0.1)
        self.serial.disconnect()
        if self.camera:
            self.camera.stop()
        if self.video_cap:
            self.video_cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTION (From perception.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerceptionResult:
    warped_binary:     np.ndarray
    lane_dbg:          np.ndarray
    sl:                object
    sr:                object
    target_x:          float
    lateral_error_px:  float
    anchor:            str
    confidence:        float
    lane_width_px:     float
    curvature:         float
    heading_rad:       float = 0.0
    heading_conf:      float = 0.0
    y_eval:            float = 400.0
    optical_yaw_rate:  float = 0.0
    optical_vel:       float = 0.0


class VisualOdometry:
    def __init__(self):
        self.feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0       = None
        self.old_gray = None

    def update(self, frame_bgr, dt: float):
        if dt <= 0: return 0.0, 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi  = gray[int(h * 0.6):, :]

        if self.p0 is None or len(self.p0) < 10:
            p0_roi = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            if p0_roi is not None:
                p0_roi[:, 0, 1] += int(h * 0.6)
                self.p0       = p0_roi
                self.old_gray = gray.copy()
            return 0.0, 0.0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)

        if p1 is None or st is None:
            self.p0 = None
            return 0.0, 0.0

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        yaw_rate = vel = 0.0
        if len(good_new) > 3:
            dx = good_new[:, 0] - good_old[:, 0]
            dy = good_new[:, 1] - good_old[:, 1]
            yaw_rate = float(-np.median(dx) * 0.015 / dt)
            vel      = float( np.median(dy) * 0.008 / dt)

        self.old_gray = gray.copy()
        self.p0       = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
        return yaw_rate, vel


class DeadReckoningNavigator:
    def __init__(self):
        self.last_valid_target    = 320.0
        self.last_valid_curvature = 0.0
        self._lost_time_s         = 0.0

    def reset_lost_timer(self):
        self._lost_time_s = 0.0

    def accumulate(self, dt: float):
        self._lost_time_s += dt

    def predict_target(self, last_speed, last_steering):
        t = max(0.0, self._lost_time_s)
        lateral_drift   = last_steering * 2.0 * t
        predicted_target = self.last_valid_target + lateral_drift
        if abs(self.last_valid_curvature) > 0.001:
            predicted_target += self.last_valid_curvature * 5000 * t
        predicted_target = float(np.clip(predicted_target, 150, 490))
        confidence       = max(0.0, 1.0 - t / 2.0)
        return predicted_target, confidence


class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.55
    EMA_ALPHA_TURN   = 0.75
    STALE_FIT_FRAMES = 12

    WIDE_ROAD_PX             = 420
    SINGLE_LANE_PX           = 200
    RIGHT_LANE_BIAS_PX       = 0
    DIVIDER_FOLLOW_OFFSET_PX = 90

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None
        self.sr         = None
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0
        self.dead_reckoner = DeadReckoningNavigator()
        self.estimated_lane_width = 280.0

    def update(self, warped_binary, map_hint: str = "STRAIGHT"):
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv, map_hint=map_hint)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy, map_hint=map_hint)
            mode_label  = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit  = fl
            curv_now = self.get_curvature(self.h // 2)
            alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
            self.sl        = self._ema(self.sl, fl, alpha)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit, self.sl = None, None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            curv_now = self.get_curvature(self.h // 2)
            alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
            self.sr         = self._ema(self.sr, fr, alpha)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit, self.sr = None, None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit, self.sl, self.left_stale, has_l = None, None, self.STALE_FIT_FRAMES, False
                else:
                    self.right_fit, self.sr, self.right_stale, has_r = None, None, self.STALE_FIT_FRAMES, False
            else:
                y_positions = [100, 200, 300, 400]
                widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y) for y in y_positions]
                weighted_avg_width = np.average(widths, weights=[4, 3, 2, 1])
                self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0,
                     nav_state="NORMAL", frames_lost=0, last_speed=0.0, last_steering=0.0):
        sl, sr = self.sl, self.sr
        hw = lane_width_px / 2.0

        def ev(fit): return float(np.polyval(fit, y_eval))

        has_right = (sr is not None)
        has_left  = (sl is not None)

        if not has_right and not has_left:
            predicted_x, conf = self.dead_reckoner.predict_target(last_speed, last_steering)
            return predicted_x + extra_offset_px, f"DEAD_RECKONING_{conf:.2f}"

        if has_right:
            if has_left:
                base_x = (ev(sl) + ev(sr)) / 2.0 + self.RIGHT_LANE_BIAS_PX
                anchor = "RL_DUAL"
            else:
                base_x = ev(sr) - hw + self.RIGHT_LANE_BIAS_PX
                anchor = "RL_FROM_EDGE"
        else:
            base_x = ev(sl) + self.DIVIDER_FOLLOW_OFFSET_PX
            anchor = "DIVIDER_FOLLOW"

        self.dead_reckoner.last_valid_target    = base_x
        self.dead_reckoner.last_valid_curvature = self.get_curvature(y_eval)
        self.dead_reckoner.reset_lost_timer()
        return base_x + extra_offset_px, anchor

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy, map_hint: str = "STRAIGHT"):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid, margin = int(self.w * 0.40), self.SW_MARGIN

        shift = 0
        if map_hint == "LEFT":  shift = -80
        elif map_hint == "RIGHT": shift = 80

        l_lo =  max(margin, margin + shift)
        l_hi =  max(l_lo + 1, mid - margin + shift)
        r_lo =  max(margin, mid + margin + shift)
        r_hi =  min(self.w - margin, self.w - margin)

        lb = int(np.argmax(hist[l_lo:l_hi])) + l_lo if l_hi > l_lo else margin
        rb = int(np.argmax(hist[r_lo:r_hi])) + r_lo if r_hi > r_lo else mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode='same')
            p1 = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1-40):min(self.w, p1+40)] = 0
            p2 = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo, y_hi = self.h - (win + 1) * wh, self.h - win * wh
            xl0, xl1 = max(0, lx - self.SW_MARGIN), min(self.w, lx + self.SW_MARGIN)
            xr0, xr1 = max(0, rx - self.SW_MARGIN), min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]
            li.append(gl); ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li, ri = np.concatenate(li), np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0, map_hint: str = "STRAIGHT"):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)

        def band(fit): return ((nzx > np.polyval(fit, nzy) - m) & (nzx < np.polyval(fit, nzy) + m)).nonzero()[0]
        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy, map_hint=map_hint)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 180 < w < 420

    def _ema(self, prev, new, alpha=None):
        if alpha is None: alpha = self.EMA_ALPHA
        if prev is None: return new.copy()
        return alpha * new + (1.0 - alpha) * prev


class VisionPipeline:
    def __init__(self):
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.vo = VisualOdometry()
        self.lost_frames = 0
        self.last_target_x = 320.0
        self._heading_ema = 0.0

    def process(self, raw_frame, dt: float = 0.033, extra_offset_px=0.0,
                nav_state="NORMAL", velocity_ms=0.0, last_steering=0.0,
                upcoming_curve: str = "STRAIGHT", pitch_rad: float = 0.0) -> PerceptionResult:
        if raw_frame.shape[:2] != (480, 640):
            process_frame = cv2.resize(raw_frame, (640, 480))
        else:
            process_frame = raw_frame

        opt_yaw_rate, opt_vel = self.vo.update(process_frame, dt)

        if abs(pitch_rad) > 0.001:
            shift_px  = int(pitch_rad * 400)
            dyn_src   = self.SRC_PTS.copy()
            dyn_src[0][1] += shift_px
            dyn_src[1][1] += shift_px
            M_use = cv2.getPerspectiveTransform(dyn_src, self.DST_PTS)
        else:
            M_use = self.M_forward

        warped_colour = cv2.warpPerspective(process_frame, M_use, (640, 480))
        lab = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2LAB)
        L = self.clahe.apply(lab[:, :, 0])
        
        mean_l = np.mean(L)
        if mean_l < 100:
            L = cv2.convertScaleAbs(L, alpha=1.0 + (100 - mean_l)/200, beta=int((100 - mean_l)*0.6))
        elif mean_l > 180:
            L = cv2.convertScaleAbs(L, alpha=1.0 - (mean_l - 180)/350, beta=int(-(mean_l - 180)*0.4))

        binary = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        map_hint = upcoming_curve if upcoming_curve in ("LEFT", "RIGHT") else "STRAIGHT"
        sl, sr, line_dbg, mode_label = self.tracker.update(warped_binary, map_hint=map_hint)
        
        y_eval = 400.0
        lw = self.tracker.estimated_lane_width
        
        target_x, anchor = self.tracker.get_target_x(
            y_eval, lw, extra_offset_px, nav_state, self.lost_frames, velocity_ms, last_steering
        )
        if not hasattr(self, "_target_ema"):
            self._target_ema = target_x
        self._target_ema = 0.8 * self._target_ema + 0.2 * target_x
        target_x = self._target_ema
        
        if target_x is None:
            self.lost_frames += 1
            self.tracker.dead_reckoner.accumulate(dt)
            target_x = self.last_target_x
        else:
            self.lost_frames = 0
            self.last_target_x = target_x

        curv = self.tracker.get_curvature(y_eval)
        conf = 1.0 if (sl is not None and sr is not None) else 0.5 if (sl is not None or sr is not None) else 0.0
        
        heading_rad = 0.0
        def _lane_heading(fit, y):
            return math.atan2(np.polyval(fit, y - 50) - np.polyval(fit, y), 50)
        
        if sl is not None and sr is not None:
            heading_rad = (_lane_heading(sl, y_eval) + _lane_heading(sr, y_eval)) / 2.0
        elif sl is not None:
            heading_rad = _lane_heading(sl, y_eval)
        elif sr is not None:
            heading_rad = _lane_heading(sr, y_eval)
            
        self._heading_ema = 0.7 * self._heading_ema + 0.3 * heading_rad
        heading_rad = self._heading_ema

        return PerceptionResult(
            warped_binary=warped_binary, lane_dbg=line_dbg, sl=sl, sr=sr,
            target_x=target_x, lateral_error_px=target_x - 320.0, anchor=anchor,
            confidence=conf, lane_width_px=lw, curvature=curv,
            heading_rad=heading_rad, heading_conf=conf, y_eval=y_eval,
            optical_yaw_rate=opt_yaw_rate, optical_vel=opt_vel,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL (From control.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm:       float
    target_x:        float
    anchor:          str
    lookahead_px:    float
    steer_ff_deg:    float = 0.0
    steer_react_deg: float = 0.0


class StanleyController:
    def __init__(self, k: float = 1.2, ks: float = 0.2, wheelbase_m: float = 0.23):
        self.k  = k
        self.ks = ks
        self.L  = wheelbase_m

    def compute(self, target_x_px: float, heading_rad: float, velocity_ms: float, 
                lane_width_px: float, map_curvature: float = 0.0):
        ppm  = max(lane_width_px, 50) / 0.35
        ce_m = (320.0 - target_x_px) / ppm

        k_eff = self.k * min(1.0, velocity_ms / 0.25)
        reactive_rad = heading_rad + math.atan2(k_eff * ce_m, velocity_ms + self.ks)
        feed_forward_rad = math.atan(self.L * map_curvature)

        total_deg    = math.degrees(reactive_rad + feed_forward_rad)
        reactive_deg = math.degrees(reactive_rad)
        ff_deg       = math.degrees(feed_forward_rad)
        return total_deg, reactive_deg, ff_deg


class DividerGuard:
    DIVIDER_SAFE_PX = 130
    EDGE_SAFE_PX    = 100
    GAIN            = 0.35
    MAX_CORR        = 25.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered = 0.0, 1.0, False
        div_corr = edge_corr = 0.0

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))
                triggered   = True

        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


class Controller:
    MAX_STEER      = 45.0
    MAX_STEER_RATE = 20.0
    BRAKING_DISTANCE_M = 1.8
    MIN_CURVE_SPEED_F  = 0.45

    def __init__(self):
        self.prev_steer = 0.0
        self.guard      = DividerGuard()
        self.stanley    = StanleyController(k=1.2, ks=0.2, wheelbase_m=0.23)

    def compute(self, perc_res, nav_state: str = "NORMAL", velocity_ms: float = 0.0,
                dt: float = 0.033, base_speed: float = 50.0, traffic_mult: float = 1.0,
                map_curvature: float = 0.0, upcoming_curve: str = "STRAIGHT",
                curve_dist_m: float = 99.0) -> ControlOutput:

        raw_steer, react_steer_deg, ff_steer_deg = self.stanley.compute(
            perc_res.target_x, perc_res.heading_rad, velocity_ms, perc_res.lane_width_px, map_curvature)

        rate_delta  = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, raw_steer - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta

        alpha = 0.7
        steer_angle = alpha * self.prev_steer + (1 - alpha) * steer_angle
        self.prev_steer = steer_angle

        steer_guarded, guard_spd_mult, _ = self.guard.apply(
            steer_angle, perc_res.sl, perc_res.sr, y_eval=perc_res.y_eval)
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_guarded))

        speed           = float(base_speed)
        min_curve_speed = base_speed * self.MIN_CURVE_SPEED_F

        if upcoming_curve != "STRAIGHT" and curve_dist_m < self.BRAKING_DISTANCE_M:
            decel_factor = max(0.0, curve_dist_m / self.BRAKING_DISTANCE_M)
            braked_speed = min_curve_speed + (base_speed - min_curve_speed) * decel_factor
            speed = min(speed, braked_speed)
        elif abs(steer_angle) < 5:
            speed = min(speed * 1.15, base_speed * 1.20)

        if "DEAD_RECKONING" in perc_res.anchor:
            try: dr_conf = float(perc_res.anchor.split("_")[2])
            except: dr_conf = 0.5
            speed *= (0.4 + 0.4 * dr_conf)

        if perc_res.anchor == "DIVIDER_FOLLOW":
            speed *= 0.75

        final_speed = speed * traffic_mult * guard_spd_mult

        MINIMUM_DRIVE_PWM = 18.0
        if final_speed > 0:
            final_speed = max(final_speed, MINIMUM_DRIVE_PWM)

        return ControlOutput(
            steer_angle_deg = steer_angle,
            speed_pwm       = final_speed,
            target_x        = perc_res.target_x,
            anchor          = perc_res.anchor,
            lookahead_px    = 0.0,
            steer_ff_deg    = ff_steer_deg,
            steer_react_deg = react_steer_deg,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & ORCHESTRATION (From main.py)
# ══════════════════════════════════════════════════════════════════════════════

def _lbl(img, txt, x, y, scale=0.38, color=C_WHITE, t=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)

def _annotate_bev(perc, ctrl):
    dbg = perc.lane_dbg.copy() if perc.lane_dbg is not None else np.zeros((480,640,3),np.uint8)

    def draw_poly(fit, color):
        if fit is None: return
        ys  = np.linspace(40,479,240).astype(np.float32)
        xs  = np.clip(np.polyval(fit,ys),0,639).astype(np.float32)
        pts = np.stack([xs,ys],axis=1).reshape(-1,1,2).astype(np.int32)
        cv2.polylines(dbg,[pts],False,color,3,cv2.LINE_AA)

    draw_poly(perc.sl,(255,80,80))
    draw_poly(perc.sr,(80,80,255))

    if perc.sl is not None and perc.sr is not None:
        lx = int(np.clip(np.polyval(perc.sl,400),0,639))
        rx = int(np.clip(np.polyval(perc.sr,400),0,639))
        cv2.line(dbg,(lx,400),(rx,400),(70,170,70),1,cv2.LINE_AA)
        _lbl(dbg,f"w={perc.lane_width_px:.0f}px",(lx+rx)//2-20,396,scale=0.34,color=(70,170,70))

    yrow = int(perc.y_eval)
    yc   = C_GREEN if "DUAL" in ctrl.anchor else (C_AMBER if "DEAD" not in ctrl.anchor else C_RED)
    for xi in range(0,640,18): cv2.line(dbg,(xi,yrow),(xi+9,yrow),yc,1,cv2.LINE_AA)

    tx = max(4, min(636, int(ctrl.target_x)))
    for yi in range(360,440,12): cv2.line(dbg,(tx,yi),(tx,yi+6),(0,255,255),2,cv2.LINE_AA)
    cv2.line(dbg,(tx-12,yrow),(tx+12,yrow),(0,255,255),2,cv2.LINE_AA)

    curv = perc.curvature
    if curv > 1e-5:
        R = min(int(1.0/curv),1400)
        if R < 700:
            sign = 1 if (perc.sl is not None and perc.sl[0]>0) else -1
            cv2.ellipse(dbg,(tx+sign*R,400),(R,R),0,84,96,(190,70,170),2,cv2.LINE_AA)

    _lbl(dbg,ctrl.anchor,10,25,scale=0.50,color=C_WHITE)
    _lbl(dbg,f"steer={ctrl.steer_angle_deg:+.1f}",10,50,scale=0.44,color=(70,225,70))
    _lbl(dbg,f"conf={perc.confidence:.2f}  curv={perc.curvature:.5f}",10,72,scale=0.37,color=(150,150,150))
    return dbg


class StandalonePilot:
    """A stripped-down autonomous orchestrator that purely tracks and follows the lane."""
    def __init__(self, sim_mode=False, base_speed=50.0, sim_video=None):
        self.sim_mode = sim_mode
        self.base_speed = base_speed
        self.hw = HardwareIO(sim_mode=sim_mode, sim_video=sim_video)
        self.vision = VisionPipeline()
        self.controller = Controller()
        self.running = False
        self._last_ctrl = None

    def run(self):
        self.running = True
        log.info("Lane Follower loop started")
        startup_time = time.time()
        t_prev = time.time()
        _ll = 0; _LLC = 15; _LLS = 90
        
        try:
            while self.running:
                ts = time.time()
                dt = max(ts - t_prev, 0.001)
                t_prev = ts
                elapsed_run = ts - startup_time

                raw_frame = self.hw.read_camera()
                if raw_frame is None or raw_frame.size == 0:
                    raw_frame = np.zeros((480, 640, 3), np.uint8)
                velocity_ms = self.hw.get_velocity_ms()

                perc = self.vision.process(
                    raw_frame,
                    dt=dt,
                    extra_offset_px=0.0,
                    nav_state="NORMAL",
                    velocity_ms=velocity_ms,
                    last_steering=getattr(self._last_ctrl, 'steer_angle_deg', 0.0),
                    upcoming_curve="STRAIGHT"
                )

                ctrl = self.controller.compute(
                    perc_res=perc, nav_state="NORMAL",
                    base_speed=float(self.base_speed),
                    traffic_mult=1.0,
                    velocity_ms=velocity_ms, dt=dt,
                    map_curvature=0.0,
                    upcoming_curve="STRAIGHT",
                    curve_dist_m=99.0
                )

                # Startup calibration override
                if elapsed_run < 3.0:
                    ctrl.speed_pwm = 0.0
                    ctrl.steer_angle_deg = 0.0
                    cv2.putText(perc.lane_dbg, f"CAM CALIB: {3.0 - elapsed_run:.1f}s",
                                (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                elif elapsed_run < 6.0:
                    ctrl.speed_pwm = min(ctrl.speed_pwm, 15.0)
                    cv2.putText(perc.lane_dbg, f"LANE CALIB: {6.0 - elapsed_run:.1f}s",
                                (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

                self._last_ctrl = ctrl

                _ll = _ll + 1 if (perc.sl is None and perc.sr is None) else 0
                if _ll >= _LLS:
                    self.hw.set_speed(0)
                    self.hw.set_steering(0)
                else:
                    speed = ctrl.speed_pwm
                    if _ll >= _LLC: speed = min(speed, 20.0)
                    if 0.0 < speed < self.hw.DEADBAND_PWM: speed = self.hw.DEADBAND_PWM
                    self.hw.set_speed(speed)
                    self.hw.set_steering(ctrl.steer_angle_deg)

                # BEV Visualization Window
                dbg = _annotate_bev(perc, ctrl)
                cv2.imshow("Standalone BEV Lane Tracker", dbg)
                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    self.running = False

                elapsed = time.time() - ts
                time.sleep(max(0.001, FRAME_PERIOD - elapsed))

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        except Exception as e:
            log.error(f"FATAL Pilot crash: {e}", exc_info=True)
        finally:
            log.info("Pilot loop exited")
            self.hw.set_speed(0)
            self.hw.set_steering(0)
            self.hw.shutdown()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Standalone Lane Detection and Following")
    ap.add_argument("--sim",       action="store_true", help="Run in simulation mode")
    ap.add_argument("--speed",     type=float, default=50, help="Base speed PWM")
    ap.add_argument("--sim-video", type=str,   default=None, help="Path to video for testing")
    args = ap.parse_args()

    pilot = StandalonePilot(sim_mode=args.sim, base_speed=args.speed, sim_video=args.sim_video)
    pilot.run()