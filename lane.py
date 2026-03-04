"""
lane_follower.py — Pure Lane Tracking Pilot (VROOM Method) + IMU Yaw Lock
===========================================================================
Stripped down to strictly follow the lane using the robust VROOM pipeline:
1. Inverse Perspective Mapping (IPM)
2. HLS Color Isolation + Sobel Edge Detection (Binarization)
3. Histogram + Sliding Window Tracking
4. Least-Squares Polynomial Fit & Pure Pursuit Control
5. IMU Yaw-Lock: Drives perfectly straight for 4s if the right lane is lost.
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
    sys.path.insert(0, "..")
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

# ---------------------------------------------------------------------------
# IMU - graceful fallback
# ---------------------------------------------------------------------------
try:
    from bno055_imu import BNO055_IMU
    _IMU_AVAILABLE = True
except ImportError:
    _IMU_AVAILABLE = False
    print("WARNING: bno055_imu not found - IMU disabled")
    class BNO055_IMU:
        def get_yaw(self): return 0.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# PHYSICAL CONSTANTS
# ===========================================================================
WHEELBASE_M          = 0.23    # front-to-rear axle distance (m)
LANE_WIDTH_M         = 0.35    # one-lane physical width (m)

# ===========================================================================
# CAMERA - Bird's Eye View calibration
# ===========================================================================
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# RIGHT-LANE OFFSET (Positive = shift right)
# ===========================================================================
RIGHT_LANE_OFFSET_PX = 70
DUAL_OFFSET_PX       = 0
SINGLE_DIV_OFFSET_PX = 40    # Only left divider seen -> push right
SINGLE_EDGE_OFFSET_PX = -40  # Only right edge seen -> push left

# ===========================================================================
# TIMING & GRACE PERIOD
# ===========================================================================
TARGET_FPS    = 30
FRAME_PERIOD  = 1.0 / TARGET_FPS


# ===========================================================================
# HYBRID LANE TRACKER (Pure Lane Tracking Only)
# ===========================================================================
class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60     
    MINPIX           = 50     
    POLY_MARGIN_BASE = 60     
    POLY_MARGIN_CURV = 120    
    MIN_PIX_OK       = 200    
    EMA_ALPHA        = 0.50   
    STALE_FIT_FRAMES = 5

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

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit  = None
                    self.sl        = None
                    self.left_stale = self.STALE_FIT_FRAMES  
                    has_l          = False
                else:
                    self.right_fit  = None
                    self.sr         = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r           = False

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0):
        """Pure Right-Lane convention tracking."""
        sl = self.sl
        sr = self.sr

        def ev(fit):
            return float(np.polyval(fit, y_eval))

        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX + extra_offset_px, "DUAL"

        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + SINGLE_EDGE_OFFSET_PX + extra_offset_px, "GHOST_L"

        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + SINGLE_DIV_OFFSET_PX + extra_offset_px, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN

        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

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
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            xl0 = max(0, lx - self.SW_MARGIN)
            xl1 = min(self.w, lx + self.SW_MARGIN)
            xr0 = max(0, rx - self.SW_MARGIN)
            xr1 = min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]

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
        m = self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE
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
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


# ===========================================================================
# DIVIDER GUARD (Hard safety layer)
# ===========================================================================
class DividerGuard:
    DIVIDER_SAFE_PX = 55    
    EDGE_SAFE_PX    = 50    
    GAIN            = 0.09  
    MAX_CORR        = 8.0   
    DEADBAND_PX     = 5

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction  = 0.0
        speed_scale = 1.0
        triggered   = False
        div_corr = 0.0

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x          
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)   
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x        
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)  
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ===========================================================================
# MAIN PILOT (Strictly Lane Following)
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
    DUAL_SPEED_SCALE = 1.15

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.handler   = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()

        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.warning(f"Camera init failed: {e}")

        # Instantiate IMU logic
        self.imu = BNO055_IMU()
        self.target_yaw_on_loss = 0.0
        self.is_yaw_locked      = False
        self.lane_lost_time     = 0.0

        self.M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)

        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.guard   = DividerGuard()

        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0   
        self.last_target   = 320.0 + RIGHT_LANE_OFFSET_PX

        self._fps_t = time.time()
        self._fps   = 0.0

        cv2.namedWindow("BFMC_Pure_Lane_Follower")
        cv2.createTrackbar("Look Ahead",    "BFMC_Pure_Lane_Follower", 150, 300, lambda x: None)
        cv2.createTrackbar("Lane Width PX", "BFMC_Pure_Lane_Follower", 280, 400, lambda x: None)
        cv2.createTrackbar("Fine Offset",   "BFMC_Pure_Lane_Follower",  50, 100, lambda x: None)
        cv2.createTrackbar("Base Speed",    "BFMC_Pure_Lane_Follower", 100, 200, lambda x: None)

    def _get_bev(self, frame):
        """
        VROOM-Style Binarization:
        Fuses HLS Color Masking (for bright white tape) with Sobel Edge Detection 
        (for sharp lane boundaries). Completely immune to shadows and track seams.
        """
        warped_colour = cv2.warpPerspective(frame, self.M, (640, 480), flags=cv2.INTER_LINEAR)
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        
        lower_white = np.array([0, 180, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 60], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        gray = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        edge_mask = np.zeros_like(scaled_sobel)
        edge_mask[(scaled_sobel >= 40) & (scaled_sobel <= 255)] = 255

        combined_binary = cv2.bitwise_or(white_mask, edge_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel)

        return cleaned

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

    def _draw_poly(self, img, fit, colour):
        if fit is None: return
        ploty = np.linspace(0, 479, 240).astype(np.float32)
        xs    = np.polyval(fit, ploty).astype(np.float32)
        pts = np.stack([xs, ploty], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(img, [pts], isClosed=False, color=colour, thickness=3)

    def run(self):
        print("BFMC Pure Lane Follower: STARTING")
        try:
            while True:
                t_frame_start = time.time()

                look_ahead    = cv2.getTrackbarPos("Look Ahead",    "BFMC_Pure_Lane_Follower")
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_Pure_Lane_Follower")
                fine_offset   = cv2.getTrackbarPos("Fine Offset",   "BFMC_Pure_Lane_Follower")
                base_speed    = cv2.getTrackbarPos("Base Speed",    "BFMC_Pure_Lane_Follower")

                fine_px      = (fine_offset - 50) * 2
                total_offset = RIGHT_LANE_OFFSET_PX + fine_px

                if self.cam_ok:
                    frame = self.picam2.capture_array()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                warped = self._get_bev(frame)

                # Pure Lane Tracking
                sl, sr, dbg, detect_mode = self.tracker.update(warped)

                curvature_pre = self.tracker.get_curvature(self.tracker.h // 2)
                
                # Dynamic Lookahead
                if curvature_pre > self.HIGH_CURV_THRESH:
                    eff_la = int(look_ahead * 0.60)
                elif curvature_pre > self.MED_CURV_THRESH:
                    eff_la = int(look_ahead * 0.80)
                else:
                    eff_la = look_ahead

                eff_la = max(60, eff_la)
                y_eval = max(0, 480 - eff_la)

                target_x, anchor = self.tracker.get_target_x(y_eval, lane_width_px, total_offset)
                
                # Fetch Current IMU Yaw 
                current_yaw = self.imu.get_yaw()

                # Trigger IMU Lock if we completely lose the lane OR specifically lose the right outer edge
                trigger_imu_lock = (target_x is None) or (sr is None)

                if not trigger_imu_lock:
                    self.last_target  = target_x
                    self.target_yaw_on_loss = current_yaw  # Constantly save yaw while tracking normally
                    self.is_yaw_locked = False
                    self.lane_lost_time = 0.0

                    # -----------------------------------------------
                    # Normal Vision Steering Math
                    # -----------------------------------------------
                    raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)
                else:
                    if target_x is None:
                        target_x = self.last_target
                        
                    if not self.is_yaw_locked:
                        self.is_yaw_locked = True
                        self.lane_lost_time = time.time()
                        
                    time_lost = time.time() - self.lane_lost_time
                    
                    if time_lost <= 4.0:
                        # -----------------------------------------------
                        # IMU Yaw-Lock Steering Mode (Active for 4 secs)
                        # -----------------------------------------------
                        yaw_err = (self.target_yaw_on_loss - current_yaw + math.pi) % (2 * math.pi) - math.pi
                        
                        # P-controller for steering based purely on IMU deviation. 
                        # *NOTE: If the car steers opposite to correction during a gap, change 1.5 to -1.5*
                        raw_steer = math.degrees(yaw_err) * 1.5 
                        
                        anchor = f"IMU_LOCK ({4.0 - time_lost:.1f}s)"
                    else:
                        # Timeout triggered
                        raw_steer = 0.0
                        anchor = "LOST_STOP"

                # Apply Steering EMA
                steer_delta_abs = abs(raw_steer - self.smooth_steer)
                alpha = self.STEER_EMA_FAST if steer_delta_abs > 8.0 else self.STEER_EMA_SLOW
                self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
                steer_angle = self.smooth_steer

                # Apply Rate Limiter
                rate_delta = steer_angle - self.prev_steer
                rate_delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, rate_delta))
                steer_angle    = self.prev_steer + rate_delta
                self.prev_steer = steer_angle

                # Safety Guards
                guard_left  = self.tracker.sl if self.tracker.left_stale  == 0 else None
                guard_right = self.tracker.sr if self.tracker.right_stale == 0 else None

                raw_steer_guarded, guard_spd, guard_on = self.guard.apply(steer_angle, guard_left, guard_right, y_eval=y_eval)

                # Disable guard interference while blind
                if self.is_yaw_locked:
                    self.smooth_guard = 0.0
                    guard_on = False
                else:
                    guard_delta       = raw_steer_guarded - steer_angle
                    self.smooth_guard = (self.GUARD_EMA * guard_delta + (1.0 - self.GUARD_EMA) * self.smooth_guard)
                
                steer_angle = steer_angle + self.smooth_guard

                # Speed Policy
                curvature = self.tracker.get_curvature(y_eval)

                if self.is_yaw_locked:
                    time_lost = time.time() - self.lane_lost_time
                    if time_lost > 4.0 or base_speed == 0:
                        speed = 0.0
                    else:
                        speed = base_speed * 0.75  # Coast at safe speed while driving blind
                else:
                    if curvature > self.HIGH_CURV_THRESH:
                        speed = base_speed * self.HIGH_CURV_SCALE
                    elif curvature > self.MED_CURV_THRESH:
                        speed = base_speed * self.MED_CURV_SCALE
                    elif anchor == "DUAL" and abs(steer_angle) < 10:
                        speed = base_speed * self.DUAL_SPEED_SCALE
                    elif abs(steer_angle) > 18:
                        speed = base_speed * 0.60
                    elif abs(steer_angle) > 10:
                        speed = base_speed * 0.80
                    else:
                        speed = float(base_speed)

                if guard_on:
                    speed *= guard_spd

                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

                # Actuate
                if self.connected:
                    self.handler.set_speed(speed)
                    self.handler.set_steering(steer_angle)

                # Dashboard 
                self._draw_poly(dbg, sl, (255, 220, 0))    
                self._draw_poly(dbg, sr, (0,   200, 255))  
                cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                cv2.line(dbg, (int(target_x), y_eval), (320, 470), (0, 255, 0), 2)
                cv2.line(dbg, (320, 450), (320, 480), (0, 0, 255), 3)

                ref_x = 320 + RIGHT_LANE_OFFSET_PX
                for y_tick in range(0, 480, 20):
                    cv2.line(dbg, (ref_x, y_tick), (ref_x, y_tick + 10), (100, 100, 100), 1)

                if guard_on:
                    cv2.putText(dbg, "! GUARD !", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                now = time.time()
                dt  = now - self._fps_t
                self._fps_t = now
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

                line1 = f"{detect_mode} | {anchor} | {self._fps:.0f}fps"
                line2 = f"Steer:{steer_angle:.1f}  Speed:{speed:.0f}  Curv:{curvature:.4f}  Off:{int(total_offset)}"

                cv2.putText(dbg, line1, (10,  26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(dbg, line2, (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                cv2.imshow("BFMC_Pure_Lane_Follower", dbg)

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
        print("BFMC Pure Lane Follower: STOPPED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Pure Lane Follower")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    args = parser.parse_args()

    pilot = BFMC_Pilot(sim_mode=args.sim)
    pilot.run()