"""
perception.py — Simplified BEV Lane Tracker (White Road / Black Lines)
======================================================================
REMOVED: 
- Dead reckoning, Kalman filters, Visual Odometry.
- Aggressive CLAHE lighting adjustments that crash on white backgrounds.

ADDED:
- Adaptive thresholding specifically formulated for DARK lines on BRIGHT surfaces.
- Robust, lightweight sliding window tracker with simple EMA smoothing.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass

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
    optical_yaw_rate:  float = 0.0  # Kept as dummy variable so main.py doesn't crash
    optical_vel:       float = 0.0  # Kept as dummy variable so main.py doesn't crash


class SimpleLaneTracker:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.n_windows = 9
        self.margin = 80
        self.minpix = 40
        
        self.sl = None
        self.sr = None
        
    def update(self, binary):
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        nz = binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])
        
        li = np.array([], dtype=int)
        ri = np.array([], dtype=int)
        
        # 1. Poly Search (Fast tracking if we have stable previous lines)
        if self.sl is not None or self.sr is not None:
            if self.sl is not None:
                li = ((nzx > np.polyval(self.sl, nzy) - self.margin) & 
                      (nzx < np.polyval(self.sl, nzy) + self.margin)).nonzero()[0]
            if self.sr is not None:
                ri = ((nzx > np.polyval(self.sr, nzy) - self.margin) & 
                      (nzx < np.polyval(self.sr, nzy) + self.margin)).nonzero()[0]
            
            # Fallback to full sliding window if poly search lost the lines
            if len(li) < 200 and len(ri) < 200:
                li, ri = self._sliding_window(binary, nzx, nzy, dbg)
        else:
            # 2. Sliding Window Search (from scratch)
            li, ri = self._sliding_window(binary, nzx, nzy, dbg)
            
        # 3. Fit polynomials and apply EMA smoothing
        if len(li) > 200:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.sl = self._ema(self.sl, fl)
            dbg[nzy[li], nzx[li]] = [255, 80, 80]  # Blue-ish left dots
        else:
            self.sl = None
            
        if len(ri) > 200:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.sr = self._ema(self.sr, fr)
            dbg[nzy[ri], nzx[ri]] = [80, 80, 255]  # Red-ish right dots
        else:
            self.sr = None
            
        return self.sl, self.sr, dbg
        
    def _ema(self, old, new, alpha=0.7):
        """Exponential Moving Average to smooth steering targets."""
        if old is None: return new
        return alpha * new + (1.0 - alpha) * old

    def _sliding_window(self, binary, nzx, nzy, dbg):
        hist = np.sum(binary[self.h//2:, :], axis=0)
        mid = self.w // 2
        
        # Find strongest histogram peaks in left and right halves
        l_base = np.argmax(hist[:mid])
        r_base = np.argmax(hist[mid:]) + mid
        
        wh = self.h // self.n_windows
        lx, rx = l_base, r_base
        li, ri = [], []
        
        for win in range(self.n_windows):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            
            xl0, xl1 = max(0, lx - self.margin), min(self.w, lx + self.margin)
            xr0, xr1 = max(0, rx - self.margin), min(self.w, rx + self.margin)
            
            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)
            
            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]
            
            li.append(gl)
            ri.append(gr)
            
            if len(gl) > self.minpix: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.minpix: rx = int(np.mean(nzx[gr]))
            
        return np.concatenate(li), np.concatenate(ri)


class VisionPipeline:
    def __init__(self):
        # Existing BEV mapping matrix
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.tracker = SimpleLaneTracker()
        
        # Default tracking targets (keeps car hugging right-lane)
        self.RIGHT_LANE_BIAS_PX = 10
        self.DIVIDER_FOLLOW_OFFSET_PX = 155

    def update_bev_transform(self, src_pts):
        self.SRC_PTS = np.float32(src_pts)
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)

    def process(self, raw_frame, dt=0.033, extra_offset_px=0.0,
                nav_state="NORMAL", velocity_ms=0.0, last_steering=0.0,
                upcoming_curve="STRAIGHT", pitch_rad=0.0):
        
        # 1. Dynamic Pitch Adjustments 
        if abs(pitch_rad) > 0.001:
            shift_px = int(pitch_rad * 400)
            dyn_src = self.SRC_PTS.copy()
            dyn_src[0][1] += shift_px
            dyn_src[1][1] += shift_px
            M_use = cv2.getPerspectiveTransform(dyn_src, self.DST_PTS)
        else:
            M_use = self.M_forward
            
        if raw_frame.shape[:2] != (480, 640):
            process_frame = cv2.resize(raw_frame, (640, 480))
        else:
            process_frame = raw_frame

        warped_colour = cv2.warpPerspective(process_frame, M_use, (640, 480))
        
        # ─────────────────────────────────────────────────────────────────
        # 2. WHITE ROAD / BLACK LINES DETECTION MAGIC
        # ─────────────────────────────────────────────────────────────────
        gray = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2GRAY)
        
        # THRESH_BINARY_INV with adaptive mean threshold:
        # If a pixel is 15 units DARKER than its local 61x61 neighborhood mean, 
        # it gets flipped to pure white (255) for the tracker to read.
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 61, 15)
        
        # Clean up small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # ─────────────────────────────────────────────────────────────────
        
        # 3. Update Tracker
        sl, sr, line_dbg = self.tracker.update(warped_binary)
        
        y_eval = 400.0
        
        # 4. Estimate Lane Width
        lane_width_px = 300.0
        if sl is not None and sr is not None:
            lane_width_px = np.polyval(sr, y_eval) - np.polyval(sl, y_eval)
            lane_width_px = np.clip(lane_width_px, 180, 520) 
        
        # 5. Target Selection Logic
        target_x = 320.0
        anchor = "LOST"
        def ev(fit): return float(np.polyval(fit, y_eval))

        if sl is not None and sr is not None:
            target_x = (ev(sl) + ev(sr)) / 2.0 + self.RIGHT_LANE_BIAS_PX
            anchor = "RL_DUAL"
        elif sr is not None:
            target_x = ev(sr) - (lane_width_px / 2.0)
            anchor = "RL_FROM_EDGE"
        elif sl is not None:
            target_x = ev(sl) + self.DIVIDER_FOLLOW_OFFSET_PX
            anchor = "DIVIDER_FOLLOW"
            
        target_x += extra_offset_px
        
        # 6. Confidence & Curvature
        conf = 1.0 if (sl is not None and sr is not None) else 0.5 if (sl is not None or sr is not None) else 0.0
        
        curv = 0.0
        fit = sr if sr is not None else sl
        if fit is not None:
            a, b = fit[0], fit[1]
            denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
            curv = abs(2.0 * a) / max(denom, 1e-6)
            
        # 7. Heading Angle
        heading_rad = 0.0
        heading_conf = conf
        
        def _lane_heading(f, y):
            return math.atan2(-(np.polyval(f, y - 50) - np.polyval(f, y)), 50)
            
        if sl is not None and sr is not None:
            h_sl = _lane_heading(sl, y_eval)
            h_sr = _lane_heading(sr, y_eval)
            heading_rad = (h_sl + h_sr) / 2.0
            
            angle_diff = abs((h_sl - h_sr + math.pi) % (2 * math.pi) - math.pi)
            heading_conf = max(0.0, 1.0 - angle_diff / math.radians(45))
        elif sl is not None:
            heading_rad = _lane_heading(sl, y_eval)
            heading_conf = 0.4
        elif sr is not None:
            heading_rad = _lane_heading(sr, y_eval)
            heading_conf = 0.4

        return PerceptionResult(
            warped_binary=warped_binary,
            lane_dbg=line_dbg,
            sl=sl, sr=sr,
            target_x=target_x,
            lateral_error_px=target_x - 320.0,
            anchor=anchor,
            confidence=conf,
            lane_width_px=lane_width_px,
            curvature=curv,
            heading_rad=heading_rad,
            heading_conf=heading_conf,
            y_eval=y_eval,
            optical_yaw_rate=0.0,  
            optical_vel=0.0        
        )