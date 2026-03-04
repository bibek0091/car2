"""
control.py — Vision + IMU Stabilized Controller
===========================================================================
- Strictly Right-Lane Driving: Explicit left/right divider identification and centering.
- Active IMU Damping: Counter-steers against yaw-rate to eliminate oscillations.
- Dynamic Lighting Mode: Steering stiffens and relies on IMU when visual confidence drops.
- Aggressive Divider Guard: Hard forcefields strictly prevent touching boundaries.
- Localization dependencies completely removed.
"""

import math
import numpy as np
from dataclasses import dataclass

@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm:       float
    target_x:        float
    anchor:          str
    lookahead_px:    float
    steer_ff_deg:    float = 0.0   
    steer_react_deg: float = 0.0   


# ═══════════════════════════════════════════════════════════════════════════════
class StanleyController:
    """
    Pure kinematic Stanley controller for Vision-Only Tracking
    """
    def __init__(self, k: float = 1.2, ks: float = 0.2):
        self.k  = k
        self.ks = ks

    def compute(self, target_x_px: float, heading_rad: float,
                velocity_ms: float, lane_width_px: float):
        
        ppm  = max(lane_width_px, 50) / 0.45    # pixels per metre
        ce_m = (320.0 - target_x_px) / ppm      # cross-track error (metres)

        # Velocity-scaled cross-track gain to prevent startup oscillation.
        k_eff = self.k * min(1.0, velocity_ms / 0.25)

        # Reactive Stanley term
        reactive_rad = heading_rad + math.atan2(k_eff * ce_m, velocity_ms + self.ks)
        return math.degrees(reactive_rad)


# ═══════════════════════════════════════════════════════════════════════════════
class DividerGuard:
    """
    Aggressive Force-Field around lane boundaries.
    Strictly forbids touching the lines. Left-divider (oncoming traffic) is highly repelled.
    """
    DIVIDER_SAFE_PX = 110   # Extremely strict margin from the left center-divider
    EDGE_SAFE_PX    = 90    # Strict margin from the right edge drop-off
    GAIN            = 0.15  # Aggressive proportional correction gain
    MAX_CORR        = 15.0  # Max correction per trigger (degrees)
    DEADBAND_PX     = 2     # Very small deadband for near-instant reaction

    def apply(self, steer_angle, left_divider, right_edge, y_eval=440, car_x=320):
        correction  = 0.0
        speed_scale = 1.0
        triggered   = False

        # --- LEFT: Center Divider (Strictly Do Not Touch) ---
        div_corr = 0.0
        if left_divider is not None:
            div_x = float(np.polyval(left_divider, y_eval))
            gap   = car_x - div_x          # positive = car is right of divider
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err * 1.5, self.MAX_CORR)   # Steer right (+), x1.5 multiplier for extra safety
                speed_scale = min(speed_scale, max(0.4, 1.0 - err / 80.0))
                triggered   = True

        # --- RIGHT: Outer Edge ---
        edge_corr = 0.0
        if right_edge is not None:
            edge_x = float(np.polyval(right_edge, y_eval))
            gap    = edge_x - car_x        # positive = car is left of edge
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)  # Steer left (-)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))
                triggered   = True

        # If both fire, the left divider wins (prevent head-on collision).
        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ═══════════════════════════════════════════════════════════════════════════════
class Controller:

    STEER_EMA_SLOW = 0.20   # EMA weight on straights (smooth)
    STEER_EMA_FAST = 0.45   # EMA weight on sharp turns (responsive)
    GUARD_EMA      = 0.60   # EMA weight for guard correction
    MAX_STEER      = 35.0   # hard clamp (degrees)
    MAX_STEER_RATE = 6.0    # cap change per frame to avoid servo overshoot

    # Vision-based Curvature Speed Reduction
    HIGH_CURV_THRESH = 0.003
    MED_CURV_THRESH  = 0.0015
    HIGH_CURV_SCALE  = 0.55
    MED_CURV_SCALE   = 0.75
    DUAL_SPEED_SCALE = 1.15

    # IMU Damping
    IMU_YAW_GAIN     = 0.65 # Counter-steering gain against rotational velocity

    def __init__(self):
        self.prev_steer   = 0.0
        self.smooth_steer = 0.0
        self.smooth_guard = 0.0
        self.guard        = DividerGuard()
        self.stanley      = StanleyController(k=1.2, ks=0.2)

    def compute(self, perc_res,
                nav_state:   str   = "NORMAL",
                velocity_ms: float = 0.0,
                dt:          float = 0.033,
                base_speed:  float = 50.0,
                imu_yaw_rate: float = 0.0) -> ControlOutput:  # <--- Added IMU Yaw Rate

        # ── 1. Explicit Divider Identification & Centering ──────────────────────
        left_divider = perc_res.sl
        right_edge   = perc_res.sr
        
        # We calculate the exact midpoint offset dynamically based on detected lane width
        center_offset_px = perc_res.lane_width_px / 2.0
        
        if left_divider is not None and right_edge is not None:
            # Both seen: Car drives perfectly in the middle
            target_x = (np.polyval(left_divider, perc_res.y_eval) + np.polyval(right_edge, perc_res.y_eval)) / 2.0
            anchor = "DUAL_LINES"
            
        elif left_divider is not None:
            # Left Divider identified: Push the car to the right by exactly half a lane
            target_x = np.polyval(left_divider, perc_res.y_eval) + center_offset_px
            anchor = "LEFT_DIVIDER_ONLY"
            
        elif right_edge is not None:
            # Right Edge identified: Push the car to the left by exactly half a lane
            target_x = np.polyval(right_edge, perc_res.y_eval) - center_offset_px
            anchor = "RIGHT_EDGE_ONLY"
            
        else:
            target_x = 320.0
            anchor = "LOST"

        # ── 2. Vision-Based Stanley Steering ────────────────────────────────────
        raw_steer = self.stanley.compute(
            target_x, 
            perc_res.heading_rad,
            velocity_ms, 
            perc_res.lane_width_px
        )

        # ── 3. Active IMU Yaw Damping (Electronic Stability Control) ────────────
        # Counter-steers against the car's rotational momentum to stop oscillations dead.
        # If the car snaps right (positive yaw rate), this subtracts steering angle.
        yaw_damping_deg = math.degrees(imu_yaw_rate) * self.IMU_YAW_GAIN
        raw_steer -= yaw_damping_deg

        # ── 4. Dynamic Lighting Adaptation & EMA ────────────────────────────────
        # If camera confidence drops (shadows, glare), stiffen steering and trust the IMU
        if perc_res.confidence < 0.40:
            alpha = 0.05  # Lock steering, heavily trust previous state and IMU damping
        else:
            steer_delta_abs = abs(raw_steer - self.smooth_steer)
            alpha = self.STEER_EMA_FAST if steer_delta_abs > 8.0 else self.STEER_EMA_SLOW
            
        self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
        steer_angle = self.smooth_steer

        # ── 5. Hardware Rate Limiting ──────────────────────────────────────────
        rate_delta  = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, steer_angle - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        self.prev_steer = steer_angle

        # ── 6. Strict Divider Guard (Visual Safety Bounds) ─────────────────────
        raw_steer_guarded, guard_spd, guard_on = self.guard.apply(
            steer_angle, left_divider, right_edge, y_eval=perc_res.y_eval)

        if anchor == "LOST":
            self.smooth_guard = 0.0
            guard_on = False
        else:
            guard_delta = raw_steer_guarded - steer_angle
            self.smooth_guard = (self.GUARD_EMA * guard_delta 
                                 + (1.0 - self.GUARD_EMA) * self.smooth_guard)
        
        steer_angle = steer_angle + self.smooth_guard
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle))

        # ── 7. Vision-Only Speed Profiling ─────────────────────────────────────
        speed = float(base_speed)
        curvature = perc_res.curvature

        if base_speed == 0:
            speed = 0.0
        elif nav_state == "ROUNDABOUT":
            speed = base_speed * 0.45
        elif "JUNCTION" in nav_state:
            speed = base_speed * 0.50
        elif curvature > self.HIGH_CURV_THRESH:
            speed = base_speed * self.HIGH_CURV_SCALE
        elif curvature > self.MED_CURV_THRESH:
            speed = base_speed * self.MED_CURV_SCALE
        elif anchor == "DUAL_LINES" and abs(steer_angle) < 8:
            speed = base_speed * self.DUAL_SPEED_SCALE
        elif abs(steer_angle) > 20:
            speed = base_speed * 0.55
        elif abs(steer_angle) > 10:
            speed = base_speed * 0.75

        # Crawl if lines are totally lost
        if anchor == "LOST":
            speed *= 0.3

        if guard_on:
            speed *= guard_spd

        # Floor constraint to prevent stalling in tight turns
        MINIMUM_DRIVE_PWM = 16.0
        if nav_state not in ("SYS_STOP", "STOPPED") and speed > 0:
            speed = max(speed, MINIMUM_DRIVE_PWM)

        return ControlOutput(
            steer_angle_deg = steer_angle,
            speed_pwm       = speed,
            target_x        = target_x,
            anchor          = anchor,
            lookahead_px    = 0.0,
            steer_ff_deg    = 0.0,
            steer_react_deg = steer_angle,
        )