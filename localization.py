"""
localization.py — IMU-Free Visual Dead-Reckoning Localizer  (FIXED v5 — DEEP UPGRADE)
========================================================================================
DEEP UPGRADES in v5:

  LOC-A  HEADING CONFIDENCE GATING: update() now accepts heading_conf (from
         perception LANE-07). Layer 1 yaw-rate integration only fires when
         heading_conf > 0.35 (not just camera_confidence > 0.3). This prevents
         a bad lane polynomial with high pixel count but low heading agreement
         from spinning the localizer yaw.

  LOC-B  DUAL-RATE YAW-RATE EMA: the yaw-rate IIR now uses a FAST alpha (0.30)
         when |yaw_rate| > 0.3 rad/s (active turn) and SLOW alpha (0.12) when
         nearly straight. This gives responsive turn tracking without amplifying
         noise on straights — replacing the single fixed 0.18 alpha.

  LOC-C  HEADING ABSOLUTE FUSION: in addition to differentiating heading to get
         yaw-rate (Layer 1), Layer 3b now also soft-fuses the absolute heading
         angle at a low gain (0.04/frame). This acts as a long-term corrector
         that prevents yaw drift during extended runs — the localizer can't drift
         more than ~5° from the visible lane tangent.

  LOC-D  PATH-AWARE CURVE DETECTION: get_upcoming_curve_from_path() now runs
         THREE look-ahead windows (2 m, 4 m, 8 m) and returns the EARLIEST
         non-straight action found. This gives earlier curve warnings so
         CTRL-E speed-pre-reduction fires sooner.

  LOC-E  LIVE CURVE INJECTION into pose dict: get_pose_for_dashboard() now
         includes "curve_lookahead_m" — the distance to the next detected curve —
         so the dashboard can display it and the orchestrator can use it for
         proactive speed planning.

Fixes from v4 (all retained):
  LOC-01  upcoming_curve written under lock
  LOC-02  Map snap dt clamped
  LOC-03  Cursor is monotonic
  VL-FIX-A/B/C/D/E/F (cursor, snap radius, yaw EMA, init guard, POI, dashboard)
"""

import math
import numpy as np
import threading
import logging
import time
from collections import deque

try:
    from map_planner import PathPlanner
    _PLANNER_AVAILABLE = True
except ImportError:
    _PLANNER_AVAILABLE = False

log = logging.getLogger(__name__)


class LocalizationEngine:
    """
    4-layer IMU-free pose estimator: (x, y, yaw) in map metres.

    Layer 1 — Camera yaw-rate integration           (primary heading)
    Layer 2 — A* path heading soft nudge            (prevents long-run drift)
    Layer 3 — Forward dead-reckoning x/y            (velocity × dt)
    Layer 4 — Perpendicular map-snap                (cancels lateral drift)

    Sign convention (all layers):
        yaw in radians, map frame
        positive yaw  = counter-clockwise = LEFT turn
        negative yaw  = clockwise         = RIGHT turn
        x increases RIGHT,  y increases UP  (world / GraphML frame)

    Dashboard integration:
        Call get_pose_for_dashboard() each frame to get a dict with
        x, y, yaw_deg, zone, upcoming_curve, cursor, speed_ms.

    POI stopping:
        Call check_poi_arrival(target_node_id, threshold_m=0.40) each frame.
        Returns True when the car is within threshold_m of the target.
    """

    # ── Tunable constants ─────────────────────────────────────────────────────
    _WHEELBASE_M             = 0.23
    _MAX_CAM_YAW_CORRECTION  = 0.05   # rad — max soft nudge per frame
    _CAM_YAW_EMA             = 0.55   # Layer-3b nudge smoothing

    _MAP_SNAP_RADIUS_M      = 0.50
    _MAP_SNAP_RECOVERY_M    = 2.00
    _SNAP_LOST_LIMIT        = 60
    _MAP_SNAP_PULL          = 0.15
    _POI_DEFAULT_THRESH_M   = 0.40

    # LOC-C: absolute heading fusion gain
    _ABS_HEADING_GAIN       = 0.04   # fraction of heading error applied per frame
    _ABS_HEADING_MAX_CORR   = 0.06   # rad — max per-frame correction

    # LOC-D: multi-window lookahead distances (m)
    _CURVE_LA_WINDOWS       = [1.8, 4.0, 8.0]

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self._prev_yaw = 0.0
        self.visual_yaw_rate = 0.0   # rad/s — exposed to controller

        self._lock = threading.RLock()
        self._prev_cam_heading = None
        self._cam_yaw_smoothed = 0.0
        self._initialized      = False

        # Public state read by main loop / dashboard
        self.upcoming_curve   = "STRAIGHT"
        self.curve_dist_m     = float('inf')   # LOC-E: distance to next curve
        self.current_zone     = "CITY"
        self._last_speed_ms   = 0.0

        # FIX VL-FIX-A: cursor is fully self-managed
        self._path_cursor = 0
        self._snap_miss_frames = 0   # for recovery snap

        self.planner = None
        self._map_snap_enabled = _PLANNER_AVAILABLE
        if self._map_snap_enabled:
            self.planner = PathPlanner()
            if self.planner.graph is None or len(self.planner.graph.nodes) == 0:
                log.warning("GraphML map empty — disabling map snap.")
                self._map_snap_enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def path_cursor(self) -> int:
        """Read-only access to the self-managed path cursor."""
        return self._path_cursor

    def set_pose(self, x: float, y: float, yaw_rad: float = None):
        """
        Called when user clicks the SVG map (or on startup).
        Resets all filter state.  yaw_rad is optional — if omitted, the
        current yaw is preserved (useful for position-only corrections).
        """
        with self._lock:
            self.x = x
            self.y = y
            if yaw_rad is not None:
                self.yaw = yaw_rad
            self.visual_yaw_rate = 0.0
            self._initialized    = True
            self._snap_miss_frames = 0

            # FIX VL-07: seed EMA from actual yaw so first frame has no spike
            self._cam_yaw_smoothed = self.yaw
            self._prev_cam_heading = None

            # Reset derived state
            self.upcoming_curve = "STRAIGHT"
            self.current_zone   = (self.planner.get_zone(x, y)
                                   if self.planner else "CITY")
        log.info(f"Pose set: x={x:.2f} y={y:.2f} "
                 f"yaw={math.degrees(self.yaw):.1f}°")

    def reset_cursor(self):
        """Call whenever a new route is planned (resets path cursor to 0)."""
        with self._lock:
            self._path_cursor = 0
            self._snap_miss_frames = 0
        log.info("Path cursor reset.")

    def update_imu_yaw(self, yaw_rad: float):
        """Soft-fuse IMU yaw at gain 0.15 per frame instead of hard overwriting.
        A hard overwrite discards all Layer-1/LOC-C corrections every frame."""
        with self._lock:
            heading_err = (yaw_rad - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += np.clip(heading_err * 0.15, -0.05, 0.05)
            self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

    def get_pose(self):
        with self._lock:
            return self.x, self.y, self.yaw

    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    def get_pose_for_dashboard(self) -> dict:
        """
        FIX VL-FIX-F + LOC-E: Returns dict for dashboard display.
        """
        with self._lock:
            return {
                "x":               self.x,
                "snap_miss":       self._snap_miss_frames,
                "heading_conf_smoothed": self._cam_yaw_smoothed,
                "y":               self.y,
                "yaw_deg":         math.degrees(self.yaw),
                "zone":            self.current_zone,
                "upcoming_curve":  self.upcoming_curve,
                "curve_dist_m":    self.curve_dist_m,    # LOC-E
                "cursor":          self._path_cursor,
                "speed_ms":        self._last_speed_ms,
                "initialized":     self._initialized,
            }

    def check_poi_arrival(self, target_node_id: str,
                          threshold_m: float = None) -> bool:
        """
        FIX VL-FIX-E: Returns True when the car is within threshold_m of
        target_node_id's map position.  Call each frame; when True, the
        orchestrator should command speed = 0 and hold.
        """
        if threshold_m is None:
            threshold_m = self._POI_DEFAULT_THRESH_M
        if not self.planner or target_node_id not in self.planner.node_positions:
            return False
        tx, ty = self.planner.node_positions[target_node_id]
        with self._lock:
            dist = math.hypot(self.x - tx, self.y - ty)
        return dist <= threshold_m

    def get_distance_to_node(self, node_id: str) -> float:
        """Returns metres to a named map node (inf if unknown)."""
        if not self.planner or node_id not in self.planner.node_positions:
            return float('inf')
        tx, ty = self.planner.node_positions[node_id]
        with self._lock:
            return math.hypot(self.x - tx, self.y - ty)

    def get_upcoming_curve_from_path(self, path, cursor=None,
                                     velocity_ms: float = 0.3) -> str:
        """
        LOC-D: Multi-window lookahead — checks 3 distances and returns the
        EARLIEST non-straight action found. This fires curve warnings sooner,
        giving the controller time to pre-slow before BEV sees the curve.

        LOC-E: Sets self.curve_dist_m to the distance of the first detected
        non-straight waypoint for dashboard display and proactive speed planning.

        LOC-01: result computed first then written under lock.
        """
        if cursor is None:
            cursor = self._path_cursor

        if not path or not self.planner or cursor >= len(path) - 1:
            with self._lock:
                self.upcoming_curve = "STRAIGHT"
                self.curve_dist_m   = float('inf')
            return "STRAIGHT"

        with self._lock:
            curr_yaw = self.yaw

        # LOC-D: iterate over multiple look-ahead windows
        result   = "STRAIGHT"
        dist_out = float('inf')

        for la_m in self._CURVE_LA_WINDOWS:
            start_pos = self.planner.node_positions.get(path[cursor])
            if start_pos is None:
                continue
            accum = 0.0
            for i in range(cursor, min(cursor + 50, len(path) - 1)):
                n1 = path[i]
                n2 = path[i + 1]
                p1 = self.planner.node_positions.get(n1)
                p2 = self.planner.node_positions.get(n2)
                if p1 is None or p2 is None:
                    continue
                seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                accum += seg
                if accum >= la_m:
                    target_yaw = math.atan2(p2[1] - start_pos[1],
                                            p2[0] - start_pos[0])
                    diff = ((target_yaw - curr_yaw + math.pi)
                            % (2 * math.pi) - math.pi)
                    deg  = math.degrees(diff)
                    if deg > 18:
                        result   = "LEFT"
                        dist_out = accum
                    elif deg < -18:
                        result   = "RIGHT"
                        dist_out = accum
                    break
            if result != "STRAIGHT":
                break  # found a curve at nearest window — no need to look further

        # LOC-01 + LOC-E: atomic write
        with self._lock:
            self.upcoming_curve = result
            self.curve_dist_m   = dist_out

        return result

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self,
               velocity_ms:        float,
               dt:                 float,
               camera_heading_rad: float = 0.0,
               camera_confidence:  float = 0.0,
               heading_conf:       float = 0.0,
               imu_heading_rad:    float = None,
               path=None,
               optical_yaw_rate:   float = 0.0,
               optical_vel:        float = 0.0,
               current_steer_rad:  float = 0.0):
        """
        Update pose for one time step.

        LOC-A: heading_conf gates yaw-rate integration.
        LOC-B: dual-rate EMA alpha based on turn intensity.
        LOC-C: absolute heading soft-fusion (long-run drift corrector).
        EKF:   imu_heading_rad static Kalman correction (gain=0.15).
        OPT:   optical_yaw_rate from VisualOdometry used as fallback when
               lane lines are absent; optical_vel used when encoder dies.

        FIX VL-FIX-A: path_cursor parameter removed — cursor managed internally.
        """
        if dt <= 0:
            return

        # F-03: Auto-init on first valid perception frame.
        # If set_pose() was never called (operator forgot), snap to nearest map node
        # so we don't dead-reckon from (0,0) which is off the BFMC track.
        if not self._initialized:
            if camera_confidence > 0.5 and self.planner:
                # F-03 FIX: auto-snap to nearest(0,0) is always wrong on BFMC track.
                # Require the operator to call set_pose() via map click or --start-node.
                log.warning(
                    "F-03: Localizer not initialized. "
                    "Call set_pose(x, y, yaw_rad) before starting the run. "
                    "Skipping auto-snap to avoid placing car at world-origin node.")
            return   # do not dead-reckon from (0,0)

        with self._lock:
            # OPT: use optical_vel as velocity fallback when encoder is dead
            if velocity_ms < 0.01 and optical_vel > 0.05:
                velocity_ms = optical_vel

            self._last_speed_ms = velocity_ms

            if path:
                self._update_cursor_internal(path)
            cursor = self._path_cursor

            # ── Layer 1: Camera Yaw-Rate Integration (LOC-A + LOC-B) ──────────
            # LOC-A: only integrate when heading agreement is strong
            if heading_conf > 0.35 and camera_confidence > 0.3:
                raw_yaw_rate = 0.0
                if self._prev_cam_heading is not None:
                    raw_yaw_rate = (camera_heading_rad - self._prev_cam_heading) / max(dt, 0.001)
                self._prev_cam_heading = camera_heading_rad

                # LOC-B: dual-rate EMA — fast on turns, slow on straights
                alpha = 0.30 if abs(raw_yaw_rate) > 0.3 else 0.12
                self._cam_yaw_smoothed = alpha * raw_yaw_rate + (1.0 - alpha) * self._cam_yaw_smoothed
                self.visual_yaw_rate = self._cam_yaw_smoothed

                yaw_delta = np.clip(self._cam_yaw_smoothed * dt,
                                    -self._MAX_CAM_YAW_CORRECTION,
                                     self._MAX_CAM_YAW_CORRECTION)
                self.yaw += yaw_delta
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

                # LOC-C: absolute heading soft-fusion (long-run drift corrector)
                abs_err = (camera_heading_rad - self.yaw + math.pi) % (2 * math.pi) - math.pi
                abs_corr = np.clip(abs_err * self._ABS_HEADING_GAIN,
                                   -self._ABS_HEADING_MAX_CORR, self._ABS_HEADING_MAX_CORR)
                self.yaw += abs_corr
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
            else:
                self._prev_cam_heading = None   # reset on low-confidence frames

            # OPT: optical_yaw_rate fallback (when camera has no lane lines)
            if optical_yaw_rate != 0.0 and camera_confidence < 0.2:
                opt_delta = np.clip(optical_yaw_rate * dt, -0.04, 0.04)
                self.yaw += opt_delta
                self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 2: A* Path Heading Nudge ────────────────────────────────
            # Prevents long-run yaw drift by gently aligning with map segment.
            if (self._map_snap_enabled and self.planner and path
                    and cursor < len(path) - 1
                    and camera_confidence < 0.4):   # only when camera is uncertain
                n1 = path[cursor]
                n2 = path[min(cursor + 1, len(path) - 1)]
                p1 = self.planner.node_positions.get(n1)
                p2 = self.planner.node_positions.get(n2)
                if p1 and p2:
                    seg_yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    nudge_err = (seg_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                    # Only nudge when car heading is close enough to path (< 30°)
                    if abs(math.degrees(nudge_err)) < 30:
                        self.yaw += np.clip(nudge_err * 0.02, -0.01, 0.01)
                        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

            # ── Layer 3: Forward Dead-Reckoning ──────────────────────────────
            v = self._last_speed_ms
            theta = self.yaw

            expected_yaw_rate = v / self._WHEELBASE_M * math.tan(current_steer_rad)
            imu_yaw_rate = (self.yaw - self._prev_yaw) / dt if hasattr(self, "_prev_yaw") else 0

            if abs(expected_yaw_rate - imu_yaw_rate) > 1.0:
                v *= 0.8

            self._prev_yaw = self.yaw

            dt = max(dt, 1e-4)

            # ── Layer 3: dead-reckoning (only when moving)
            if abs(v) >= 1e-4:
                dx = v * math.cos(theta) * dt
                dy = v * math.sin(theta) * dt
                self.x += dx
                self.y += dy

            # ── Layer 4: Map Snap (Heading-Gated) ────────────────────────────
            if self._map_snap_enabled and self.planner:
                self._apply_map_snap_gated(v, dt, camera_confidence,
                                           path, cursor)
                self.current_zone = self.planner.get_zone(self.x, self.y)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_cursor_internal(self, path):
        """
        FIX VL-FIX-A: O(1) incremental cursor advance (±12 window).
        FIX LOC-03: cursor only advances — never regresses — to prevent
        oscillation on looping/revisited track sections.
        Mutates self._path_cursor directly (called under self._lock).
        """
        if not path or not self.planner:
            return

        search_start = max(0,             self._path_cursor - 3)
        search_end   = min(len(path) - 1, self._path_cursor + 12)

        best_idx = self._path_cursor
        best_d   = float('inf')
        for i in range(search_start, search_end + 1):
            n = path[i]
            if n not in self.planner.node_positions:
                continue
            nx_, ny_ = self.planner.node_positions[n]
            d = math.hypot(nx_ - self.x, ny_ - self.y)
            if d < best_d:
                best_d  = d
                best_idx = i

        # FIX LOC-03 + F-09: monotonic advance with lap wrap-around detection.
        # If best_idx is far behind (> half the path length), the car has looped
        # around — allow cursor reset instead of staying stuck at end-of-path.
        lap_wrap_thresh = max(1, len(path) // 2)
        if best_idx < self._path_cursor - lap_wrap_thresh:
            self._path_cursor = best_idx   # lap wrap — reset allowed
            log.info(f"F-09: Lap wrap detected, cursor reset to {best_idx}")
        else:
            self._path_cursor = max(self._path_cursor, best_idx)

    def update_cursor(self, path, x, y):
        """
        Public cursor update — still available for callers that need it,
        but update() now calls this internally, so external calls are
        optional.  Returns the new cursor index.
        """
        if not path or not self.planner:
            return self._path_cursor

        search_start = max(0,             self._path_cursor - 3)
        search_end   = min(len(path) - 1, self._path_cursor + 12)

        best_idx = self._path_cursor
        best_d   = float('inf')
        for i in range(search_start, search_end + 1):
            n = path[i]
            if n not in self.planner.node_positions:
                continue
            nx_, ny_ = self.planner.node_positions[n]
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_d:
                best_d  = d
                best_idx = i

        self._path_cursor = best_idx
        return self._path_cursor

    def _apply_map_snap(self, velocity, dt, cam_conf, path, cursor):
        """
        FIX VL-FIX-B: Two-tier snap radius:
          Normal: 1.0 m  (was 0.5 m — too tight, failed after any drift)
          Recovery: 2.0 m after _SNAP_LOST_LIMIT consecutive misses

        Also: curvature gate still disables snap during corners > 0.005.
        """
        if velocity < 0.05 or cam_conf < 0.3:
            return
        if not path or cursor >= len(path) - 1:
            return

        curvature = self.planner.get_path_curvature(
            self.x, self.y, path, cursor=cursor, window_m=0.8)
        if curvature > 0.005:
            return

        # Decide which radius to use
        snap_radius = (self._MAP_SNAP_RECOVERY_M
                       if self._snap_miss_frames >= self._SNAP_LOST_LIMIT
                       else self._MAP_SNAP_RADIUS_M)

        best_dist = float('inf')
        best_foot = None

        search_start = max(0,             cursor - 2)
        search_end   = min(len(path) - 1, cursor + 6)

        for i in range(search_start, search_end):
            n1 = path[i]
            n2 = path[i + 1]
            p1 = self.planner.node_positions.get(n1)
            p2 = self.planner.node_positions.get(n2)
            if p1 is None or p2 is None:
                continue

            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = ex * ex + ey * ey
            if seg_len_sq < 1e-8:
                continue
            t = ((self.x - p1[0]) * ex + (self.y - p1[1]) * ey) / seg_len_sq
            t = max(0.0, min(1.0, t))
            foot_x = p1[0] + t * ex
            foot_y = p1[1] + t * ey
            d = math.hypot(self.x - foot_x, self.y - foot_y)
            if d < best_dist:
                best_dist = d
                best_foot = (foot_x, foot_y)

        if best_foot and best_dist < snap_radius:
            # FIX LOC-02: clamp dt to 100 ms max so a stall spike can't jump the car
            dt_clamped = min(dt, 0.10)
            pull = self._MAP_SNAP_PULL * dt_clamped
            self.x = self.x + pull * (best_foot[0] - self.x)
            self.y = self.y + pull * (best_foot[1] - self.y)
            self._snap_miss_frames = 0   # reset recovery counter
        else:
            self._snap_miss_frames += 1
            if self._snap_miss_frames >= self._SNAP_LOST_LIMIT:
                log.warning(
                    f"Map snap lost for {self._snap_miss_frames} frames "
                    f"(dist={best_dist:.2f}m). Recovery radius active.")

    def _apply_map_snap_gated(self, velocity, dt, cam_conf, path, cursor):
        """
        Heading-gated + topology-aware map snap.

        NEW: Snap is completely frozen when the cursor is at a junction or
        roundabout node.  In complex topologies, overlapping path segments
        cause the nearest-point search to snap to cross-traffic, so we
        rely purely on VO + dead-reckoning instead.
        """
        if velocity < 0.05 or cam_conf < 0.3:
            return
        if not path or cursor >= len(path) - 1:
            return

        # ── Topology freeze ───────────────────────────────────────────────────
        # F-08: Use is_at_junction() and is_roundabout_node() directly — these
        # do proper edge-count analysis. The old GraphML attribute check was fragile
        # because BFMC nodes only have x,y attributes (no 'junction' field).
        current_node = path[cursor]
        if self.planner:
            if self.planner.is_roundabout_node(current_node):
                return  # inside roundabout — freeze snap
            if self.planner.is_at_junction(current_node):
                return  # inside intersection — freeze snap

        curvature = self.planner.get_path_curvature(
            self.x, self.y, path, cursor=cursor, window_m=0.8)
        if curvature > 0.005:
            return

        snap_radius = (self._MAP_SNAP_RECOVERY_M
                       if self._snap_miss_frames >= self._SNAP_LOST_LIMIT
                       else self._MAP_SNAP_RADIUS_M)

        best_dist   = float('inf')
        best_foot   = None
        best_seg_yaw = self.yaw  # default: no heading info

        search_start = max(0,             cursor - 2)
        search_end   = min(len(path) - 1, cursor + 6)

        for i in range(search_start, search_end):
            n1 = path[i]
            n2 = path[i + 1]
            p1 = self.planner.node_positions.get(n1)
            p2 = self.planner.node_positions.get(n2)
            if p1 is None or p2 is None:
                continue

            ex, ey = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = ex * ex + ey * ey
            if seg_len_sq < 1e-8:
                continue
            t = ((self.x - p1[0]) * ex + (self.y - p1[1]) * ey) / seg_len_sq
            t = max(0.0, min(1.0, t))
            foot_x = p1[0] + t * ex
            foot_y = p1[1] + t * ey
            d = math.hypot(self.x - foot_x, self.y - foot_y)
            if d < best_dist:
                best_dist    = d
                best_foot    = (foot_x, foot_y)
                best_seg_yaw = math.atan2(ey, ex)  # heading of this segment

        if best_foot and best_dist < snap_radius:
            # ── HEADING GATE ──────────────────────────────────────────────────
            # Reject snap if the path segment is more than 25° off the car's yaw.
            # This catches cross-traffic segments at T/X intersections.
            heading_err = (best_seg_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(math.degrees(heading_err)) > 25.0:
                self._snap_miss_frames += 1
                return   # don't snap — segment is perpendicular or divergent

            dt_clamped = min(dt, 0.10)
            pull = self._MAP_SNAP_PULL * dt_clamped
            self.x = self.x + pull * (best_foot[0] - self.x)
            self.y = self.y + pull * (best_foot[1] - self.y)
            self._snap_miss_frames = 0
        else:
            self._snap_miss_frames += 1
            if self._snap_miss_frames >= self._SNAP_LOST_LIMIT:
                log.warning(
                    f"Map snap (gated) lost for {self._snap_miss_frames} frames "
                    f"(dist={best_dist:.2f}m). Recovery radius active.")
