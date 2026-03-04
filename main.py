"""
main.py — Vision + IMU Autonomous Pilot (Simplified Dashboard)
===========================================================================
- Removed: Tkinter GUI, map_planner, localization, graph logic.
- Added: IMU Yaw Rate differentiation for Electronic Stability Control.
- Added: Clean, high-performance OpenCV telemetry dashboard.
- Strictly uses right-lane tracking and divider forcefields.
"""

import cv2
import time
import math
import numpy as np
import logging
import argparse

# Hardware & Subsystems
from perception import VisionPipeline
from control import Controller
from safety import SafetySupervisor

# Graceful hardware fallbacks
try:
    from STM32_SerialHandler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    class STM32_SerialHandler:
        def connect(self): return False
        def send_command(self, cmd, val): pass
        def disconnect(self): pass

try:
    from bno055_imu import BNO055_IMU
    _IMU_AVAILABLE = True
except ImportError:
    _IMU_AVAILABLE = False
    class BNO055_IMU:
        def get_yaw(self): return 0.0

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

class BFMC_Pilot:
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.base_speed = 40.0
        
        # --- Hardware Initialization ---
        self.serial = STM32_SerialHandler()
        self.connected = False if sim_mode else self.serial.connect()
        
        self.imu = BNO055_IMU()
        self.prev_yaw = self.imu.get_yaw()
        
        self.safety = SafetySupervisor()
        
        self.cam_ok = False
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.error(f"Camera init failed: {e}")

        # --- Stack Initialization ---
        self.vision = VisionPipeline()
        self.controller = Controller()

        # OpenCV Dashboard UI
        cv2.namedWindow("BFMC_Pilot_Dashboard", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Base Speed", "BFMC_Pilot_Dashboard", int(self.base_speed), 100, self._update_speed)
        
        self.last_time = time.time()
        self.fps = 0.0

    def _update_speed(self, val):
        self.base_speed = float(val)

    def compute_yaw_rate(self, dt):
        """Calculates rotational velocity (rad/s) from the IMU to feed the ESC."""
        current_yaw = self.imu.get_yaw()
        # Wrap angle difference to [-pi, pi] to prevent jumps when crossing 0/360
        delta_yaw = (current_yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
        self.prev_yaw = current_yaw
        return delta_yaw / max(dt, 0.001)

    def draw_dashboard(self, frame, perc_res, ctrl_res, imu_yaw_rate):
        """Creates a unified heads-up display combining camera and BEV tracker."""
        # 1. Prepare BEV debug image
        bev_dbg = perc_res.lane_dbg.copy()
        
        # Draw target crosshair
        cv2.circle(bev_dbg, (int(ctrl_res.target_x), int(perc_res.y_eval)), 8, (0, 255, 0), -1)
        cv2.line(bev_dbg, (int(ctrl_res.target_x), int(perc_res.y_eval)), (320, 480), (0, 255, 0), 2)
        cv2.line(bev_dbg, (320, 450), (320, 480), (0, 0, 255), 3) # Car center

        # 2. Resize original frame to fit next to BEV
        frame_resized = cv2.resize(frame, (640, 480))
        
        # 3. Stack images horizontally
        dashboard = np.hstack((frame_resized, bev_dbg))
        
        # 4. Draw Telemetry Overlay
        h, w = dashboard.shape[:2]
        overlay = dashboard.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, dashboard, 0.3, 0, dashboard)

        color_ok = (0, 255, 0)
        color_warn = (0, 165, 255)
        color_err = (0, 0, 255)

        anchor_color = color_ok if "DUAL" in ctrl_res.anchor else (color_warn if "ONLY" in ctrl_res.anchor else color_err)

        cv2.putText(dashboard, f"FPS: {self.fps:.1f} | Speed PWM: {ctrl_res.speed_pwm:.1f}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(dashboard, f"Steer: {ctrl_res.steer_angle_deg:.1f} deg | Yaw Rate: {math.degrees(imu_yaw_rate):.1f} deg/s", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(dashboard, f"Anchor: {ctrl_res.anchor}", (w // 2 + 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, anchor_color, 2)
                    
        cv2.putText(dashboard, f"Lane Width: {perc_res.lane_width_px:.0f}px", (w // 2 + 10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("BFMC_Pilot_Dashboard", dashboard)

    def run(self):
        log.info("Starting Vision+IMU Pilot. (Localization disabled)")
        
        try:
            while True:
                now = time.time()
                dt = now - self.last_time
                self.last_time = now
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 0.001))

                # 1. Fetch Image
                if self.cam_ok:
                    frame = self.picam2.capture_array()
                    self.safety.update_camera()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # 2. Vision Pipeline
                perc_res = self.vision.process(frame, dt=dt)

                # 3. IMU Dynamics
                imu_yaw_rate = self.compute_yaw_rate(dt)

                # 4. Control Logic (Vision + IMU stabilization)
                ctrl_res = self.controller.compute(
                    perc_res=perc_res,
                    nav_state="NORMAL",
                    velocity_ms=0.0,      # PWM-based control doesn't strictly need precise m/s
                    dt=dt,
                    base_speed=self.base_speed,
                    imu_yaw_rate=imu_yaw_rate
                )

                # 5. Safety & Actuation
                if self.safety.should_stop():
                    speed_cmd = 0.0
                    steer_cmd = 0.0
                else:
                    speed_cmd = ctrl_res.speed_pwm
                    steer_cmd = ctrl_res.steer_angle_deg

                if self.connected:
                    self.serial.send_command("speed", str(int(speed_cmd)))
                    self.serial.send_command("steer", str(int(steer_cmd * 10))) # Assuming API needs deg * 10

                # 6. Render Dashboard
                self.draw_dashboard(frame, perc_res, ctrl_res, imu_yaw_rate)

                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    break

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        if self.connected:
            self.serial.send_command("speed", "0")
            self.serial.send_command("steer", "0")
            self.serial.disconnect()
        if self.cam_ok:
            self.picam2.stop()
        cv2.destroyAllWindows()
        log.info("Pilot stopped safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC Vision+IMU Pilot")
    parser.add_argument("--sim", action="store_true", help="Run without hardware")
    args = parser.parse_args()

    pilot = BFMC_Pilot(sim_mode=args.sim)
    pilot.run()