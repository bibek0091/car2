"""
safety.py — Streamlined Safety Supervisor
===========================================================================
Monitors the critical Vision pipeline. If the camera freezes or drops frames 
for more than 2 seconds, it triggers an emergency stop to prevent the car 
from driving blind.
"""

import time
import logging

log = logging.getLogger(__name__)

class SafetySupervisor:
    def __init__(self):
        # Initialize the watchdog timer
        self.camera_last = time.time()
        self._stop_triggered = False

    def update_camera(self):
        """Called by main.py every time a new frame is successfully grabbed."""
        self.camera_last = time.time()
        if self._stop_triggered:
            log.info("SAFETY RESOLVED: Camera feed restored.")
            self._stop_triggered = False

    def should_stop(self):
        """Checks if any critical systems have timed out."""
        now = time.time()
        
        # If no frame is received for 2.0 seconds, trigger E-STOP
        if now - self.camera_last > 2.0:
            if not self._stop_triggered:
                log.error("SAFETY STOP: Camera timeout! Car is driving blind.")
                self._stop_triggered = True
            return True
            
        return False