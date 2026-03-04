import time

class SafetySupervisor:
    def __init__(self):
        self.yolo_last    = time.time()
        self.camera_last  = time.time()
        self.serial_last  = time.time()
        self.encoder_last = time.time()   # updated each time encoder velocity is read

    def update_yolo(self):
        self.yolo_last = time.time()

    def update_camera(self):
        self.camera_last = time.time()

    def update_serial(self):
        self.serial_last = time.time()

    def update_encoder(self):
        self.encoder_last = time.time()

    def should_stop(self):
        now = time.time()
        if now - self.yolo_last > 2.0:       # was 0.5s — YOLO runs in background thread
            return True
        if now - self.camera_last > 0.5:
            return True
        if now - self.serial_last > 0.5:
            return True
        if now - self.encoder_last > 1.0:    # encoder silent for 1 s — speed sensing lost
            return True
        return False

    def safe_speed_override(self):
        """Returns 0.0 to force stop, 1.0 for normal operation."""
        if self.should_stop():
            return 0.0
        return 1.0
