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
        if now - self.yolo_last > 5.0:       # was 2.0s
            print("SAFETY STOP: yolo timeout")
            return "yolo timeout"
        if now - self.camera_last > 3.0:     # was 1.5s
            print("SAFETY STOP: camera timeout")
            return "camera timeout"
        if now - self.serial_last > 2.0:     # was 0.5s
            print("SAFETY STOP: serial timeout")
            return "serial timeout"
        if now - self.encoder_last > 3.0:    # was 1.0s
            print("SAFETY STOP: encoder timeout")
            return "encoder timeout"
        return False

    def safe_speed_override(self):
        """Returns 0.0 to force stop, 1.0 for normal operation."""
        if self.should_stop():
            return 0.0
        return 1.0
