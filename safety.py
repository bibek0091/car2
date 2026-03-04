import time

class SafetySupervisor:
    def __init__(self):
        self.yolo_last = time.time()
        self.camera_last = time.time()
        self.serial_last = time.time()

    def update_yolo(self):
        self.yolo_last = time.time()

    def update_camera(self):
        self.camera_last = time.time()

    def update_serial(self):
        self.serial_last = time.time()

    def should_stop(self):
        now = time.time()
        if now - self.yolo_last > 0.5:
            return True
        if now - self.camera_last > 0.5:
            return True
        if now - self.serial_last > 0.5:
            return True
        return False
