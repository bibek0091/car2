import smbus
import time
import math
import threading

BNO_ADDR = 0x28
OPR_MODE = 0x3D
CHIP_ID = 0x00
QUAT_W_LSB = 0x20

class BNO055_IMU:
    def __init__(self, bus_id=1):
        self.bus = smbus.SMBus(bus_id)
        self.yaw_rad = 0.0
        self.offset = 0.0
        self.running = False
        self.lock = threading.Lock()
        self._initialize_sensor()

    def _read8(self, reg):
        return self.bus.read_byte_data(BNO_ADDR, reg)

    def _write8(self, reg, value):
        self.bus.write_byte_data(BNO_ADDR, reg, value)

    def _read16(self, reg):
        lsb = self.bus.read_byte_data(BNO_ADDR, reg)
        msb = self.bus.read_byte_data(BNO_ADDR, reg + 1)
        value = (msb << 8) | lsb
        if value > 32767:
            value -= 65536
        return value

    def _initialize_sensor(self):
        time.sleep(1)
        chip = self._read8(CHIP_ID)
        if chip != 0xA0:
            raise RuntimeError("BNO055 not detected")

        self._write8(OPR_MODE, 0x00)
        time.sleep(0.05)
        self._write8(OPR_MODE, 0x0C)
        time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _loop(self):
        while self.running:
            qw = self._read16(QUAT_W_LSB) / 16384.0
            qx = self._read16(QUAT_W_LSB + 2) / 16384.0
            qy = self._read16(QUAT_W_LSB + 4) / 16384.0
            qz = self._read16(QUAT_W_LSB + 6) / 16384.0

            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )

            with self.lock:
                self.yaw_rad = yaw + self.offset

            time.sleep(0.02)

    def get_yaw(self):
        with self.lock:
            return self.yaw_rad

    def set_offset(self, offset_rad):
        with self.lock:
            self.offset = offset_rad
