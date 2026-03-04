try:
    import smbus as _smbus_mod
except ImportError:
    _smbus_mod = None
import time
import math

BNO_ADDR = 0x29


class BNO055_IMU:

    def __init__(self, bus_num=1):
        self._yaw_offset = 0.0
        self._sim_mode   = False
        self._sim_yaw    = 0.0
        try:
            import smbus as _smbus
            self.bus = _smbus.SMBus(bus_num)
            self._initialize_sensor()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"BNO055 hardware unavailable ({e}) — using software yaw simulation.")
            self.bus       = None
            self._sim_mode = True

    def _write_reg(self, reg, value):
        self.bus.write_byte_data(BNO_ADDR, reg, value)
        time.sleep(0.02)

    def _read8(self, reg):
        return self.bus.read_byte_data(BNO_ADDR, reg)

    def _read16(self, reg):
        lsb = self.bus.read_byte_data(BNO_ADDR, reg)
        msb = self.bus.read_byte_data(BNO_ADDR, reg + 1)
        value = (msb << 8) | lsb
        if value > 32767:
            value -= 65536
        return value

    def _initialize_sensor(self):
        print("Starting BNO055")
        time.sleep(1)
        chip_id = self._read8(0x00)
        print("Chip ID:", hex(chip_id))
        if chip_id != 0xA0:
            raise RuntimeError("BNO055 not detected")
        print("Sensor detected")
        # CONFIG MODE
        self._write_reg(0x3D, 0x00)
        time.sleep(0.05)
        # NDOF MODE
        self._write_reg(0x3D, 0x0C)
        time.sleep(0.2)
        print("Fusion Mode Activated")

    def read_orientation(self):
        if self._sim_mode:
            return 0.0, 0.0, 0.0
        heading = self._read16(0x1A) / 16.0
        roll    = self._read16(0x1C) / 16.0
        pitch   = self._read16(0x1E) / 16.0
        return heading, roll, pitch

    def read_yaw(self) -> float:
        if self._sim_mode:
            return self._sim_yaw + self._yaw_offset
        heading, _, _ = self.read_orientation()
        return math.radians(heading) + self._yaw_offset

    def update_sim_yaw(self, delta_rad: float):
        """Called from pilot loop in sim mode to integrate steering-based yaw."""
        if self._sim_mode:
            self._sim_yaw += delta_rad

    def start(self):
        """No-op — sensor is initialised in __init__. Exists for API compatibility."""
        pass

    def get_yaw(self) -> float:
        """Returns current yaw in radians (same as read_yaw)."""
        return self.read_yaw()

    def set_offset(self, offset_rad: float):
        """Stores a heading offset applied to all subsequent get_yaw() calls."""
        self._yaw_offset = offset_rad


# ---------------------------
# Standalone Test
# ---------------------------

if __name__ == "__main__":

    print("Running standalone IMU test")

    imu = BNO055_IMU()

    while True:

        yaw, roll, pitch = imu.read_orientation()

        print("Yaw:", yaw, "Roll:", roll, "Pitch:", pitch)

        time.sleep(0.1)