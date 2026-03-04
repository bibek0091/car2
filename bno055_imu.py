import smbus
import time
import math

BNO_ADDR = 0x29


class BNO055_IMU:

    def __init__(self, bus_num=1):
        self.bus = smbus.SMBus(bus_num)
        self._initialize_sensor()

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

        heading = self._read16(0x1A) / 16.0
        roll = self._read16(0x1C) / 16.0
        pitch = self._read16(0x1E) / 16.0

        return heading, roll, pitch

    def read_yaw(self):

        heading, _, _ = self.read_orientation()

        return math.radians(heading)


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