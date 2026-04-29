import hid
import time

class MotorStimulator:
    def __init__(self, vendor_id=0xCafe, product_id=0x4001):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()
        self.device.open(self.vendor_id, self.product_id)
        self.device.set_nonblocking(1)

    def set_carrier_frequency(self, freq: int):
        if (freq < 32) or (freq > 1028):
            raise(Exception("Frequency must be greater than or equal to 32-Hz, and less-than or equal to 1028-Hz."))
        report = MotorStimulator.init_pwm_command_report()
        report[1] = freq >> 8
        report[2] = (freq << 4) & 0xFF 
        # print(f"Sending: {report.hex()}")  # DEBUG
        self.device.write(report)

    def send_single_motor_command(self, motor: int, pct: float, brake: bool = False, reverse: bool = False):
        report = MotorStimulator.init_motor_command_report()
        scaled = round(pct * 4095)
        report[1+motor*2] = scaled >> 8
        report[2+motor*2] = ((scaled << 4) | (brake << 1) | reverse) & 0xFF
        # print(f"Sending: {report.hex()}")  # DEBUG
        self.device.write(report)

    def send_multi_motor_command(self, *args):
        report = MotorStimulator.init_motor_command_report()
        for (motor, pct, brake, reverse) in args:
            scaled = round(pct * 4095)
            report[1+motor*2] = scaled >> 8
            report[2+motor*2] = ((scaled << 4) | (brake << 1) | reverse) & 0xFF
        # print(f"Sending: {report.hex()}")  # DEBUG
        self.device.write(report)

    def read_encoder_feedback(self):
        """
        Read encoder feedback (4x int16_t = 8 bytes). Returns a list of 4 signed integers.
        """
        response = self.device.read(9)  # 1 byte report ID + 8 byte payload
        if len(response) == 9:
            return [
                int.from_bytes(response[1+i*2:3+i*2], byteorder='little', signed=True)
                for i in range(4)
            ]
        return None

    @staticmethod
    def init_motor_command_report():
        return bytearray([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    @staticmethod
    def init_pwm_command_report():
        return bytearray([0x03, 0x00, 0x00])


    def close(self):
        self.device.close()
