import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import serial
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from numpy.typing import NDArray

from nml.imu_serial_reader import data_len_from_pt, f32_be, i16_be, u16_be, checksum
from nml.processing import ArmIncrementalKinematics, euler_xyz_rad_to_quaternion


@dataclass
class UpperImuState:
    quaternion: NDArray[np.float64]
    last_update: float


class MindRoveOrientationEstimator:
    """Simple complementary filter for forearm orientation from MindRove acc+gyro."""

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def update(self, acc_xyz: NDArray[np.float64], gyro_xyz: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        # Integrate gyro (assumes deg/s; convert to rad/s)
        gx, gy, gz = np.deg2rad(gyro_xyz)
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        # Accelerometer tilt correction
        ax, ay, az = acc_xyz
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        self.roll = self.alpha * self.roll + self.beta * roll_acc
        self.pitch = self.alpha * self.pitch + self.beta * pitch_acc

        return euler_xyz_rad_to_quaternion(self.roll, self.pitch, self.yaw)


def parse_upper_quaternion_from_serial(ser: serial.Serial, previous_q: NDArray[np.float64]) -> NDArray[np.float64]:
    # Parse one packet if available; otherwise return previous quaternion.
    if ser.read(1) != b"s":
        return previous_q
    if ser.read(1) != b"n":
        return previous_q
    if ser.read(1) != b"p":
        return previous_q

    hdr = ser.read(2)
    if len(hdr) < 2:
        return previous_q
    pt, addr = hdr[0], hdr[1]

    dlen = data_len_from_pt(pt)
    data = ser.read(dlen)
    cks = ser.read(2)
    if len(data) != dlen or len(cks) != 2:
        return previous_q

    pkt_wo_cks = b"snp" + bytes([pt, addr]) + data
    if u16_be(cks) != checksum(pkt_wo_cks):
        return previous_q

    # 0x70..0x74 : Euler(deg) + rates + time packet
    if pt == 0xD4 and addr == 0x70 and dlen == 20:
        roll_raw = i16_be(data[0:2])
        pitch_raw = i16_be(data[2:4])
        yaw_raw = i16_be(data[4:6])
        roll_deg = roll_raw / 91.02222
        pitch_deg = pitch_raw / 91.02222
        yaw_deg = yaw_raw / 91.02222
        return euler_xyz_rad_to_quaternion(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )

    # Optional float packet (if firmware sends float Euler elsewhere, keep previous if unknown)
    if pt == 0xF0 and addr == 0x61 and dlen == 48:
        _ = [f32_be(data[i : i + 4]) for i in range(0, 48, 4)]

    return previous_q


def main() -> None:
    parser = argparse.ArgumentParser(description="Live dual-IMU incremental kinematics test (external serial upper IMU + MindRove forearm IMU).")
    parser.add_argument("--port", default="COM6", help="Upper-arm serial IMU COM port")
    parser.add_argument("--baud", type=int, default=115200, help="Upper-arm serial IMU baud")
    parser.add_argument("--upper-len", type=float, default=0.30, help="Upper arm length in meters")
    parser.add_argument("--fore-len", type=float, default=0.26, help="Forearm length in meters")
    parser.add_argument("--smooth-window", type=int, default=10, help="Smoothing window length")
    parser.add_argument("--alpha", type=float, default=0.98, help="Complementary filter alpha for MindRove orientation")
    parser.add_argument("--print-hz", type=float, default=20.0, help="Print rate limit")
    parser.add_argument("--mindrove-ip", default="192.168.4.1", help="MindRove board IP address")
    parser.add_argument("--mindrove-port", type=int, default=4210, help="MindRove board UDP/TCP port")
    parser.add_argument("--mindrove-timeout-ms", type=int, default=5000, help="MindRove connect timeout in milliseconds")
    parser.add_argument("--use-synth", action="store_true", help="Use synthetic board instead of MindRove hardware")
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if args.use_synth else BoardIds.MINDROVE_WIFI_BOARD
    input_params = MindRoveInputParams()
    input_params.ip_address = args.mindrove_ip
    input_params.ip_port = args.mindrove_port
    input_params.timeout = args.mindrove_timeout_ms
    board = BoardShim(board_id, input_params)
    board.prepare_session()

    sample_rate = BoardShim.get_sampling_rate(board_id)
    accel_channels = BoardShim.get_accel_channels(board_id)
    gyro_channels = BoardShim.get_gyro_channels(board_id)

    kin = ArmIncrementalKinematics(
        upper_arm_length_m=args.upper_len,
        forearm_length_m=args.fore_len,
        smooth_window=args.smooth_window,
    )
    forearm_est = MindRoveOrientationEstimator(alpha=args.alpha)
    upper_state = UpperImuState(quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), last_update=time.time())

    board_name = "SYNTHETIC" if args.use_synth else "MindRove"
    print(f"Starting {board_name} stream...")
    board.start_stream(256)

    print(f"Opening serial IMU on {args.port} @ {args.baud}...")
    with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
        print("Running live dual-IMU test. Press Ctrl+C to stop.")

        last_print = 0.0
        last_t = time.time()
        try:
            while True:
                # Update upper-arm quaternion from serial packets
                upper_state.quaternion = parse_upper_quaternion_from_serial(ser, upper_state.quaternion)

                # Pull latest MindRove samples
                data = board.get_board_data()
                if data is None or data.shape[1] < 1:
                    time.sleep(0.002)
                    continue

                acc = data[accel_channels, -1].astype(np.float64)
                gyro = data[gyro_channels, -1].astype(np.float64)

                now = time.time()
                dt = max(1.0 / sample_rate, now - last_t)
                last_t = now

                forearm_q = forearm_est.update(acc, gyro, dt)
                delta_p, delta_q, delta_p_avg, delta_q_avg = kin.update(upper_state.quaternion, forearm_q)

                if (now - last_print) >= (1.0 / max(1e-3, args.print_hz)):
                    last_print = now
                    print(
                        f"dP=({delta_p[0]:+0.5f},{delta_p[1]:+0.5f},{delta_p[2]:+0.5f})  "
                        f"dPavg=({delta_p_avg[0]:+0.5f},{delta_p_avg[1]:+0.5f},{delta_p_avg[2]:+0.5f})  "
                        f"dQavg=({delta_q_avg[0]:+0.5f},{delta_q_avg[1]:+0.5f},{delta_q_avg[2]:+0.5f},{delta_q_avg[3]:+0.5f})"
                    )
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                if board.is_prepared():
                    board.release_session()
            except Exception:
                pass


if __name__ == "__main__":
    main()
