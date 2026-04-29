"""
Live test: verify upper-arm (serial IMU) vs forearm (MindRove) motion are observable separately.

Uses the same two-link wrist split as the paper (Eq. 2): p_wrist = R_SE @ p_SE + R_SW @ p_EW.

Run from repo root:
  python -m nml.test_imu_motion_separation --port COM6 --mindrove-ip 192.168.4.1

Protocols (do each ~10–20 s while watching columns):
  A — Hold forearm still; move only upper arm → upper RPY and |Δp_upper| should dominate.
  B — Hold upper arm still; rotate only forearm → forearm RPY and |Δp_fore| should dominate.
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Optional, Tuple

import numpy as np
import serial
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from numpy.typing import NDArray

from nml.imu_serial_reader import checksum, data_len_from_pt, i16_be, u16_be
from nml.processing import euler_xyz_rad_to_quaternion, wrist_position_decomposition
from nml.test_dual_imu_live import MindRoveOrientationEstimator


def _try_parse_upper_euler_serial(
    ser: serial.Serial,
    prev_q: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], Optional[Tuple[float, float, float]]]:
    """
    Non-blocking: try to read one full SNP packet. Returns (q, euler_deg_or_none).
    euler_deg is (roll, pitch, yaw) when a 0x70 Euler packet is decoded; else None.
    """
    if ser.read(1) != b"s":
        return prev_q, None
    if ser.read(1) != b"n":
        return prev_q, None
    if ser.read(1) != b"p":
        return prev_q, None

    hdr = ser.read(2)
    if len(hdr) < 2:
        return prev_q, None
    pt, addr = hdr[0], hdr[1]

    dlen = data_len_from_pt(pt)
    data = ser.read(dlen)
    cks = ser.read(2)
    if len(data) != dlen or len(cks) != 2:
        return prev_q, None

    pkt_wo_cks = b"snp" + bytes([pt, addr]) + data
    if u16_be(cks) != checksum(pkt_wo_cks):
        return prev_q, None

    if pt == 0xD4 and addr == 0x70 and dlen == 20:
        roll_raw = i16_be(data[0:2])
        pitch_raw = i16_be(data[2:4])
        yaw_raw = i16_be(data[4:6])
        roll_deg = roll_raw / 91.02222
        pitch_deg = pitch_raw / 91.02222
        yaw_deg = yaw_raw / 91.02222
        q = euler_xyz_rad_to_quaternion(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
        return q, (roll_deg, pitch_deg, yaw_deg)

    return prev_q, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Upper vs forearm motion separation diagnostic.")
    parser.add_argument("--port", default="COM6", help="Upper-arm serial IMU COM port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud")
    parser.add_argument("--upper-len", type=float, default=0.30, help="Upper arm length (m)")
    parser.add_argument("--fore-len", type=float, default=0.26, help="Forearm length (m)")
    parser.add_argument("--alpha", type=float, default=0.98, help="MindRove complementary filter alpha")
    parser.add_argument("--print-hz", type=float, default=10.0, help="Print rate (Hz)")
    parser.add_argument("--mindrove-ip", default="192.168.4.1", help="MindRove IP")
    parser.add_argument("--mindrove-port", type=int, default=4210, help="MindRove port")
    parser.add_argument("--mindrove-timeout-ms", type=int, default=8000, help="MindRove timeout (ms)")
    parser.add_argument("--use-synth", action="store_true", help="Synthetic MindRove board")
    parser.add_argument(
        "--csv",
        default="",
        help="Optional path to append CSV (time, upper rpy, fore rpy, |dup|, |dfore|)",
    )
    args = parser.parse_args()

    print(__doc__)
    print("Columns: t | upper RPY deg | fore RPY deg | |Δp_u| m | |Δp_f| m | hint")
    print("hint: which segment moved more since last line (magnitude of FK segment delta).\n")

    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if args.use_synth else BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    params.ip_address = args.mindrove_ip
    params.ip_port = args.mindrove_port
    params.timeout = args.mindrove_timeout_ms
    board = BoardShim(board_id, params)
    board.prepare_session()
    sample_rate = BoardShim.get_sampling_rate(board_id)
    acc_ch = BoardShim.get_accel_channels(board_id)
    gyro_ch = BoardShim.get_gyro_channels(board_id)

    fore_est = MindRoveOrientationEstimator(alpha=args.alpha)
    upper_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    last_upper_rpy: Optional[Tuple[float, float, float]] = None

    prev_p_u = np.zeros(3, dtype=np.float64)
    prev_p_f = np.zeros(3, dtype=np.float64)
    have_prev = False

    board.start_stream(256)
    csv_f = open(args.csv, "a", encoding="utf-8") if args.csv else None
    if csv_f:
        csv_f.write("t,ur,up,uy,fr,fp,fy,dpu,dfore\n")

    last_print = 0.0
    last_t = time.time()
    t0 = time.time()

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
            while True:
                # Drain serial for freshest upper pose
                for _ in range(32):
                    upper_q, euler_deg = _try_parse_upper_euler_serial(ser, upper_q)
                    if euler_deg is not None:
                        last_upper_rpy = euler_deg

                data = board.get_board_data()
                if data is None or data.shape[1] < 1:
                    time.sleep(0.001)
                    continue

                acc = data[acc_ch, -1].astype(np.float64)
                gyro = data[gyro_ch, -1].astype(np.float64)
                now = time.time()
                dt = max(1.0 / sample_rate, now - last_t)
                last_t = now
                fore_q = fore_est.update(acc, gyro, dt)

                p_u, p_f, _p_tot = wrist_position_decomposition(
                    upper_q, fore_q, args.upper_len, args.fore_len
                )

                if not have_prev:
                    prev_p_u, prev_p_f = p_u.copy(), p_f.copy()
                    have_prev = True

                du = float(np.linalg.norm(p_u - prev_p_u))
                df = float(np.linalg.norm(p_f - prev_p_f))
                prev_p_u, prev_p_f = p_u.copy(), p_f.copy()

                if now - last_print < 1.0 / max(0.5, args.print_hz):
                    continue
                last_print = now

                fr_deg = math.degrees(fore_est.roll)
                fp_deg = math.degrees(fore_est.pitch)
                fy_deg = math.degrees(fore_est.yaw)

                if last_upper_rpy is None:
                    ur, up, uy = float("nan"), float("nan"), float("nan")
                else:
                    ur, up, uy = last_upper_rpy

                eps = 1e-5
                if du > 2.0 * df + 1e-4 and du > eps:
                    hint = "upper>FK"
                elif df > 2.0 * du + 1e-4 and df > eps:
                    hint = "fore>FK"
                elif du < eps and df < eps:
                    hint = "quiet"
                else:
                    hint = "both"

                t = now - t0
                print(
                    f"{t:6.2f} | "
                    f"{ur:7.2f} {up:7.2f} {uy:7.2f} | "
                    f"{fr_deg:7.2f} {fp_deg:7.2f} {fy_deg:7.2f} | "
                    f"{du:8.5f} {df:8.5f} | {hint}"
                )
                if csv_f:
                    csv_f.write(f"{t},{ur},{up},{uy},{fr_deg},{fp_deg},{fy_deg},{du},{df}\n")
                    csv_f.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if csv_f:
            csv_f.close()
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
