"""
Live 3D hand-point simulation for dual-IMU incremental mapping.

This visualizes a moving dot in 3D space (plus short trail) driven by:
- Upper arm IMU from serial (external sensor)
- Forearm IMU from MindRove

The point is integrated from smoothed incremental wrist motion (delta_p_avg).

Run:
  python -m nml.live_3d_hand_sim --port COM6 --mindrove-ip 192.168.4.1

Controls:
  r  -> recenter (reset point to origin)
  q  -> close window and exit
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np


def main() -> None:
    import matplotlib.pyplot as plt
    import serial
    from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams

    from nml.processing import ArmIncrementalKinematics
    from nml.test_dual_imu_live import MindRoveOrientationEstimator, parse_upper_quaternion_from_serial

    parser = argparse.ArgumentParser(description="Live 3D hand-point simulation from dual IMUs.")
    parser.add_argument("--port", default="COM6", help="Upper-arm serial IMU COM port")
    parser.add_argument("--baud", type=int, default=115200, help="Upper-arm serial IMU baud")
    parser.add_argument("--mindrove-ip", default="192.168.4.1", help="MindRove board IP")
    parser.add_argument("--mindrove-port", type=int, default=4210, help="MindRove board port")
    parser.add_argument("--mindrove-timeout-ms", type=int, default=8000, help="MindRove timeout in milliseconds")
    parser.add_argument("--use-synth", action="store_true", help="Use synthetic MindRove board")
    parser.add_argument("--upper-len", type=float, default=0.30, help="Upper arm length (m)")
    parser.add_argument("--fore-len", type=float, default=0.26, help="Forearm length (m)")
    parser.add_argument("--alpha", type=float, default=0.98, help="Forearm complementary-filter alpha")
    parser.add_argument("--smooth-window", type=int, default=10, help="Increment smoothing window")
    parser.add_argument("--gain", type=float, default=1.0, help="Scale factor on delta_p_avg")
    parser.add_argument("--trail", type=int, default=200, help="Number of points in trajectory trail")
    parser.add_argument("--update-hz", type=float, default=30.0, help="UI update rate")
    parser.add_argument("--axis-range", type=float, default=0.5, help="Half-range for X/Y/Z axis in meters")
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if args.use_synth else BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    params.ip_address = args.mindrove_ip
    params.ip_port = args.mindrove_port
    params.timeout = args.mindrove_timeout_ms
    board = BoardShim(board_id, params)
    board.prepare_session()

    sample_rate = float(BoardShim.get_sampling_rate(board_id))
    accel_channels = BoardShim.get_accel_channels(board_id)
    gyro_channels = BoardShim.get_gyro_channels(board_id)

    kin = ArmIncrementalKinematics(
        upper_arm_length_m=args.upper_len,
        forearm_length_m=args.fore_len,
        smooth_window=args.smooth_window,
    )
    fore_est = MindRoveOrientationEstimator(alpha=args.alpha)
    upper_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    pos = np.zeros(3, dtype=np.float64)
    trail: deque[np.ndarray] = deque(maxlen=max(10, args.trail))
    trail.append(pos.copy())
    running = True

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Live 3D Hand Point (Dual-IMU)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    r = float(args.axis_range)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.grid(True)

    # Dot, trail, and origin marker.
    dot = ax.scatter([pos[0]], [pos[1]], [pos[2]], s=80, c="red", label="hand point")
    trail_line, = ax.plot([pos[0]], [pos[1]], [pos[2]], color="blue", linewidth=1.2, alpha=0.8, label="trail")
    ax.scatter([0.0], [0.0], [0.0], s=30, c="black", label="origin")
    ax.legend(loc="upper left")

    def on_key(event) -> None:
        nonlocal running, pos, trail
        if event.key == "r":
            pos = np.zeros(3, dtype=np.float64)
            trail.clear()
            trail.append(pos.copy())
            print("Recentered hand point.")
        elif event.key == "q":
            running = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Starting MindRove stream...")
    board.start_stream(256)
    last_t = time.time()
    ui_dt = 1.0 / max(1.0, args.update_hz)
    next_ui = time.time()

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
            print("Live 3D simulation running. Press 'r' to recenter, 'q' to quit.")
            while running and plt.fignum_exists(fig.number):
                for _ in range(24):
                    upper_q = parse_upper_quaternion_from_serial(ser, upper_q)

                data = board.get_board_data()
                if data is None or data.shape[1] < 1:
                    plt.pause(0.001)
                    continue

                acc = data[accel_channels, -1].astype(np.float64)
                gyro = data[gyro_channels, -1].astype(np.float64)
                now = time.time()
                dt = max(1.0 / sample_rate, now - last_t)
                last_t = now

                fore_q = fore_est.update(acc, gyro, dt)
                _dp, _dq, dp_avg, _dq_avg = kin.update(upper_q, fore_q)
                pos = pos + float(args.gain) * dp_avg
                trail.append(pos.copy())

                if now >= next_ui:
                    next_ui = now + ui_dt
                    arr = np.array(trail, dtype=np.float64)
                    dot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])  # type: ignore[attr-defined]
                    trail_line.set_data(arr[:, 0], arr[:, 1])
                    trail_line.set_3d_properties(arr[:, 2])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
    except KeyboardInterrupt:
        pass
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
        print("Stopped live 3D simulation.")


if __name__ == "__main__":
    main()
