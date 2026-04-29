"""
Dual-IMU -> HEBI Cartesian teleop pipeline (paper-inspired incremental mapping).

Pipeline summary:
1) Upper IMU quaternion (serial) + forearm quaternion (MindRove) -> incremental arm kinematics
2) Paper-style pose update:
     p(t+1) = p(t) + gain_p * delta_p_avg
     Q(t+1) = deltaQ_scaled * Q(t)
3) Solve IK at each step and command HEBI joint positions

Typical run (with HRDF for your arm model):
  python -m nml.hebi_dual_imu_teleop --hrdf "C:/path/to/arm.hrdf" \\
    --family Arm --names J1_base,J2_shoulder,J3_elbow,J4_wrist1,J5_wrist2,J6_wrist3 \\
    --port COM6 --mindrove-ip 192.168.4.1
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from nml.imu_serial_reader import checksum, data_len_from_pt, i16_be, u16_be
from nml.processing.arm_incremental_kinematics import (
    ArmIncrementalKinematics,
    euler_xyz_rad_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)


def parse_csv_list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


class MindRoveOrientationEstimator:
    """Simple complementary filter for forearm orientation from MindRove acc+gyro."""

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def update(self, acc_xyz: np.ndarray, gyro_xyz: np.ndarray, dt: float) -> np.ndarray:
        gx, gy, gz = np.deg2rad(gyro_xyz)
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        ax, ay, az = acc_xyz
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        self.roll = self.alpha * self.roll + self.beta * roll_acc
        self.pitch = self.alpha * self.pitch + self.beta * pitch_acc
        return euler_xyz_rad_to_quaternion(self.roll, self.pitch, self.yaw)


def parse_upper_quaternion_from_serial(ser, previous_q: np.ndarray) -> np.ndarray:
    """Try parsing one SNP packet. If unsuccessful, return previous quaternion."""
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

    if pt == 0xD4 and addr == 0x70 and dlen == 20:
        roll_deg = i16_be(data[0:2]) / 91.02222
        pitch_deg = i16_be(data[2:4]) / 91.02222
        yaw_deg = i16_be(data[4:6]) / 91.02222
        return euler_xyz_rad_to_quaternion(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
    return previous_q


def rotation_matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to [w,x,y,z] quaternion."""
    m = np.asarray(r, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion(np.array([w, x, y, z], dtype=np.float64))


def scale_increment_quaternion(dq: np.ndarray, gain: float, max_angle_deg: float) -> np.ndarray:
    """
    Scale quaternion increment magnitude and clip max step angle.
    Returns [w,x,y,z].
    """
    q = normalize_quaternion(dq.astype(np.float64))
    if q[0] < 0:
        q = -q  # shortest-angle representative
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1e-12, 1.0 - w * w))
    axis = q[1:] / s
    scaled = angle * gain
    scaled = min(scaled, math.radians(max_angle_deg))
    return normalize_quaternion(
        np.array(
            [math.cos(0.5 * scaled), *(axis * math.sin(0.5 * scaled))],
            dtype=np.float64,
        )
    )


def build_fallback_two_dof_model(hebi_module):
    model = hebi_module.robot_model.RobotModel()
    model.add_actuator("X5-1")
    model.add_bracket("X5-LightBracket", "right")
    model.add_link("X5", 0.5, 0)
    model.add_actuator("X5-1")
    model.add_link("X5", 0.5, 0)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="HEBI teleop from dual-IMU incremental kinematics.")
    parser.add_argument("--family", default="Arm", help="HEBI family")
    parser.add_argument(
        "--names",
        default="J1_base,J2_shoulder,J3_elbow,J4_wrist1,J5_wrist2,J6_wrist3",
        help="Comma-separated HEBI module names",
    )
    parser.add_argument("--lookup-wait", type=float, default=2.0, help="Seconds to wait for HEBI discovery")
    parser.add_argument("--hrdf", default="", help="Path to arm HRDF (recommended for >=3 DoF)")

    parser.add_argument("--port", default="COM6", help="Upper-arm serial IMU COM port")
    parser.add_argument("--baud", type=int, default=115200, help="Upper-arm serial IMU baud")
    parser.add_argument("--mindrove-ip", default="192.168.4.1", help="MindRove board IP")
    parser.add_argument("--mindrove-port", type=int, default=4210, help="MindRove board port")
    parser.add_argument("--mindrove-timeout-ms", type=int, default=8000, help="MindRove timeout in ms")
    parser.add_argument("--use-synth", action="store_true", help="Use synthetic MindRove board")

    parser.add_argument("--upper-len", type=float, default=0.30, help="Upper arm length (m)")
    parser.add_argument("--fore-len", type=float, default=0.26, help="Forearm length (m)")
    parser.add_argument("--alpha", type=float, default=0.98, help="Forearm complementary-filter alpha")
    parser.add_argument("--smooth-window", type=int, default=10, help="Smoothing window for delta kinematics")

    parser.add_argument("--pos-gain", type=float, default=1.0, help="Gain on delta_p_avg")
    parser.add_argument("--rot-gain", type=float, default=1.0, help="Gain on delta_q_avg step angle")
    parser.add_argument("--max-pos-step", type=float, default=0.02, help="Max position step per update (m)")
    parser.add_argument("--max-rot-step-deg", type=float, default=8.0, help="Max rotation step per update (deg)")
    parser.add_argument("--rate-hz", type=float, default=40.0, help="Control loop rate")
    args = parser.parse_args()

    import hebi
    import serial
    from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams

    names = parse_csv_list(args.names)
    family = [args.family] * len(names)
    if len(names) < 1:
        raise RuntimeError("At least one module name is required in --names")

    # --- HEBI discovery / group ---
    lookup = hebi.Lookup()
    time.sleep(max(0.5, args.lookup_wait))
    group = lookup.get_group_from_names(family, names)
    if group is None:
        entries = getattr(lookup, "entrylist", None)
        discovered = []
        if entries:
            for e in entries:
                discovered.append(f"{getattr(e,'family','?')}/{getattr(e,'name','?')}")
        raise RuntimeError(
            f"Could not create HEBI group for {args.family}/{names}. "
            f"Discovered: {', '.join(discovered) if discovered else 'none'}"
        )
    group.feedback_frequency = float(args.rate_hz)

    # --- Robot model (HRDF strongly recommended for real arm) ---
    if args.hrdf:
        model = hebi.robot_model.import_from_hrdf(args.hrdf)
    else:
        if len(names) > 2:
            raise RuntimeError(
                "Provide --hrdf for >2 DoF arm. Fallback model only supports demo-style 2 DoF."
            )
        model = build_fallback_two_dof_model(hebi)

    # --- MindRove setup ---
    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if args.use_synth else BoardIds.MINDROVE_WIFI_BOARD
    input_params = MindRoveInputParams()
    input_params.ip_address = args.mindrove_ip
    input_params.ip_port = args.mindrove_port
    input_params.timeout = args.mindrove_timeout_ms
    board = BoardShim(board_id, input_params)
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

    print("Starting MindRove stream...")
    board.start_stream(256)

    feedback = group.get_next_feedback(reuse_fbk=None, timeout_ms=1000)
    if feedback is None:
        raise RuntimeError("Failed to read initial HEBI feedback.")
    q_cmd = feedback.position.copy()

    ee = model.get_end_effector(q_cmd)
    target_pos = ee[:3, 3].astype(np.float64)
    target_q = rotation_matrix_to_quaternion(ee[:3, :3])
    target_rot = quaternion_to_rotation_matrix(target_q)

    cmd = hebi.GroupCommand(group.size)
    dt_target = 1.0 / max(5.0, args.rate_hz)
    last_t = time.time()

    print("Teleop loop running. Ctrl+C to stop.")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
            while True:
                start = time.time()
                for _ in range(24):
                    upper_q = parse_upper_quaternion_from_serial(ser, upper_q)

                data = board.get_board_data()
                if data is None or data.shape[1] < 1:
                    time.sleep(0.001)
                    continue

                acc = data[accel_channels, -1].astype(np.float64)
                gyro = data[gyro_channels, -1].astype(np.float64)
                now = time.time()
                dt_imu = max(1.0 / sample_rate, now - last_t)
                last_t = now
                fore_q = fore_est.update(acc, gyro, dt_imu)

                _dp, _dq, dp_avg, dq_avg = kin.update(upper_q, fore_q)
                step = np.asarray(dp_avg, dtype=np.float64) * float(args.pos_gain)
                step_norm = float(np.linalg.norm(step))
                if step_norm > args.max_pos_step > 0:
                    step = step * (args.max_pos_step / step_norm)
                target_pos = target_pos + step

                dq_scaled = scale_increment_quaternion(
                    np.asarray(dq_avg, dtype=np.float64),
                    gain=float(args.rot_gain),
                    max_angle_deg=float(args.max_rot_step_deg),
                )
                target_q = normalize_quaternion(quaternion_multiply(dq_scaled, target_q))
                target_rot = quaternion_to_rotation_matrix(target_q)

                objectives: list = [hebi.robot_model.endeffector_position_objective(target_pos)]
                if model.dof_count >= 3:
                    objectives.append(hebi.robot_model.endeffector_so3_objective(target_rot))
                q_cmd = model.solve_inverse_kinematics(q_cmd, *objectives, max_iterations=50)

                cmd.position = q_cmd
                group.send_command(cmd)

                elapsed = time.time() - start
                if elapsed < dt_target:
                    time.sleep(dt_target - elapsed)
    except KeyboardInterrupt:
        print("\nStopping teleop.")
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
