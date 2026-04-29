import argparse
import math
import time

from nml.processing import ArmIncrementalKinematics, euler_xyz_rad_to_quaternion


def run_simulation(upper_len: float, fore_len: float, smooth_window: int, hz: float, seconds: float) -> None:
    kin = ArmIncrementalKinematics(
        upper_arm_length_m=upper_len,
        forearm_length_m=fore_len,
        smooth_window=smooth_window,
    )

    dt = 1.0 / hz
    steps = max(1, int(seconds * hz))

    print(f"Running simulation for {seconds:.1f}s at {hz:.1f} Hz ({steps} steps)")
    print("Columns: step, dP(x y z), dPavg(x y z), dQavg(w x y z)")

    for i in range(steps):
        t = i * dt

        # Smooth synthetic arm motion in radians
        upper_roll = 0.10 * math.sin(2.0 * math.pi * 0.20 * t)
        upper_pitch = 0.20 * math.sin(2.0 * math.pi * 0.15 * t + 0.3)
        upper_yaw = 0.10 * math.sin(2.0 * math.pi * 0.10 * t + 0.7)

        fore_roll = 0.15 * math.sin(2.0 * math.pi * 0.30 * t + 0.1)
        fore_pitch = 0.25 * math.sin(2.0 * math.pi * 0.18 * t + 0.4)
        fore_yaw = 0.12 * math.sin(2.0 * math.pi * 0.12 * t + 1.0)

        q_upper = euler_xyz_rad_to_quaternion(upper_roll, upper_pitch, upper_yaw)
        q_fore = euler_xyz_rad_to_quaternion(fore_roll, fore_pitch, fore_yaw)

        delta_p, _delta_q, delta_p_avg, delta_q_avg = kin.update(q_upper, q_fore)
        print(
            f"{i:04d}  "
            f"dP=({delta_p[0]:+0.5f} {delta_p[1]:+0.5f} {delta_p[2]:+0.5f})  "
            f"dPavg=({delta_p_avg[0]:+0.5f} {delta_p_avg[1]:+0.5f} {delta_p_avg[2]:+0.5f})  "
            f"dQavg=({delta_q_avg[0]:+0.5f} {delta_q_avg[1]:+0.5f} {delta_q_avg[2]:+0.5f} {delta_q_avg[3]:+0.5f})"
        )

        time.sleep(dt)


def run_manual(upper_len: float, fore_len: float, smooth_window: int) -> None:
    kin = ArmIncrementalKinematics(
        upper_arm_length_m=upper_len,
        forearm_length_m=fore_len,
        smooth_window=smooth_window,
    )

    print("Manual mode.")
    print("Enter six Euler angles in degrees each line:")
    print("upper_roll upper_pitch upper_yaw fore_roll fore_pitch fore_yaw")
    print("Example: 0 5 0 3 10 0")
    print("Type 'q' to quit.")

    while True:
        raw = input("> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        parts = raw.split()
        if len(parts) != 6:
            print("Expected 6 values.")
            continue
        try:
            vals_deg = [float(v) for v in parts]
        except ValueError:
            print("Please enter numeric values.")
            continue

        vals_rad = [math.radians(v) for v in vals_deg]
        q_upper = euler_xyz_rad_to_quaternion(vals_rad[0], vals_rad[1], vals_rad[2])
        q_fore = euler_xyz_rad_to_quaternion(vals_rad[3], vals_rad[4], vals_rad[5])

        delta_p, delta_q, delta_p_avg, delta_q_avg = kin.update(q_upper, q_fore)
        print(f"dP      = {delta_p}")
        print(f"dQ      = {delta_q}")
        print(f"dP_avg  = {delta_p_avg}")
        print(f"dQ_avg  = {delta_q_avg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test runner for ArmIncrementalKinematics.")
    parser.add_argument("--mode", choices=["sim", "manual"], default="sim", help="sim: synthetic motion; manual: type Euler angles")
    parser.add_argument("--upper-len", type=float, default=0.30, help="Upper arm length in meters")
    parser.add_argument("--fore-len", type=float, default=0.26, help="Forearm length in meters")
    parser.add_argument("--smooth-window", type=int, default=10, help="Moving average window length")
    parser.add_argument("--hz", type=float, default=20.0, help="Simulation sample rate")
    parser.add_argument("--seconds", type=float, default=10.0, help="Simulation duration")
    args = parser.parse_args()

    if args.mode == "manual":
        run_manual(args.upper_len, args.fore_len, args.smooth_window)
    else:
        run_simulation(args.upper_len, args.fore_len, args.smooth_window, args.hz, args.seconds)


if __name__ == "__main__":
    main()
