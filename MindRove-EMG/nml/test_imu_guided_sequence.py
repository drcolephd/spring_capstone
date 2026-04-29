"""
Guided timed test: rest → forearm motions → elbow → upper arm.

Each phase shows instructions, a short countdown, then records for a fixed duration
and prints a summary (mean |Δp_upper|, mean |Δp_fore|, and RPY range).

Run from repo root:
  python -m nml.test_imu_guided_sequence --port COM6 --mindrove-ip 192.168.4.1

Press Ctrl+C to abort between phases.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import serial
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from numpy.typing import NDArray

from nml.processing import wrist_position_decomposition
from nml.test_dual_imu_live import MindRoveOrientationEstimator
from nml.test_imu_motion_separation import _try_parse_upper_euler_serial


@dataclass
class Phase:
    key: str
    title: str
    instruction: str
    expect: str  # short hint for interpreting summary


PHASES: List[Phase] = [
    Phase(
        "rest",
        "REST",
        "Sit comfortably. Keep upper arm and forearm still and relaxed for the whole window.",
        "Expect low |Δp_u| and |Δp_f| (quiet).",
    ),
    Phase(
        "forearm_right",
        "FOREARM — rotate one way",
        "Pin upper arm against your body or chair. Rotate the forearm (pronation/supination) in ONE direction only, smoothly, then hold.",
        "Expect forearm RPY range > upper; |Δp_f| often larger than |Δp_u|.",
    ),
    Phase(
        "forearm_left",
        "FOREARM — rotate the other way",
        "Same hold on upper arm. Rotate the forearm the OPPOSITE way, smoothly.",
        "Same as previous: forearm-dominated signals.",
    ),
    Phase(
        "elbow_flex",
        "ELBOW — bend and extend",
        "Keep shoulder relatively quiet. Bend elbow ~90° then straighten, repeat slowly.",
        "Both segments may move; upper RPY and forearm RPY often change together.",
    ),
    Phase(
        "upper_raise",
        "UPPER ARM — raise / lower",
        "Keep forearm as still as you can. Raise and lower the upper arm (shoulder flex) without twisting forearm.",
        "Expect upper RPY range and |Δp_u| to lead more than pure forearm phases.",
    ),
]


def _countdown(seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"  Starting in {i}…", flush=True)
        time.sleep(1.0)


def _run_phase(
    ser: serial.Serial,
    board: BoardShim,
    acc_ch: Tuple[int, ...],
    gyro_ch: Tuple[int, ...],
    sample_rate: float,
    fore_est: MindRoveOrientationEstimator,
    upper_q: NDArray[np.float64],
    last_upper_rpy: Optional[Tuple[float, float, float]],
    upper_len: float,
    fore_len: float,
    duration_s: float,
) -> Tuple[NDArray[np.float64], Optional[Tuple[float, float, float]], dict]:
    """Returns (new_upper_q, new_last_upper_rpy, stats)."""
    dus: List[float] = []
    dfs: List[float] = []
    upper_samples: List[Tuple[float, float, float]] = []
    fore_samples: List[Tuple[float, float, float]] = []

    prev_p_u = np.zeros(3, dtype=np.float64)
    prev_p_f = np.zeros(3, dtype=np.float64)
    have_prev = False
    last_t = time.time()
    t_end = time.time() + duration_s

    uq = upper_q.copy()
    lur = last_upper_rpy

    while time.time() < t_end:
        for _ in range(48):
            uq, euler_deg = _try_parse_upper_euler_serial(ser, uq)
            if euler_deg is not None:
                lur = euler_deg

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

        p_u, p_f, _ = wrist_position_decomposition(uq, fore_q, upper_len, fore_len)
        if not have_prev:
            prev_p_u, prev_p_f = p_u.copy(), p_f.copy()
            have_prev = True
            continue

        dus.append(float(np.linalg.norm(p_u - prev_p_u)))
        dfs.append(float(np.linalg.norm(p_f - prev_p_f)))
        prev_p_u, prev_p_f = p_u.copy(), p_f.copy()

        if lur is not None:
            upper_samples.append(lur)
        fr = math.degrees(fore_est.roll)
        fp = math.degrees(fore_est.pitch)
        fy = math.degrees(fore_est.yaw)
        fore_samples.append((fr, fp, fy))

    def _ranges(rows: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0
        arr = np.array(rows, dtype=np.float64)
        return float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1])), float(np.ptp(arr[:, 2]))

    ur, up, uy = _ranges(upper_samples)
    fr, fp, fy = _ranges(fore_samples)
    stats = {
        "n": len(dus),
        "mean_du": float(np.mean(dus)) if dus else 0.0,
        "mean_df": float(np.mean(dfs)) if dfs else 0.0,
        "std_du": float(np.std(dus)) if dus else 0.0,
        "std_df": float(np.std(dfs)) if dfs else 0.0,
        "upper_range_rpy": (ur, up, uy),
        "fore_range_rpy": (fr, fp, fy),
    }
    return uq, lur, stats


def _print_phase_summary(phase: Phase, stats: dict) -> None:
    ur, up, uy = stats["upper_range_rpy"]
    fr, fp, fy = stats["fore_range_rpy"]
    print(f"\n--- Summary: {phase.title} ({phase.key}) ---")
    print(f"  Samples (FK deltas): {stats['n']}")
    print(f"  mean |Δp_upper| = {stats['mean_du']:.6f} m   std = {stats['std_du']:.6f}")
    print(f"  mean |Δp_fore|  = {stats['mean_df']:.6f} m   std = {stats['std_df']:.6f}")
    print(f"  upper RPY range (max-min): roll={ur:.2f}° pitch={up:.2f}° yaw={uy:.2f}°")
    print(f"  fore  RPY range:           roll={fr:.2f}° pitch={fp:.2f}° yaw={fy:.2f}°")
    print(f"  Hint: {phase.expect}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided IMU motion sequence (rest, forearm, elbow, upper).")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--upper-len", type=float, default=0.30)
    parser.add_argument("--fore-len", type=float, default=0.26)
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--phase-seconds", type=float, default=12.0, help="Duration of each motion phase")
    parser.add_argument("--prep-seconds", type=int, default=3, help="Countdown before each phase")
    parser.add_argument("--mindrove-ip", default="192.168.4.1")
    parser.add_argument("--mindrove-port", type=int, default=4210)
    parser.add_argument("--mindrove-timeout-ms", type=int, default=8000)
    parser.add_argument("--use-synth", action="store_true")
    parser.add_argument("--skip", default="", help="Comma-separated phase keys to skip, e.g. upper_raise")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.SYNTHETIC_BOARD if args.use_synth else BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    params.ip_address = args.mindrove_ip
    params.ip_port = args.mindrove_port
    params.timeout = args.mindrove_timeout_ms
    board = BoardShim(board_id, params)
    board.prepare_session()
    sr = float(BoardShim.get_sampling_rate(board_id))
    acc_ch = BoardShim.get_accel_channels(board_id)
    gyro_ch = BoardShim.get_gyro_channels(board_id)

    fore_est = MindRoveOrientationEstimator(alpha=args.alpha)
    upper_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    last_upper_rpy: Optional[Tuple[float, float, float]] = None

    board.start_stream(256)
    print("\nGuided IMU sequence test\n")

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
            for phase in PHASES:
                if phase.key in skip:
                    print(f"Skipping phase: {phase.key}")
                    continue

                print("\n" + "=" * 72)
                print(f"  PHASE: {phase.title}")
                print("=" * 72)
                print(phase.instruction)
                print()
                _countdown(max(1, int(args.prep_seconds)))

                print(f"\n>>> RECORDING {args.phase_seconds:.0f} s — follow the instruction now.\n")
                upper_q, last_upper_rpy, stats = _run_phase(
                    ser,
                    board,
                    acc_ch,
                    gyro_ch,
                    sr,
                    fore_est,
                    upper_q,
                    last_upper_rpy,
                    args.upper_len,
                    args.fore_len,
                    args.phase_seconds,
                )
                _print_phase_summary(phase, stats)

                print("Pause. Get ready for the next phase (or Ctrl+C to stop).\n")
                time.sleep(2.0)

            print("All phases complete.\n")
    except KeyboardInterrupt:
        print("\nAborted by user.\n")
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
