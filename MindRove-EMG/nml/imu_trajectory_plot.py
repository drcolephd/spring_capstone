"""
Record 3D wrist position from dual IMU FK, then plot vs a reference shape (paper-style).

Similar to the paper's motion-mapping figures: dashed blue = reference / target path,
solid red = measured wrist trajectory (here: FK from upper serial IMU + MindRove forearm).

Usage - record (shoulder-frame wrist position in meters, ~MindRove rate):
  python -m nml.imu_trajectory_plot record --seconds 45 --out wrist_run1.csv \\
    --port COM6 --mindrove-ip 192.168.4.1

Usage - plot (aligns reference shape to recorded path with similarity transform):
  python -m nml.imu_trajectory_plot plot --csv wrist_run1.csv --shape square --save fig_square.png

Shapes: square, triangle, circle, star
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from typing import Tuple

import numpy as np

# Optional heavy imports only when needed
def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, F811
    except ImportError as e:
        raise SystemExit("matplotlib is required for plotting. pip install matplotlib") from e


def generate_reference_shape(shape: str, n: int = 400, scale: float = 0.15) -> np.ndarray:
    """
    Closed reference path in 3D (mostly XY plane, z small).
    `scale` ~ characteristic size in meters for the template before alignment.
    """
    shape = shape.lower().strip()
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)

    if shape == "circle":
        x = scale * np.cos(t)
        y = scale * np.sin(t)
        z = 0.02 * scale * np.sin(2 * t)
        return np.column_stack([x, y, z])

    if shape == "square":
        # 4 edges, perimeter param u in [0,4)
        u = np.linspace(0.0, 4.0, n, endpoint=False)
        s = scale * 0.9
        x = np.empty(n)
        y = np.empty(n)
        for i, ui in enumerate(u):
            f = ui % 1.0
            seg = int(ui) % 4
            if seg == 0:
                x[i], y[i] = -s + 2 * s * f, -s
            elif seg == 1:
                x[i], y[i] = s, -s + 2 * s * f
            elif seg == 2:
                x[i], y[i] = s - 2 * s * f, s
            else:
                x[i], y[i] = -s, s - 2 * s * f
        z = np.zeros(n)
        return np.column_stack([x, y, z])

    if shape == "triangle":
        ang = np.array([0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=float)
        vx = scale * np.cos(ang)
        vy = scale * np.sin(ang)
        pts = np.column_stack([vx, vy, np.zeros(3)])
        # repeat perimeter
        out = np.zeros((n, 3))
        for i in range(n):
            a = (i / n) * 3.0
            k = int(a) % 3
            f = a - int(a)
            p0, p1 = pts[k], pts[(k + 1) % 3]
            out[i] = (1 - f) * p0 + f * p1
        return out

    if shape == "star":
        R = scale
        r = 0.38 * scale
        pts = []
        for k in range(10):
            rad = R if k % 2 == 0 else r
            theta = math.pi / 2 + k * math.pi / 5
            pts.append([rad * math.cos(theta), rad * math.sin(theta), 0.0])
        pts = np.array(pts, dtype=float)
        out = np.zeros((n, 3))
        for i in range(n):
            a = (i / n) * 10.0
            k = int(a) % 10
            f = a - int(a)
            p0, p1 = pts[k], pts[(k + 1) % 10]
            out[i] = (1 - f) * p0 + f * p1
        return out

    raise ValueError(f"Unknown shape: {shape}. Use square|triangle|circle|star")


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Least-squares similarity transform: dst ≈ s * src @ R.T + t
    src, dst: (m, 3)
    Returns (s, R, t) with R orthonormal det +1.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    m = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / m
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / m
    c = float(np.sum(D * np.diag(S)))
    s = c / var_s if var_s > 1e-12 else 1.0
    t = mu_d - s * (mu_s @ R.T)
    return s, R, t


def resample_by_arclength(points: np.ndarray, n_out: int) -> np.ndarray:
    """Resample polyline to n_out points uniformly in arclength (closed if first≈last)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    total = d[-1]
    if total < 1e-9:
        return np.repeat(p[:1], n_out, axis=0)
    targets = np.linspace(0.0, total, n_out, endpoint=False)
    out = np.zeros((n_out, 3))
    j = 0
    for i, tg in enumerate(targets):
        while j + 1 < len(d) and d[j + 1] < tg:
            j += 1
        j = min(j, len(p) - 2)
        a = (tg - d[j]) / max(d[j + 1] - d[j], 1e-12)
        out[i] = (1 - a) * p[j] + a * p[j + 1]
    return out


def cmd_record(args: argparse.Namespace) -> None:
    import serial
    from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams

    from nml.processing import wrist_position_decomposition
    from nml.test_dual_imu_live import MindRoveOrientationEstimator, parse_upper_quaternion_from_serial

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

    fore = MindRoveOrientationEstimator(alpha=args.alpha)
    upper_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    board.start_stream(256)

    t0 = time.time()
    rows = []
    last_t = time.time()
    print(f"Recording wrist (FK) for {args.seconds:.1f} s → {args.out}")
    print("Trace your shape slowly; Ctrl+C stops early.\n")

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as ser:
            while time.time() - t0 < args.seconds:
                for _ in range(32):
                    upper_q = parse_upper_quaternion_from_serial(ser, upper_q)

                data = board.get_board_data()
                if data is None or data.shape[1] < 1:
                    time.sleep(0.001)
                    continue

                acc = data[acc_ch, -1].astype(np.float64)
                gyro = data[gyro_ch, -1].astype(np.float64)
                now = time.time()
                dt = max(1.0 / sr, now - last_t)
                last_t = now
                fore_q = fore.update(acc, gyro, dt)
                _pu, _pf, p_tot = wrist_position_decomposition(
                    upper_q, fore_q, args.upper_len, args.fore_len
                )
                rows.append((now - t0, float(p_tot[0]), float(p_tot[1]), float(p_tot[2])))
    except KeyboardInterrupt:
        print("Stopped early.")
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

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} samples to {args.out}")


def cmd_plot(args: argparse.Namespace) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    pts = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if len(pts) < 10:
        print("Not enough points in CSV.", file=sys.stderr)
        sys.exit(1)
    actual = np.array(pts, dtype=np.float64)

    ref_hi = generate_reference_shape(args.shape, n=max(600, args.ref_points * 2), scale=args.ref_scale)
    act_rs = resample_by_arclength(actual, args.ref_points)
    ref_rs = resample_by_arclength(ref_hi, args.ref_points)
    s, R, t = umeyama_similarity(ref_rs, act_rs)
    ref_aligned = s * (ref_hi @ R.T) + t

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        actual[:, 0],
        actual[:, 1],
        actual[:, 2],
        color="red",
        linewidth=1.2,
        label="Actual (measured)",
    )
    ax.plot(
        ref_aligned[:, 0],
        ref_aligned[:, 1],
        ref_aligned[:, 2],
        color="blue",
        linestyle="--",
        linewidth=1.0,
        label=f"Reference ({args.shape})",
    )
    ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], color="k", s=40, label="start")
    ax.scatter(actual[-1, 0], actual[-1, 1], actual[-1, 2], color="green", s=40, label="end")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Wrist FK trajectory vs aligned reference (shoulder frame)")
    ax.legend(loc="upper right", fontsize=8)
    try:
        plt.tight_layout()
    except Exception:
        pass
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Record wrist position CSV")
    pr.add_argument("--seconds", type=float, default=30.0)
    pr.add_argument("--out", default="wrist_trajectory.csv")
    pr.add_argument("--port", default="COM6")
    pr.add_argument("--baud", type=int, default=115200)
    pr.add_argument("--upper-len", type=float, default=0.30)
    pr.add_argument("--fore-len", type=float, default=0.26)
    pr.add_argument("--alpha", type=float, default=0.98)
    pr.add_argument("--mindrove-ip", default="192.168.4.1")
    pr.add_argument("--mindrove-port", type=int, default=4210)
    pr.add_argument("--mindrove-timeout-ms", type=int, default=8000)
    pr.add_argument("--use-synth", action="store_true")
    pr.set_defaults(func=cmd_record)

    pp = sub.add_parser("plot", help="Plot recorded trajectory vs reference shape")
    pp.add_argument("--csv", required=True)
    pp.add_argument("--shape", choices=("square", "triangle", "circle", "star"), default="square")
    pp.add_argument("--ref-scale", type=float, default=0.12, dest="ref_scale", help="Template size before alignment (m)")
    pp.add_argument("--ref-points", type=int, default=300)
    pp.add_argument("--save", default="", help="PNG path (if empty, opens window)")
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
