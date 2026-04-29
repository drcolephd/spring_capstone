"""
Tune dual-IMU wrist-FK mapping with repeatable scoring.

Modes:
  - analyze: score existing CSV trajectory runs
  - sweep: record multiple runs over alpha/smoothing settings, then rank
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from nml.imu_trajectory_plot import (
    cmd_record,
    generate_reference_shape,
    resample_by_arclength,
    umeyama_similarity,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_csv_xyz(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if len(rows) < 10:
        raise ValueError(f"{path} has too few samples")
    return np.asarray(rows, dtype=np.float64)


def smooth_points(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points.copy()
    out = np.zeros_like(points)
    acc = np.zeros(3, dtype=np.float64)
    buf = [np.zeros(3, dtype=np.float64) for _ in range(window)]
    for i, p in enumerate(points):
        k = i % window
        acc -= buf[k]
        buf[k] = p
        acc += p
        n = min(i + 1, window)
        out[i] = acc / n
    return out


def score_trajectory(points: np.ndarray, shape: str, ref_points: int, ref_scale: float, rest_ratio: float) -> dict:
    p = np.asarray(points, dtype=np.float64)
    p = smooth_points(p, 3)
    p_rs = resample_by_arclength(p, ref_points)
    ref = generate_reference_shape(shape, n=max(600, ref_points * 2), scale=ref_scale)
    ref_rs = resample_by_arclength(ref, ref_points)
    s, r, t = umeyama_similarity(ref_rs, p_rs)
    ref_fit = s * (ref_rs @ r.T) + t

    rmse = float(np.sqrt(np.mean(np.sum((ref_fit - p_rs) ** 2, axis=1))))
    mae = float(np.mean(np.linalg.norm(ref_fit - p_rs, axis=1)))
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path_len = float(np.sum(d))
    n_rest = max(5, int(len(d) * max(0.01, min(0.5, rest_ratio))))
    rest_std = float(np.std(d[:n_rest])) if len(d) else 0.0
    if len(p) >= 3:
        d2 = p[2:] - 2 * p[1:-1] + p[:-2]
        jerk_rms = float(np.sqrt(np.mean(np.sum(d2 * d2, axis=1))))
    else:
        jerk_rms = 0.0

    score = rmse + 0.5 * mae + 2.0 * rest_std + 0.5 * jerk_rms
    return {
        "rmse": rmse,
        "mae": mae,
        "path_len": path_len,
        "rest_std": rest_std,
        "jerk_rms": jerk_rms,
        "score": score,
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    paths = sorted(Path().glob(args.csv_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.csv_glob}")
    rows = []
    for p in paths:
        pts = load_csv_xyz(p)
        m = score_trajectory(pts, args.shape, args.ref_points, args.ref_scale, args.rest_ratio)
        rows.append((p, m))
    rows.sort(key=lambda x: x[1]["score"])

    print(f"Scored {len(rows)} runs (lower is better):")
    print("rank  score     rmse      mae       rest_std  jerk_rms  path_len   file")
    for i, (p, m) in enumerate(rows, start=1):
        print(
            f"{i:>3}  {m['score']:8.5f} {m['rmse']:8.5f} {m['mae']:8.5f} "
            f"{m['rest_std']:8.5f} {m['jerk_rms']:8.5f} {m['path_len']:8.4f}  {p}"
        )


def cmd_sweep(args: argparse.Namespace) -> None:
    alphas = parse_float_list(args.alphas)
    smooths = parse_int_list(args.smooth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    total = len(alphas) * len(smooths)
    idx = 0
    for alpha in alphas:
        for smooth in smooths:
            idx += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            stem = f"run_a{alpha:.3f}_s{smooth}_{ts}".replace(".", "p")
            csv_path = out_dir / f"{stem}.csv"
            print(f"\n[{idx}/{total}] Recording alpha={alpha}, smooth={smooth} -> {csv_path}")
            print("Trace same motion/shape protocol each run for fair comparison.")

            rec_args = argparse.Namespace(
                seconds=args.seconds,
                out=str(csv_path),
                port=args.port,
                baud=args.baud,
                upper_len=args.upper_len,
                fore_len=args.fore_len,
                alpha=alpha,
                mindrove_ip=args.mindrove_ip,
                mindrove_port=args.mindrove_port,
                mindrove_timeout_ms=args.mindrove_timeout_ms,
                use_synth=args.use_synth,
            )
            cmd_record(rec_args)

            pts = load_csv_xyz(csv_path)
            if smooth > 1:
                pts = smooth_points(pts, smooth)
            m = score_trajectory(pts, args.shape, args.ref_points, args.ref_scale, args.rest_ratio)
            summary_rows.append((alpha, smooth, csv_path, m))
            print(
                f"score={m['score']:.5f} rmse={m['rmse']:.5f} "
                f"rest_std={m['rest_std']:.5f} path_len={m['path_len']:.4f}"
            )

    summary_rows.sort(key=lambda x: x[3]["score"])
    summary_csv = out_dir / "tuning_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "alpha", "smooth", "score", "rmse", "mae", "rest_std", "jerk_rms", "path_len", "file"])
        for rank, (alpha, smooth, path, m) in enumerate(summary_rows, start=1):
            w.writerow(
                [
                    rank,
                    alpha,
                    smooth,
                    m["score"],
                    m["rmse"],
                    m["mae"],
                    m["rest_std"],
                    m["jerk_rms"],
                    m["path_len"],
                    str(path),
                ]
            )

    print("\nTop runs:")
    print("rank  alpha  smooth  score     rmse      rest_std  file")
    for rank, (alpha, smooth, path, m) in enumerate(summary_rows[: min(10, len(summary_rows))], start=1):
        print(f"{rank:>3}  {alpha:>5.3f}  {smooth:>6}  {m['score']:8.5f} {m['rmse']:8.5f} {m['rest_std']:8.5f}  {path.name}")
    print(f"\nSaved ranking CSV: {summary_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Score existing CSV runs")
    pa.add_argument("--csv-glob", default="*.csv", help="Glob for trajectory CSV files")
    pa.add_argument("--shape", choices=("square", "triangle", "circle", "star"), default="square")
    pa.add_argument("--ref-points", type=int, default=300)
    pa.add_argument("--ref-scale", type=float, default=0.12)
    pa.add_argument("--rest-ratio", type=float, default=0.15, help="Fraction of initial trajectory treated as rest")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("sweep", help="Record and rank parameter sweep")
    ps.add_argument("--shape", choices=("square", "triangle", "circle", "star"), default="square")
    ps.add_argument("--seconds", type=float, default=20.0)
    ps.add_argument("--alphas", default="0.94,0.96,0.98")
    ps.add_argument("--smooth", default="1,5,10", help="Post-record smoothing windows for ranking")
    ps.add_argument("--out-dir", default="tuning_runs")
    ps.add_argument("--ref-points", type=int, default=300)
    ps.add_argument("--ref-scale", type=float, default=0.12)
    ps.add_argument("--rest-ratio", type=float, default=0.15)
    ps.add_argument("--port", default="COM6")
    ps.add_argument("--baud", type=int, default=115200)
    ps.add_argument("--upper-len", type=float, default=0.30)
    ps.add_argument("--fore-len", type=float, default=0.26)
    ps.add_argument("--mindrove-ip", default="192.168.4.1")
    ps.add_argument("--mindrove-port", type=int, default=4210)
    ps.add_argument("--mindrove-timeout-ms", type=int, default=8000)
    ps.add_argument("--use-synth", action="store_true")
    ps.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
