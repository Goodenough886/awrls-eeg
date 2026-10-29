"""One-at-a-time parameter sensitivity around the proposed configuration.

python scripts/run_sensitivity.py --mixed data/mixed.npy --clean data/clean.npy --out results/sensitivity --jobs 8

Use --split dev for parameter selection and report the frozen choice on --split holdout.
Outputs: sensitivity_by_epoch.csv, sensitivity_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PRIMARY_METRICS, Config, deterministic_split, environment, load_pairs, run_variant, write_json
from awrls2 import sensitivity_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", choices=["all", "dev", "holdout"], default="dev")
    ap.add_argument("--params", nargs="*", default=None, help="subset of parameters (default: all in sensitivity_grid)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    mixed, clean = load_pairs(args.mixed, args.clean, args.limit)
    dev, hold = deterministic_split(mixed.shape[0])
    indices = {"all": np.arange(mixed.shape[0]), "dev": dev, "holdout": hold}[args.split]
    grid = sensitivity_grid()
    if args.params:
        grid = {k: grid[k] for k in args.params}
    write_json(out / "protocol.json", {"environment": environment(), "split": args.split, "grid": grid,
                                       "base": Config(fs=args.fs).__dict__})
    frames = []
    for param, values in grid.items():
        for value in values:
            name = f"{param}={value}"
            df = run_variant(name, {param: value}, mixed, clean, args.fs, indices, args.jobs)
            df.insert(1, "parameter", param)
            df.insert(2, "value", value)
            frames.append(df)
            print(f"{name:36s} MAE={df.mae.mean():.3f} RMSE={df.rmse.mean():.3f} CC={df.cc.mean():.3f} "
                  f"F1={df.f1.mean():.3f} DBP={df.dbp_total.mean():.2f} ms={df.elapsed_ms.mean():.1f}")
            pd.concat(frames).to_csv(out / "sensitivity_by_epoch.csv", index=False)
    by_epoch = pd.concat(frames, ignore_index=True)
    metrics = [m for m in PRIMARY_METRICS if m in by_epoch.columns]
    g = by_epoch.groupby(["parameter", "value"], sort=False)[metrics]
    summary = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1)}, axis=1)
    summary.columns = [f"{m}_{s}" for s, m in summary.columns]
    summary.to_csv(out / "sensitivity_summary.csv")
    print("done ->", out)


if __name__ == "__main__":
    main()
