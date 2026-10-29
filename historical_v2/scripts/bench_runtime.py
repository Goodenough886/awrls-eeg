"""Runtime benchmark (single thread) for the manuscript's Table 5 replacement.

python scripts/bench_runtime.py --mixed data/mixed.npy --clean data/clean.npy --out results/runtime
Measures, for V1_reproduction, PROPOSED and the compact-candidate variant:
  * per-epoch time on 10 s epochs (warm-up 1, timed 3, mean of per-epoch means)
  * time for 30 s and 300 s concatenated records with block processing
  * cost per processed event
Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 before running.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from common import Config, environment, load_pairs, write_json  # noqa: E402
from awrls2 import ablation_variants, clean, process_epoch  # noqa: E402


def time_call(fn, repeats=3):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts, ddof=1)) if repeats > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--variants", nargs="*", default=["V1_reproduction", "PROPOSED", "C6_compact_candidates", "D1_online_rls_one_pass"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mixed, _ = load_pairs(args.mixed, args.clean, args.n_epochs)
    variants = ablation_variants()
    rows = []
    for name in args.variants:
        cfg = Config(fs=args.fs).with_updates(**variants[name])
        per_epoch, events = [], 0
        for i in range(mixed.shape[0]):
            m, s = time_call(lambda: process_epoch(mixed[i], cfg))
            per_epoch.append(m)
            events += process_epoch(mixed[i], cfg)[1]["processed_count"]
        rows.append({"variant": name, "signal": f"{mixed.shape[1] / args.fs:.0f}s epoch", "n": mixed.shape[0],
                     "mean_ms": 1000 * np.mean(per_epoch), "sd_ms": 1000 * np.std(per_epoch, ddof=1),
                     "ms_per_event": 1000 * np.sum(per_epoch) / max(events, 1)})
        for seconds in (30, 300):
            reps = int(np.ceil(seconds * args.fs / mixed.shape[1]))
            long = np.concatenate([mixed[i % mixed.shape[0]] for i in range(reps)])[: int(seconds * args.fs)]
            m, s = time_call(lambda: clean(long, args.fs, cfg, block_s=30, overlap_s=4), repeats=2)
            n_ev = clean(long, args.fs, cfg, block_s=30, overlap_s=4)[1]["processed_count"]
            rows.append({"variant": name, "signal": f"{seconds}s record (30 s blocks)", "n": 1, "mean_ms": 1000 * m,
                         "sd_ms": 1000 * s, "ms_per_event": 1000 * m / max(n_ev, 1)})
        print(pd.DataFrame(rows).tail(3).round(2).to_string(index=False))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "runtime.csv", index=False)
    write_json(args.out / "runtime_environment.json", environment())
    print("done ->", args.out)


if __name__ == "__main__":
    main()
