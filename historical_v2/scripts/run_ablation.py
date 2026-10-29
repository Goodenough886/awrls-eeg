"""Ablation study on paired semi-simulated epochs.

Example
-------
python scripts/run_ablation.py --mixed data/mixed.npy --clean data/clean.npy --fs 200 \
    --out results/ablation --split all --jobs 8 --snr-levels -7 -5 -3 -1 1 3

Outputs (in --out)
------------------
protocol.json            environment, data hashes, variant definitions, split
by_epoch.csv             every metric for every epoch x variant (native SNR)
summary.csv              mean / sd / median / IQR per variant
stats_vs_V1.csv          paired Wilcoxon (+Holm), bootstrap CI, effect sizes vs V1_reproduction
stats_vs_PROPOSED.csv    the same, relative to PROPOSED (what each removal costs)
snr_sweep_by_epoch.csv   V1 vs PROPOSED (and --snr-variants) at the requested SNR levels
clean_control.csv        behaviour on artifact-free epochs (false corrections)
wavelet_selection.csv    histogram of selected mother wavelet x level (PROPOSED)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HIGHER_IS_BETTER, PRIMARY_METRICS, Config, V1_REPRODUCTION, deterministic_split, environment,
                    evaluate, load_pairs, paired_stats, process_epoch, run_variant, sha256, summarise, write_json)
from awrls2 import ablation_variants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixed", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names (default: all)")
    ap.add_argument("--split", choices=["all", "dev", "holdout"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N epochs (smoke test)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--snr-levels", nargs="*", type=float, default=[-7, -5, -3, -1, 1, 3])
    ap.add_argument("--snr-variants", nargs="*", default=["V1_reproduction", "PROPOSED"])
    ap.add_argument("--skip-snr", action="store_true")
    ap.add_argument("--skip-clean-control", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    mixed, clean = load_pairs(args.mixed, args.clean, args.limit)
    n = mixed.shape[0]
    dev, hold = deterministic_split(n)
    indices = {"all": np.arange(n), "dev": dev, "holdout": hold}[args.split]
    variants = ablation_variants()
    if args.variants:
        unknown = set(args.variants) - set(variants)
        if unknown:
            raise SystemExit(f"Unknown variants: {sorted(unknown)}")
        variants = {k: variants[k] for k in args.variants}

    write_json(out / "protocol.json", {
        "environment": environment(), "mixed": str(args.mixed), "clean": str(args.clean),
        "mixed_sha256": sha256(args.mixed), "clean_sha256": sha256(args.clean), "fs": args.fs,
        "n_epochs_total": int(n), "split": args.split, "epochs": indices.tolist(),
        "variants": variants, "proposed_defaults": Config(fs=args.fs).__dict__,
        "input_snr_db_mean": float(np.mean(evaluate.input_snr_db(clean[indices], mixed[indices]))),
        "snr_levels": args.snr_levels,
    })

    frames = []
    for name, overrides in variants.items():
        t0 = time.time()
        df = run_variant(name, overrides, mixed, clean, args.fs, indices, args.jobs)
        frames.append(df)
        print(f"{name:40s} n={len(df):4d}  MAE={df.mae.mean():.3f} RMSE={df.rmse.mean():.3f} "
              f"CC={df.cc.mean():.3f} F1={df.f1.mean():.3f} DBP={df.dbp_total.mean():.2f}  ({time.time() - t0:.1f}s)")
        pd.concat(frames).to_csv(out / "by_epoch.csv", index=False)  # checkpoint
    by_epoch = pd.concat(frames, ignore_index=True)
    by_epoch.to_csv(out / "by_epoch.csv", index=False)
    metrics = [m for m in PRIMARY_METRICS if m in by_epoch.columns]
    summarise(by_epoch, metrics).to_csv(out / "summary.csv")
    if "V1_reproduction" in variants:
        paired_stats(by_epoch, "V1_reproduction", metrics, HIGHER_IS_BETTER).to_csv(out / "stats_vs_V1.csv", index=False)
    if "PROPOSED" in variants:
        paired_stats(by_epoch, "PROPOSED", metrics, HIGHER_IS_BETTER).to_csv(out / "stats_vs_PROPOSED.csv", index=False)
        sel = Counter()
        for w in by_epoch[by_epoch.variant == "PROPOSED"].wavelets.fillna(""):
            sel.update([t for t in w.split(";") if t])
        pd.DataFrame(sorted(sel.items(), key=lambda kv: -kv[1]), columns=["wavelet_level", "count"]).to_csv(
            out / "wavelet_selection.csv", index=False)

    if not args.skip_snr and args.snr_levels:
        rows = []
        for snr in args.snr_levels:
            scaled = evaluate.rescale_to_snr(clean, mixed, snr)
            for name in args.snr_variants:
                df = run_variant(name, variants.get(name, ablation_variants()[name]), scaled, clean, args.fs, indices, args.jobs)
                df.insert(1, "snr_level_db", snr)
                rows.append(df)
                print(f"SNR {snr:+.0f} dB {name:28s} MAE={df.mae.mean():.3f} RMSE={df.rmse.mean():.3f} CC={df.cc.mean():.3f} "
                      f"RRMSE_t={df.rrmse_t.mean():.3f} F1={df.f1.mean():.3f}")
        pd.concat(rows, ignore_index=True).to_csv(out / "snr_sweep_by_epoch.csv", index=False)

    if not args.skip_clean_control:
        rows = []
        for name in args.snr_variants:
            cfg = Config(fs=args.fs).with_updates(**ablation_variants()[name])
            for i in indices:
                processed, diag = process_epoch(clean[i], cfg)
                m = evaluate.clean_control_metrics(clean[i], processed)
                m.update(variant=name, epoch=int(i), detected=diag["detected_count"], processed=diag["processed_count"])
                rows.append(m)
        cc = pd.DataFrame(rows)
        cc.to_csv(out / "clean_control.csv", index=False)
        print(cc.groupby("variant")[["clean_altered", "clean_altered_sample_percent", "clean_rmse_change"]].mean())
    print("done ->", out)


if __name__ == "__main__":
    main()
