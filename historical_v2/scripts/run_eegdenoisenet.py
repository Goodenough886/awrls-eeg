"""EEGdenoiseNet ocular-artifact benchmark (Zhang et al., J. Neural Eng. 2021).

Download EEG_all_epochs.npy (4514 x 512) and EOG_all_epochs.npy (3400 x 512),
256 Hz, from the EEGdenoiseNet repository, then:

python scripts/run_eegdenoisenet.py --eeg EEG_all_epochs.npy --eog EOG_all_epochs.npy \
    --out results/eegdenoisenet --snr-levels -7 -6 -5 -4 -3 -2 -1 0 1 2 --n-test 340 --jobs 8

Protocol: each test EEG epoch is paired with one EOG epoch (seeded), mixed as
y = x + lambda * eog with lambda chosen for the requested SNR = 10 log10(P_x / P_(lambda eog)),
exactly as in the benchmark.  Metrics: RRMSE_t, RRMSE_s, CC (plus the full paired set).
The deep-learning numbers published on this benchmark (FCNN, SimpleCNN, ComplexCNN,
RNN, Novel CNN, 1D-ResCNN, GCTNet, ...) can then be quoted at the same SNR levels.

Note: epochs are 2 s long, so the cross-event template is rarely active (fewer than
`template_min_events` blinks); the method falls back to the event's own reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PRIMARY_METRICS, environment, evaluate, run_variant, summarise, write_json
from awrls2 import ablation_variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", required=True, type=Path)
    ap.add_argument("--eog", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=256.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--snr-levels", nargs="*", type=float, default=list(range(-7, 3)))
    ap.add_argument("--n-test", type=int, default=340, help="number of test epochs per SNR (0 = all EEG epochs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=["V1_reproduction", "PROPOSED"])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    eeg = np.load(args.eeg).astype(float)
    eog = np.load(args.eog).astype(float)
    rng = np.random.default_rng(args.seed)
    n_test = eeg.shape[0] if args.n_test == 0 else min(args.n_test, eeg.shape[0])
    eeg_idx = rng.choice(eeg.shape[0], n_test, replace=False)
    eog_idx = rng.choice(eog.shape[0], n_test, replace=True)
    clean = eeg[eeg_idx]
    noise = eog[eog_idx]
    write_json(out / "protocol.json", {"environment": environment(), "n_test": int(n_test), "seed": args.seed,
                                       "eeg_idx": eeg_idx.tolist(), "eog_idx": eog_idx.tolist(),
                                       "snr_levels": args.snr_levels, "variants": args.variants})
    variants = ablation_variants()
    frames = []
    for snr in args.snr_levels:
        lam = np.sqrt(np.sum(clean ** 2, axis=1, keepdims=True) / (np.sum(noise ** 2, axis=1, keepdims=True) * 10 ** (snr / 10)))
        mixed = clean + lam * noise
        for name in args.variants:
            df = run_variant(name, variants[name], mixed, clean, args.fs, range(n_test), args.jobs)
            df.insert(1, "snr_level_db", snr)
            frames.append(df)
            print(f"SNR {snr:+.0f} dB {name:24s} RRMSE_t={df.rrmse_t.mean():.3f} RRMSE_s={df.rrmse_s.mean():.3f} "
                  f"CC={df.cc.mean():.3f}  (input RRMSE_t={np.mean(np.sqrt(np.mean((lam*noise)**2,1))/np.sqrt(np.mean(clean**2,1))):.3f})")
            pd.concat(frames).to_csv(out / "eegdenoisenet_by_epoch.csv", index=False)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "eegdenoisenet_by_epoch.csv", index=False)
    table = all_df.groupby(["variant", "snr_level_db"])[["rrmse_t", "rrmse_s", "cc"]].agg(["mean", "std"])
    table.to_csv(out / "eegdenoisenet_table.csv")
    print(table.round(4))
    print("done ->", out)


if __name__ == "__main__":
    main()
