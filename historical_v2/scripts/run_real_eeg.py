"""Real single-channel EEG (Kaya 2018 or any 1-D recording).

Two input modes
---------------
1. --npy-dir DIR          every *.npy in DIR is one preprocessed target channel at --fs
2. --wj-src ... --mat-glob  Kaya2018 .mat containers read and preprocessed with the ORIGINAL
                           strict single-channel reader (read_target + preprocess_target), so
                           the input is byte-identical to the manuscript pipeline.

For every record x channel the script runs the requested variants (block_s=30 s,
overlap 4 s by default), writes the cleaned vectors, and computes:
  * reference-free input/output metrics of the manuscript (candidate-window LF energy
    reduction, outside-window Pearson r, 13-40 Hz input/output correlation);
  * if a silver reference exists (<stem>_<channel>_silver.npy produced by
    make_silver_standard_ica.py), the full paired metric set against that reference.

python scripts/run_real_eeg.py --wj-src /path/to/wj/src --mat-glob "/data/Kalya/*.mat" \
    --channels Fp1 Fp2 Fz --seconds 30 --out results/kaya --variants V1_reproduction PROPOSED
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import Config, environment, evaluate, write_json
from awrls2 import ablation_variants, clean

KAYA_PREPROCESSING = {"reference_mode": "none", "trim_zero_edges": True, "minimum_zero_edge_s": 0.25,
                      "robust_center": True, "highpass_hz": 0.5, "lowpass_hz": 45.0, "filter_order": 4,
                      "filter_segment_s": 120.0, "final_robust_center": True}


def iter_inputs(args):
    """Yield (record, channel, x, fs, silver_path_or_None)."""
    if args.npy_dir:
        folder = Path(args.npy_dir)
        for p in sorted(folder.glob("*.npy")):
            if p.stem.endswith("_silver"):
                continue
            base = p.stem[:-4] if p.stem.endswith("_raw") else p.stem
            record, _, ch = base.rpartition("_") if "_" in base else (base, "", "ch")
            silver = (args.silver_dir or folder) / f"{base}_silver.npy"
            yield record or base, ch, np.load(p).astype(float), args.fs, (silver if silver.exists() else None)
    else:
        sys.path.insert(0, str(args.wj_src))
        from blink_repro.strict_single_channel import preprocess_target, read_target
        for f in sorted(glob.glob(args.mat_glob)):
            for ch in args.channels:
                raw, fs, _ = read_target(Path(f), "Kaya2018", ch)
                x, fs2, _ = preprocess_target(raw, fs, ch, "Kaya2018", KAYA_PREPROCESSING, target_fs=args.fs)
                silver = (args.silver_dir or Path(f).parent) / f"{Path(f).stem}_{ch}_silver.npy"
                yield Path(f).stem, ch, np.asarray(x, float), fs2, (silver if silver.exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", type=Path, default=None)
    ap.add_argument("--wj-src", type=Path, default=None)
    ap.add_argument("--mat-glob", type=str, default=None)
    ap.add_argument("--channels", nargs="*", default=["Fp1", "Fp2", "Fz"])
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="analyse the first N seconds (0 = full record)")
    ap.add_argument("--block-s", type=float, default=30.0)
    ap.add_argument("--overlap-s", type=float, default=4.0)
    ap.add_argument("--variants", nargs="*", default=["V1_reproduction", "PROPOSED"])
    ap.add_argument("--silver-dir", type=Path, default=None, help="directory with <stem>_<ch>_silver.npy references")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not args.npy_dir and not (args.wj_src and args.mat_glob):
        raise SystemExit("give --npy-dir or (--wj-src and --mat-glob)")
    out = args.out
    (out / "cleaned").mkdir(parents=True, exist_ok=True)
    variants = ablation_variants()
    write_json(out / "protocol.json", {"environment": environment(), "args": vars(args), "variants": {k: variants[k] for k in args.variants},
                                       "preprocessing": KAYA_PREPROCESSING if not args.npy_dir else "as provided"})
    rows = []
    for stem, ch, x, fs, sp in iter_inputs(args):
        if args.seconds and len(x) > int(args.seconds * fs):
            x = x[: int(args.seconds * fs)]
        silver = None
        if sp is not None:
            silver = np.load(sp).astype(float)[: len(x)]
            if len(silver) != len(x):
                print(f"warning: silver length mismatch for {stem} {ch}; skipped")
                silver = None
        for name in args.variants:
            cfg = Config(fs=fs).with_updates(**variants[name])
            t0 = time.perf_counter()
            y, diag = clean(x, fs, cfg, block_s=args.block_s if len(x) > args.block_s * fs else None, overlap_s=args.overlap_s)
            elapsed = time.perf_counter() - t0
            np.save(out / "cleaned" / f"{stem}_{ch}_{name}.npy", y)
            m = evaluate.real_metrics(x, y, fs)
            m.update(record=stem, channel=ch, variant=name, seconds=len(x) / fs, elapsed_s=elapsed,
                     detected=diag["detected_count"], processed=diag["processed_count"])
            if silver is not None:
                pm = evaluate.paired_metrics(x, silver, y, fs)
                m.update({f"silver_{k}": v for k, v in pm.items()})
            rows.append(m)
            print(f"{stem} {ch} {name:20s} LFred={m['lf_energy_reduction_percent']:.1f}% outCC={m['outside_window_cc']:.3f} "
                  f"hfCC={m['hf_13_40_cc']:.3f} events={m['candidate_count']} {elapsed*1000:.0f} ms"
                  + (f" | silver CC={m['silver_cc']:.3f} RRMSE_t={m['silver_rrmse_t']:.3f}" if silver is not None else ""))
            pd.DataFrame(rows).to_csv(out / "real_by_unit.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "real_by_unit.csv", index=False)
    num = df.select_dtypes("number").columns
    df.groupby("variant")[num].median().to_csv(out / "real_median.csv")
    df.groupby("variant")[num].quantile(0.25).to_csv(out / "real_q1.csv")
    df.groupby("variant")[num].quantile(0.75).to_csv(out / "real_q3.csv")
    print(df.groupby("variant")[["lf_energy_reduction_percent", "outside_window_cc", "hf_13_40_cc"]].median())
    print("done ->", out)


if __name__ == "__main__":
    main()
