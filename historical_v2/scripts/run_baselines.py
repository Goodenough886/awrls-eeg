"""Run the existing comparison methods (MCA, FBSE-EWT-LPATV, OVME-REG, ITMS) from the
original `wj` project on the same paired epochs with the SAME evaluation code, so
that the new protocol metrics (support-based events, RRMSE, in/out-of-window error,
D_BP) exist for every method.  Optionally at several SNR levels.

python scripts/run_baselines.py --wj-src /path/to/wj/src --mixed data/mixed.npy --clean data/clean.npy \
    --out results/baselines --methods MCA FBSE-EWT-LPATV OVME-REG --snr-levels -7 -5 -3 -1 1 3

The `wj` package needs its own dependencies (PyWavelets, PyYAML, ...).  Method
configurations are the manuscript ones (pass --config-json to override).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import HIGHER_IS_BETTER, PRIMARY_METRICS, environment, evaluate, load_pairs, summarise, write_json


def load_methods(wj_src: Path):
    sys.path.insert(0, str(wj_src))
    from blink_repro.methods.fbse_ewt_lpatv import fbse_ewt_lpatv_clean
    from blink_repro.methods.itms import itms_clean
    from blink_repro.methods.mca import mca_clean
    try:
        from blink_repro.methods.ovme_reg import ovme_reg_clean
    except Exception:  # pragma: no cover
        ovme_reg_clean = None
    table = {"MCA": mca_clean, "FBSE-EWT-LPATV": fbse_ewt_lpatv_clean, "ITMS": itms_clean}
    if ovme_reg_clean is not None:
        table["OVME-REG"] = ovme_reg_clean
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wj-src", required=True, type=Path, help="path to the original wj/src directory")
    ap.add_argument("--mixed", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--methods", nargs="*", default=["MCA", "FBSE-EWT-LPATV", "OVME-REG"])
    ap.add_argument("--config-json", type=Path, default=None, help='{"MCA": {...}, "OVME-REG": {...}}')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--snr-levels", nargs="*", type=float, default=[])
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    methods = load_methods(args.wj_src)
    configs = json.loads(args.config_json.read_text()) if args.config_json else {}
    mixed, clean = load_pairs(args.mixed, args.clean, args.limit)
    write_json(out / "protocol.json", {"environment": environment(), "methods": args.methods, "configs": configs,
                                       "n_epochs": int(mixed.shape[0]), "snr_levels": args.snr_levels})
    levels = [None] + list(args.snr_levels)
    frames = []
    for snr in levels:
        data = mixed if snr is None else evaluate.rescale_to_snr(clean, mixed, snr)
        for name in args.methods:
            fn = methods[name]
            rows = []
            for i in range(data.shape[0]):
                t0 = time.perf_counter()
                res = fn(data[i], args.fs, configs.get(name))
                elapsed = time.perf_counter() - t0
                m = evaluate.paired_metrics(data[i], clean[i], np.asarray(res.cleaned, float), args.fs)
                m.update(variant=name, epoch=i, elapsed_ms=1000 * elapsed, snr_level_db=np.nan if snr is None else snr)
                rows.append(m)
            df = pd.DataFrame(rows)
            frames.append(df)
            print(f"{'native' if snr is None else f'{snr:+.0f} dB':8s} {name:16s} MAE={df.mae.mean():.3f} "
                  f"RMSE={df.rmse.mean():.3f} CC={df.cc.mean():.3f} F1={df.f1.mean():.3f} DBP={df.dbp_total.mean():.2f}")
            pd.concat(frames).to_csv(out / "baselines_by_epoch.csv", index=False)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "baselines_by_epoch.csv", index=False)
    native = all_df[all_df.snr_level_db.isna()]
    metrics = [m for m in PRIMARY_METRICS if m in native.columns]
    summarise(native, metrics).to_csv(out / "baselines_summary.csv")
    print("done ->", out)


if __name__ == "__main__":
    main()
