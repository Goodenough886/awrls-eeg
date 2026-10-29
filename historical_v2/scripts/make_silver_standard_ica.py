"""Multichannel ICA "silver standard" for quantitative real-EEG evaluation.

Kaya 2018 records contain 19 scalp channels.  A multichannel ICA (which the
single-channel methods never see) can remove the ocular components; the
resulting Fp1/Fp2/Fz traces are a *silver* reference (not ground truth, but
independent of every single-channel method and widely used for this purpose).
Each single-channel method is then scored against it with the paired metrics.

Requires: pip install mne   (optional: mne-icalabel for ICLabel-based selection)

python scripts/make_silver_standard_ica.py --mat-glob "/data/Kalya/*.mat" --channels Fp1 Fp2 Fz \
    --out-dir results/silver --fs 200 --seconds 0

Outputs <stem>_<ch>_silver.npy (and <stem>_<ch>_raw.npy, same preprocessing as the
single-channel pipeline so that the two are sample aligned) plus a JSON log of the
ICs removed and their correlation with the frontal low-frequency signal.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


def load_kaya(path: Path):
    rec = loadmat(path, simplify_cells=True, variable_names=["o"])["o"]
    names = [str(n) for n in np.asarray(rec["chnames"]).ravel()]
    data = np.asarray(rec["data"], float)
    if data.shape[0] == len(names):
        data = data.T
    elif data.shape[1] == len(names) + 1:
        data = data[:, : len(names)]
    return data, names, float(rec["sampFreq"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-glob", required=True)
    ap.add_argument("--channels", nargs="*", default=["Fp1", "Fp2", "Fz"])
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = whole record")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--method", default="infomax", choices=["infomax", "fastica", "picard"])
    ap.add_argument("--corr-threshold", type=float, default=0.6)
    ap.add_argument("--use-iclabel", action="store_true")
    args = ap.parse_args()
    import mne
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = {}
    for f in sorted(glob.glob(args.mat_glob)):
        data, names, fs0 = load_kaya(Path(f))
        scalp = [n for n in names if n.upper() not in ("X3", "X5", "A1", "A2", "TRIGGER", "MARKER")]
        idx = [names.index(n) for n in scalp]
        x = data[:, idx].T  # channels x samples
        if fs0 > args.fs:
            from fractions import Fraction
            r = Fraction(int(args.fs), int(fs0))
            x = resample_poly(x, r.numerator, r.denominator, axis=1)
        fs = args.fs
        if args.seconds:
            x = x[:, : int(args.seconds * fs)]
        x = x - np.median(x, axis=1, keepdims=True)
        sos = butter(4, [0.5, 45.0], btype="band", fs=fs, output="sos")
        x = sosfiltfilt(sos, x, axis=1)
        info = mne.create_info(scalp, fs, ch_types="eeg")
        raw = mne.io.RawArray(x * 1e-6, info, verbose=False)
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass
        ica = mne.preprocessing.ICA(n_components=min(len(scalp) - 1, 18), method=args.method,
                                    random_state=97, max_iter="auto", verbose=False)
        ica.fit(raw, verbose=False)
        sources = ica.get_sources(raw).get_data()
        front = [i for i, n in enumerate(scalp) if n.upper() in ("FP1", "FP2")]
        ref = sosfiltfilt(butter(4, 12.0, btype="low", fs=fs, output="sos"), x[front].mean(axis=0))
        corrs = [abs(np.corrcoef(ref, s)[0, 1]) for s in sources]
        exclude = [i for i, c in enumerate(corrs) if c >= args.corr_threshold]
        if args.use_iclabel:
            try:
                from mne_icalabel import label_components
                labels = label_components(raw, ica, method="iclabel")
                exclude = sorted(set(exclude) | {i for i, lab in enumerate(labels["labels"]) if lab == "eye blink"})
            except Exception as exc:  # pragma: no cover
                print("ICLabel unavailable:", exc)
        if not exclude:
            exclude = [int(np.argmax(corrs))]
        ica.exclude = exclude
        cleaned = ica.apply(raw.copy(), verbose=False).get_data() * 1e6
        stem = Path(f).stem
        for ch in args.channels:
            if ch not in scalp:
                continue
            k = scalp.index(ch)
            np.save(args.out_dir / f"{stem}_{ch}_raw.npy", x[k])
            np.save(args.out_dir / f"{stem}_{ch}_silver.npy", cleaned[k])
        log[stem] = {"excluded_ics": exclude, "ic_frontal_corr": [float(c) for c in corrs], "fs": fs, "n_samples": int(x.shape[1])}
        print(stem, "excluded ICs", exclude, "corr", np.round([corrs[i] for i in exclude], 2))
    (args.out_dir / "ica_log.json").write_text(json.dumps(log, indent=2))
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
