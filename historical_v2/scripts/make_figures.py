"""Manuscript figures from the result CSVs (matplotlib, vector PDF + PNG).

python scripts/make_figures.py --ablation results/ablation --mixed data/mixed.npy --clean data/clean.npy \
    [--baselines results/baselines] [--sensitivity results/sensitivity] [--eegdenoisenet results/eegdenoisenet] --out figures

Figures
-------
fig_example_waveform   input / clean reference / V1 / PROPOSED with a zoom on one event
fig_ablation_forest    paired mean difference vs PROPOSED with 95 % bootstrap CI per removed component
fig_snr_curves         RRMSE_t, CC and F1 versus input SNR (variants + baselines if given)
fig_wavelet_selection  histogram of selected mother wavelet x level
fig_sensitivity        one-at-a-time parameter curves (MAE, F1)
fig_clean_control      false-correction rate on artifact-free epochs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import Config, evaluate, load_pairs, process_epoch
from awrls2 import ablation_variants

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#777777", "#000000"])


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_example(mixed, clean, fs, out, epoch=0):
    variants = ablation_variants()
    v1 = process_epoch(mixed[epoch], Config(fs=fs).with_updates(**variants["V1_reproduction"]))[0]
    prop, diag = process_epoch(mixed[epoch], Config(fs=fs))
    t = np.arange(mixed.shape[1]) / fs
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot(t, mixed[epoch], color="0.6", lw=0.7, label="contaminated input")
    ax.plot(t, clean[epoch], color="k", lw=0.8, label="clean reference")
    ax.plot(t, v1, color="tab:blue", lw=0.8, label="AWRLS v1")
    ax.plot(t, prop, color="tab:red", lw=0.8, label="AWRLS-2 (proposed)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (a.u.)"); ax.legend(ncol=4, fontsize=7, loc="upper right")
    ev = evaluate.support_events(mixed[epoch] - clean[epoch], fs)
    if ev:
        e = max(ev, key=lambda z: z.score)
        lo, hi = max(0, e.start - int(0.3 * fs)), min(mixed.shape[1], e.end + int(0.3 * fs))
        ax2 = axes[1]
        ax2.plot(t[lo:hi], mixed[epoch][lo:hi], color="0.6", lw=0.8)
        ax2.plot(t[lo:hi], clean[epoch][lo:hi], color="k", lw=1.0)
        ax2.plot(t[lo:hi], v1[lo:hi], color="tab:blue", lw=1.0)
        ax2.plot(t[lo:hi], prop[lo:hi], color="tab:red", lw=1.0)
        ax2.axvspan(e.start / fs, e.end / fs, color="tab:orange", alpha=0.12, label="implanted artifact support")
        for r in diag["events"]:
            if r["end"] > lo and r["start"] < hi:
                ax2.axvspan(max(r["start"], lo) / fs, min(r["end"], hi) / fs, color="tab:red", alpha=0.08)
        ax2.set_xlim(t[lo], t[hi - 1])
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("amplitude (a.u.)"); ax2.legend(fontsize=7)
        ax2.set_title("zoom on the largest event (orange: truth support, red: processed window)", fontsize=8)
    save(fig, out, "fig_example_waveform")


def fig_forest(stats_path: Path, out: Path, metrics=("mae", "rmse", "cc", "f1", "dbp_total")):
    st = pd.read_csv(stats_path)
    st = st[st.variant.str.startswith("A") | st.variant.str.startswith("C") | st.variant.str.startswith("D") | (st.variant == "V1_reproduction")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.3 * len(metrics) + 1, 0.28 * st.variant.nunique() + 1.2), sharey=True)
    names = list(dict.fromkeys(st.variant))
    for ax, m in zip(axes, metrics):
        sub = st[st.metric == m].set_index("variant").reindex(names)
        y = np.arange(len(names))
        ax.errorbar(sub.mean_diff, y, xerr=[sub.mean_diff - sub.ci95_low, sub.ci95_high - sub.mean_diff],
                    fmt="o", ms=3, color="tab:red", ecolor="0.4", capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("difference vs PROPOSED\n(+ = variant better)", fontsize=7)
        sig = sub.holm_p < 0.05
        for yi, s in zip(y, sig):
            if s:
                ax.text(ax.get_xlim()[1], yi, "*", va="center", fontsize=8)
    display_names = [n.replace("D1_online_rls_one_pass", "D1_one_pass_RLS_offline").replace("D2_direct_subtraction_of_reference", "D2_self_LS_zero_ridge") for n in names]
    axes[0].set_yticks(np.arange(len(names))); axes[0].set_yticklabels(display_names, fontsize=7)
    save(fig, out, "fig_ablation_forest")


def fig_snr(ablation_dir: Path, baselines_dir: Path | None, out: Path):
    df = pd.read_csv(ablation_dir / "snr_sweep_by_epoch.csv")
    if baselines_dir and (baselines_dir / "baselines_by_epoch.csv").exists():
        b = pd.read_csv(baselines_dir / "baselines_by_epoch.csv")
        df = pd.concat([df, b[b.snr_level_db.notna()]], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for m, ax, lab in zip(("rrmse_t", "cc", "f1"), axes, ("RRMSE$_t$", "CC", "F1")):
        g = df.groupby(["variant", "snr_level_db"])[m].agg(["mean", "sem"]).reset_index()
        for name, sub in g.groupby("variant"):
            ax.errorbar(sub.snr_level_db, sub["mean"], yerr=1.96 * sub["sem"], label=name, marker="o", ms=3, lw=1, capsize=2)
        ax.set_xlabel("input SNR (dB)"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=6)
    save(fig, out, "fig_snr_curves")


def fig_wavelets(ablation_dir: Path, out: Path):
    p = ablation_dir / "wavelet_selection.csv"
    if not p.exists():
        return
    sel = pd.read_csv(p).head(25)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.bar(sel.wavelet_level, sel["count"], color="tab:red")
    ax.set_ylabel("events"); ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.set_title("selected mother wavelet : level (PROPOSED)", fontsize=9)
    save(fig, out, "fig_wavelet_selection")


def fig_sensitivity(sens_dir: Path, out: Path):
    p = sens_dir / "sensitivity_by_epoch.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    params = list(dict.fromkeys(df.parameter))
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow))
    fig.suptitle("Development sensitivity: red circles = MAE (left); blue squares = F1 (right)", fontsize=10)
    for ax, prm in zip(axes.ravel(), params):
        sub = df[df.parameter == prm].groupby("value")[["mae", "f1"]].mean()
        xs = [str(v) for v in sub.index]
        ax.plot(xs, sub.mae, marker="o", ms=3, color="tab:red", label="MAE")
        ax.set_title(prm, fontsize=8); ax.tick_params(labelsize=7)
        ax2 = ax.twinx(); ax2.plot(xs, sub.f1, marker="s", ms=3, color="tab:blue", label="F1"); ax2.tick_params(labelsize=7)
        ax2.spines["top"].set_visible(False)
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(max(0.0, lo), min(1.0, hi))
        if prm in ("gls_floor", "ridge_adaptive_kappa", "boundary_pad_s"):
            ax.set_facecolor("#f2f2f2")
            ax.text(0.03, 0.03, "unchanged in default-mode dev run", transform=ax.transAxes, fontsize=6)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    save(fig, out, "fig_sensitivity")


def fig_clean_control(ablation_dir: Path, out: Path):
    p = ablation_dir / "clean_control.csv"
    if not p.exists():
        return
    cc = pd.read_csv(p).groupby("variant")[["clean_altered", "clean_altered_sample_percent", "clean_rmse_change"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
    for ax, (col, lab) in zip(axes, (("clean_altered", "fraction of clean epochs altered"),
                                     ("clean_altered_sample_percent", "altered samples (%)"),
                                     ("clean_rmse_change", "RMS change on clean EEG"))):
        labels = [dict(V1_reproduction="V1", PROPOSED="AWRLS-2").get(x, "No surrogate") for x in cc.index]
        ax.bar(labels, cc[col], color=["#009E73", "#D55E00", "#0072B2"][: len(cc)])
        ax.set_title(lab, fontsize=8); ax.tick_params(axis="x", labelsize=7)
    save(fig, out, "fig_clean_control")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--mixed", type=Path); ap.add_argument("--clean", type=Path)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--sensitivity", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--example-epoch", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mixed and args.clean:
        mixed, clean = load_pairs(args.mixed, args.clean)
        fig_example(mixed, clean, args.fs, args.out, args.example_epoch)
    if (args.ablation / "stats_vs_PROPOSED.csv").exists():
        fig_forest(args.ablation / "stats_vs_PROPOSED.csv", args.out)
    if (args.ablation / "snr_sweep_by_epoch.csv").exists():
        fig_snr(args.ablation, args.baselines, args.out)
    fig_wavelets(args.ablation, args.out)
    if args.sensitivity:
        fig_sensitivity(args.sensitivity, args.out)
    fig_clean_control(args.ablation, args.out)
    print("figures ->", args.out)


if __name__ == "__main__":
    main()
