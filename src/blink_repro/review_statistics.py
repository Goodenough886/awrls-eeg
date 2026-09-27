"""Paired effect sizes with explicit experimental-unit boundaries."""
from __future__ import annotations

from pathlib import PureWindowsPath
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def kaya_subject(path):
    name = PureWindowsPath(path).stem
    # The distributed files contain both dashed and compact names.
    match = re.search(r"(?:Subject|Subjet)([A-M])(?=-|\d{6})", name)
    if match is None:
        raise ValueError(f"Unrecognized Kaya subject, explicit mapping required: {name}")
    return match.group(1)


def kaya_groups(frame, jobs, metrics, methods):
    metadata = pd.DataFrame([
        {"unit_id": j["unit_id"], "source_file": PureWindowsPath(j["path"]).name,
         "subject": kaya_subject(j["path"]), "target": j["channel"]}
        for j in jobs if j["dataset"] == "Kaya2018"
    ])
    if metadata.unit_id.duplicated().any():
        raise ValueError("Duplicate Kaya inventory unit")
    real = frame[frame.dataset == "Kaya2018"]
    if set(real.unit_id) != set(metadata.unit_id):
        raise ValueError("Kaya result units do not match the input inventory")
    data = real.merge(metadata, on="unit_id", validate="many_to_one")
    if len(data) != len(metadata) * len(methods) or data.duplicated(["unit_id", "method"]).any():
        raise ValueError("Incomplete or duplicated method panel")
    for _, group in data.groupby("unit_id"):
        if set(group.method) != set(methods):
            raise ValueError("Incomplete method panel")
    for _, group in metadata.groupby("source_file"):
        if len(group) != 3 or set(group.target) != {"Fp1", "Fp2", "Fz"} or group.subject.nunique() != 1:
            raise ValueError("Expected three independent target runs per file")
    # Missing values propagate; a method must not average an easier channel subset.
    file_level = data.groupby(["subject", "source_file", "method"])[metrics].agg(lambda x: x.mean(skipna=False)).reset_index()
    subject_level = file_level.groupby(["subject", "method"])[metrics].agg(lambda x: x.mean(skipna=False)).reset_index()
    return metadata, file_level, subject_level


def paired_effects(frame, metrics, methods, target, unit, *, independent_clusters=False,
                   bootstrap_samples=5000, seed=20260925):
    """Positive benefit favors target. Bootstrap resamples paired top-level units."""
    if frame.duplicated([unit, "method"]).any():
        raise ValueError("Aggregate repeated observations before inference")
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap draws required")
    rows = []
    for metric, direction in metrics.items():
        if direction not in {"higher", "lower"}:
            raise ValueError("Metric direction must be higher or lower")
        pivot = frame.pivot(index=unit, columns="method", values=metric).reindex(columns=methods)
        for other in methods:
            if other == target:
                continue
            pair = pivot[[target, other]].replace([np.inf, -np.inf], np.nan).dropna()
            difference = pair[target].to_numpy() - pair[other].to_numpy()
            benefit = difference if direction == "higher" else -difference
            nonzero = benefit[benefit != 0]
            ranks = rankdata(np.abs(nonzero))
            rb = float(np.sum(np.sign(nonzero) * ranks) / ranks.sum()) if len(ranks) else 0.0
            sd = float(np.std(benefit, ddof=1)) if len(benefit) > 1 else np.nan
            low = high = np.nan
            if independent_clusters and len(benefit) >= 2:
                rng = np.random.default_rng(seed)
                samples = rng.choice(benefit, (bootstrap_samples, len(benefit)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
            rows.append({"metric": metric, "target": target, "comparator": other,
                         "unit": unit, "n_paired": len(pair), "n_excluded": len(pivot)-len(pair),
                         "target_mean": float(pair[target].mean()), "comparator_mean": float(pair[other].mean()),
                         "mean_target_minus_comparator": float(np.mean(difference)) if len(pair) else np.nan,
                         "mean_benefit": float(np.mean(benefit)) if len(pair) else np.nan,
                         "median_benefit": float(np.median(benefit)) if len(pair) else np.nan,
                         "paired_cohen_dz_benefit": float(np.mean(benefit)/sd) if sd > 0 else np.nan,
                         "rank_biserial_benefit": rb if len(pair) else np.nan,
                         "win_fraction_ties_half": float(np.mean((benefit > 0) + 0.5*(benefit == 0))) if len(pair) else np.nan,
                         "benefit_ci95_low": low, "benefit_ci95_high": high,
                         "ci_scope": "paired_subject_bootstrap_pointwise_not_simultaneous" if independent_clusters else "not_computed_dependence_unresolved",
                         "bootstrap_seed": seed, "bootstrap_draws": bootstrap_samples if independent_clusters else 0})
    return pd.DataFrame(rows)
