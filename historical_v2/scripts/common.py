"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awrls2 import Config, V1_REPRODUCTION, evaluate, process_epoch, wavelet_bank  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    import scipy
    info = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__, "wavelet_backend": wavelet_bank.BACKEND,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import pywt
        info["pywt"] = pywt.__version__
    except ImportError:
        info["pywt"] = None
    return info


def load_pairs(mixed_path: Path, clean_path: Path, limit: Optional[int] = None):
    mixed = np.load(mixed_path)
    clean = np.load(clean_path)
    if mixed.ndim == 1:
        mixed, clean = mixed[None, :], clean[None, :]
    if mixed.shape != clean.shape:
        raise ValueError(f"mixed {mixed.shape} and clean {clean.shape} differ")
    if limit:
        mixed, clean = mixed[:limit], clean[:limit]
    return np.asarray(mixed, float), np.asarray(clean, float)


def deterministic_split(n: int, seed: int = 20260925, dev_fraction: float = 0.5):
    """Reproducible development / holdout split of epoch indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_dev = int(round(dev_fraction * n))
    return np.sort(perm[:n_dev]), np.sort(perm[n_dev:])


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------
# Parallel epoch processing
# ----------------------------------------------------------------------------
_WORK: Dict[str, object] = {}


def _init_worker(mixed, clean, fs, overrides):
    _WORK["mixed"], _WORK["clean"], _WORK["fs"] = mixed, clean, fs
    _WORK["cfg"] = Config(fs=float(fs)).with_updates(**overrides)


def _run_one(i: int) -> dict:
    cfg = _WORK["cfg"]
    x, s = _WORK["mixed"][i], _WORK["clean"][i]
    t0 = time.perf_counter()
    out, diag = process_epoch(x, cfg)
    elapsed = time.perf_counter() - t0
    m = evaluate.paired_metrics(x, s, out, _WORK["fs"])
    m.update(epoch=i, elapsed_ms=1000 * elapsed, detected=diag["detected_count"], processed=diag["processed_count"],
             skipped=diag["skipped_count"], template_active=diag.get("template_active", False),
             wavelets=";".join(f"{e['wavelet']}:{e['level']}" for e in diag["events"]),
             sources=";".join(e["source"] for e in diag["events"]))
    return m


def run_variant(name: str, overrides: dict, mixed: np.ndarray, clean: np.ndarray, fs: float,
                indices: Iterable[int], jobs: int = 1) -> pd.DataFrame:
    indices = list(indices)
    if jobs <= 1:
        _init_worker(mixed, clean, fs, overrides)
        rows = [_run_one(i) for i in indices]
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                 initargs=(mixed, clean, fs, overrides)) as pool:
            rows = list(pool.map(_run_one, indices, chunksize=4))
    df = pd.DataFrame(rows)
    df.insert(0, "variant", name)
    return df


def summarise(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    g = df.groupby("variant", sort=False)[metrics]
    mean, sd, med = g.mean(), g.std(ddof=1), g.median()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    out = pd.concat({"mean": mean, "sd": sd, "median": med, "q1": q1, "q3": q3}, axis=1)
    out.columns = [f"{m}_{s}" for s, m in out.columns]
    return out


def paired_stats(df: pd.DataFrame, reference: str, metrics: List[str], higher_is_better: Dict[str, bool]) -> pd.DataFrame:
    ref = df[df.variant == reference].set_index("epoch")
    rows = []
    for name, sub in df.groupby("variant", sort=False):
        if name == reference:
            continue
        sub = sub.set_index("epoch")
        common = ref.index.intersection(sub.index)
        for m in metrics:
            x, y = sub.loc[common, m].to_numpy(float), ref.loc[common, m].to_numpy(float)
            if not higher_is_better.get(m, False):
                x, y = -x, -y  # so that positive diff = improvement
            st = evaluate.paired_comparison(x, y)
            st.update(variant=name, reference=reference, metric=m,
                      mean_diff_raw=float(np.nanmean(sub.loc[common, m] - ref.loc[common, m])))
            rows.append(st)
    out = pd.DataFrame(rows)
    if len(out):
        out["holm_p"] = np.nan
        for m, idx in out.groupby("metric").groups.items():
            out.loc[idx, "holm_p"] = evaluate.holm_correction(out.loc[idx, "wilcoxon_p"].tolist())
    return out


HIGHER_IS_BETTER = {"barr_percent": True, "recall_percent": True, "precision_percent": True, "f1": True,
                    "paper_recall_percent": True, "paper_precision_percent": True, "paper_f1": True,
                    "snr_db": True, "delta_snr_db": True, "cc": True, "mae": False, "rmse": False,
                    "rrmse_t": False, "rrmse_s": False, "dbp_total": False, "rmse_in_window": False,
                    "rmse_out_window": False, "dbp_delta": False, "dbp_theta": False, "dbp_alpha": False,
                    "dbp_beta": False, "dbp_gamma": False, "elapsed_ms": False}

PRIMARY_METRICS = ["mae", "rmse", "snr_db", "delta_snr_db", "cc", "rrmse_t", "rrmse_s", "barr_percent",
                   "recall_percent", "precision_percent", "f1", "paper_recall_percent", "paper_precision_percent",
                   "paper_f1", "dbp_delta", "dbp_theta", "dbp_alpha", "dbp_beta", "dbp_gamma", "dbp_total",
                   "rmse_in_window", "rmse_out_window", "elapsed_ms"]
