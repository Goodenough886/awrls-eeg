"""Training-free baselines on the FROZEN AWRLS3 protocol (no tuning on test data).

Runs MCA, FBSE-EWT-LPATV and OVME-REG -- the implementations and settings of the earlier
study (wj/src, config.strict_single_channel.yaml, config.ovme_reg.json) -- on

  public   the 554 frozen public test pairs x {-7,-5,-3,-1,+1,+3 dB, artifact-free}; pairs and
           power-ratio mixing are those of evaluate_models.tasks_for('test'), and every input is
           checked against the INPUT rows of results/test_models.csv before any method runs
  internal the 300 semi-simulated epochs: native, artifact-free and the six artifact-rescaled SNRs
           (same construction as full_internal.py)
  kaya     Fp1/Fp2/Fz of the Kaya recordings, both input recipes ('raw' single-channel and
           'silver' = matched to the ICA input, scored against the ICA reference), first
           --kaya-seconds seconds (0 = full length), processed in 120-s blocks with 2-s overlap
           as in the earlier study

Metrics come from the same functions as the frozen evaluation (evaluate_classical.metrics,
awrls2.evaluate.paired_metrics, full_real.context/score), so the rows merge directly with
results/test_models.csv, results/full_internal/by_epoch.csv and results/full_kaya/by_unit.csv.

Usage (from anywhere; --root is the AWRLS3 project folder that contains scripts/ and results/):
  python run_frozen_baselines.py --root D:/AWRLS3 --wj-src D:/wj/src --parts public internal --jobs 8
  python run_frozen_baselines.py --root D:/AWRLS3 --wj-src D:/wj/src --parts kaya --kaya-seconds 30 \
         --previous <the --previous folder used by full_real.py> --jobs 8
Add --methods to run a subset, --limit N for a quick smoke test, --ovme-printed-mean to add the
sensitivity variant of OVME-REG. Output: <root>/results/frozen_baselines/<part>.csv + protocol.json.
The run is resumable: finished (part, method, condition) blocks are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '1'
import numpy as np
import pandas as pd

# Settings of the earlier study (identical to wj/config.strict_single_channel.yaml "baselines" and
# wj/config.ovme_reg.json). They are passed explicitly so that the run does not depend on a YAML reader.
CONFIGS = {
    'MCA': {'wavelet': 'db4', 'level': 4, 'threshold': 'soft', 'lambda': 4.0, 'iterations': 14,
            'inner_iterations': 5, 'threshold_schedule': 'linear'},
    'FBSE-EWT-LPATV': {'fbse_block_s': 5.0, 'fbse_overlap_s': 1.0, 'gamma_upper_hz': 75.0, 'polynomial_order': 3,
                       'polynomial_block_s': 1.0, 'polynomial_overlap_s': 0.5, 'tv_lambda': 0.35, 'admm_rho': 1.0,
                       'admm_iterations': 120, 'admm_tolerance': 1e-5, 'outer_iterations': 3,
                       'preserve_uncovered_residual': True},
    'OVME-REG': {'alpha_min': 1000.0, 'alpha_max': 10000.0, 'omega_min_hz': 0.5, 'omega_max_hz': 7.5, 'population': 30,
                 'iterations': 20, 'vme_max_iterations': 300, 'vme_tolerance': 1e-7, 'vme_tau': 0.0,
                 'objective': 'stated_mse', 'polarity': 'positive', 'before_peak_s': 0.125, 'after_peak_s': 0.375,
                 'block_s': 3.0, 'overlap_s': 1.0, 'seed': 20260926},
}
CONFIGS['OVME-REG_mean'] = dict(CONFIGS['OVME-REG'], objective='printed_mean')
SNRS = [-7, -5, -3, -1, 1, 3]
_STATE = {}


def _setup(root, wj_src):
    """Worker initializer: make the AWRLS3 project and the wj baselines importable."""
    for p in (str(wj_src), str(Path(root) / 'scripts'), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from evaluate_classical import metrics  # same metric code as the frozen evaluation
    from awrls2.evaluate import paired_metrics
    from blink_repro.methods.mca import mca_clean
    from blink_repro.methods.fbse_ewt_lpatv import fbse_ewt_lpatv_clean
    from blink_repro.methods.ovme_reg import ovme_reg_clean
    _STATE.update(metrics=metrics, paired_metrics=paired_metrics,
                  fns={'MCA': mca_clean, 'FBSE-EWT-LPATV': fbse_ewt_lpatv_clean,
                       'OVME-REG': ovme_reg_clean, 'OVME-REG_mean': ovme_reg_clean})


def _run(method, x, fs):
    out = _STATE['fns'][method](np.asarray(x, float), float(fs), dict(CONFIGS[method]))
    y = np.asarray(getattr(out, 'cleaned', out), float)
    if y.shape != np.shape(x) or not np.all(np.isfinite(y)):
        raise FloatingPointError(f'{method} returned an invalid output')
    return y


def _work_paired(task):
    part, method, ident, condition, fs, x, s, extra = task
    t = time.perf_counter()
    y = _run(method, x, fs)
    row = dict(variant=method, epoch=ident, condition=condition, elapsed_s=time.perf_counter() - t,
               **_STATE['metrics'](x, s, y, fs), **extra)
    if part == 'internal' and condition != 'clean':
        ev = _STATE['paired_metrics'](x, s, y, fs)
        for k in ['f1', 'recall_percent', 'precision_percent', 'barr_percent', 'rmse_in_window', 'rmse_out_window']:
            row['removed_component_' + k] = ev[k]
    return row


def _work_kaya(task):
    method, kind, key, previous, seconds, block_s, overlap_s = task
    from full_real import load_task, context, score, common_row
    from awrls2.core import _overlap_add
    x, s, source = load_task((kind, key, previous))
    if seconds > 0:
        n = int(round(seconds * 200))
        x = x[:n]
        s = None if s is None else s[:n]
    t = time.perf_counter()
    y = _overlap_add(x, 200.0, lambda v: _run(method, v, 200.0), block_s, overlap_s)
    row = common_row(kind, key, x, source)
    row.update(variant=method, compute_s=time.perf_counter() - t, analysed_seconds=len(x) / 200.0,
               **score(x, y, s, context(x)))
    return row


def _public_tasks(root, limit):
    p = json.loads((root / 'results' / 'source_split.json').read_text())
    from evaluate_models import tasks_for
    tasks = tasks_for('test', p)
    frozen = pd.read_csv(root / 'results' / 'test_models.csv')
    frozen = frozen[frozen.variant == 'INPUT'].set_index(['condition', 'epoch'])
    worst = 0.0
    for (j, x, s, fs, cond) in tasks:
        m = _STATE['metrics'](x, s, x, fs)
        ref = frozen.loc[(cond, j)]
        worst = max(worst, abs(m['rrmse_t'] - ref.rrmse_t), abs(m['cc'] - ref.cc))
    if worst > 1e-9:
        raise RuntimeError(f'public inputs differ from the frozen test (max deviation {worst:.3g}); check AWRLS_PUBLIC_DATA')
    if limit:
        tasks = [t for t in tasks if t[0] < limit]
    return tasks, worst


def _internal_tasks(root, limit):
    p = json.loads((root / 'results' / 'source_split.json').read_text())
    sys.path.insert(0, str(root / 'scripts'))
    from data_paths import paths
    _, ip = paths(p)
    X = np.load(ip / 'mixed.npy'); S = np.load(ip / 'clean.npy')
    train = set(p['internal']['train'])
    n = len(X) if not limit else min(limit, len(X))
    tasks = []
    for condition in ['native', 'clean'] + SNRS:
        for i in range(n):
            x, s = X[i], S[i]
            if condition == 'clean':
                xx = s.copy()
            elif condition == 'native':
                xx = x.copy()
            else:
                a = x - s
                xx = s + a * np.sqrt(np.sum(s * s) / (np.sum(a * a) + 1e-12) / 10 ** (condition / 10))
            tasks.append((i, xx, s, 200, str(condition),
                          {'split_role': 'training_seen' if i in train else 'historical_development'}))
    ref_csv = root / 'results' / 'full_internal' / 'by_epoch.csv'
    if ref_csv.exists():  # inputs must reproduce the frozen semi-simulated evaluation
        ref = pd.read_csv(ref_csv)
        ref = ref[ref.variant == 'INPUT'].assign(condition=lambda d: d.condition.astype(str)).set_index(['condition', 'epoch'])
        worst = 0.0
        for (i, xx, s, fs, cond, _) in tasks:
            m = _STATE['metrics'](xx, s, xx, fs)
            r = ref.loc[(cond, i)]
            worst = max(worst, abs(m['rmse'] - r.rmse), abs(m['cc'] - r.cc))
        if worst > 1e-9:
            raise RuntimeError(f'semi-simulated inputs differ from the frozen evaluation (max deviation {worst:.3g}); check AWRLS_INTERNAL_DATA')
    return tasks


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--wj-src', type=Path, required=True)
    ap.add_argument('--parts', nargs='+', choices=['public', 'internal', 'kaya'], default=['public', 'internal'])
    ap.add_argument('--methods', nargs='+', default=['MCA', 'FBSE-EWT-LPATV', 'OVME-REG'])
    ap.add_argument('--ovme-printed-mean', action='store_true')
    ap.add_argument('--previous', type=Path, default=None, help='folder passed as --previous to full_real.py (kaya part)')
    ap.add_argument('--kaya-seconds', type=float, default=30.0, help='0 = full length')
    ap.add_argument('--kaya-block-s', type=float, default=120.0)
    ap.add_argument('--kaya-overlap-s', type=float, default=2.0)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--limit', type=int, default=None, help='first N epochs/pairs only (smoke test)')
    args = ap.parse_args()
    root = args.root.resolve()
    methods = list(args.methods) + (['OVME-REG_mean'] if args.ovme_printed_mean else [])
    out = root / 'results' / 'frozen_baselines'
    out.mkdir(parents=True, exist_ok=True)
    _setup(root, args.wj_src)
    protocol = {'methods': methods, 'configs': {m: CONFIGS[m] for m in methods}, 'parts': args.parts,
                'limit': args.limit, 'wj_src': str(args.wj_src), 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                'baseline_source_sha256': {str(p.relative_to(args.wj_src)): _sha(p) for p in
                                           sorted((args.wj_src / 'blink_repro' / 'methods').glob('*.py'))}}
    for part in args.parts:
        csv = out / f'{part}.csv'
        done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        rows = done.to_dict('records')
        if part in ('public', 'internal'):
            if part == 'public':
                tasks, worst = _public_tasks(root, args.limit)
                protocol['public_input_check_max_abs_deviation'] = worst
                tasks = [(j, x, s, fs, c, {}) for (j, x, s, fs, c) in tasks]
            else:
                tasks = _internal_tasks(root, args.limit)
            for method in methods:
                for cond in sorted({t[4] for t in tasks}):
                    if not done.empty and ((done.variant == method) & (done.condition.astype(str) == cond)).any():
                        continue
                    todo = [(part, method, t[0], t[4], t[3], t[1], t[2], t[5]) for t in tasks if t[4] == cond]
                    t0 = time.perf_counter()
                    with ProcessPoolExecutor(args.jobs, initializer=_setup, initargs=(root, args.wj_src)) as pool:
                        new = list(pool.map(_work_paired, todo, chunksize=4))
                    rows.extend(new)
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    d = pd.DataFrame(new)
                    print(f'{part:8s} {method:15s} {cond:>12s} n={len(d):4d} RRMSE_t={d.rrmse_t.mean():.3f} '
                          f'CC={d.cc.mean():.3f} DBP={d.dbp_total.mean():.2f} ({time.perf_counter() - t0:.0f} s)', flush=True)
        else:
            if args.previous is None:
                raise SystemExit('--previous is required for the kaya part')
            keys = sorted(q.stem for q in (args.previous / 'kaya_preprocessed').glob('*.npy'))
            if args.limit:
                keys = keys[:args.limit]
            for method in methods:
                if not done.empty and (done.variant == method).sum() >= 2 * len(keys):
                    continue
                todo = [(method, kind, key, str(args.previous), args.kaya_seconds, args.kaya_block_s, args.kaya_overlap_s)
                        for kind in ('raw', 'silver') for key in keys]
                t0 = time.perf_counter()
                with ProcessPoolExecutor(args.jobs, initializer=_setup, initargs=(root, args.wj_src)) as pool:
                    new = list(pool.map(_work_kaya, todo, chunksize=1))
                rows = [r for r in rows if r.get('variant') != method] + new
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(f'kaya     {method:15s} units={len(new)} ({time.perf_counter() - t0:.0f} s)', flush=True)
        protocol[f'{part}_rows'] = len(rows)
    protocol['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('done ->', out)


if __name__ == '__main__':
    main()
