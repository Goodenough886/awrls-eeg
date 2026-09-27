"""Portable orchestration of the frozen manuscript algorithms and evaluators."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'evaluation')]

import numpy as np
import pandas as pd
import pywt
import scipy
from scipy.signal import get_window
import metric_core as paper
import compute_dbp as dbp
from blink_repro.methods.mca import mca_clean
from blink_repro.methods.fbse_ewt_lpatv import fbse_ewt_lpatv_clean
from blink_repro.methods.ovme_reg import ovme_reg_clean
from blink_repro.strict_single_channel import (
    awrls_settings, read_target, preprocess_target, single_vector, strict_awrls_clean,
)
from blink_repro.signal_utils import overlap_add

METHODS = ('AWRLS-strict', 'MCA', 'FBSE-EWT-LPATV', 'OVME-HHO-REG')
COLORS = {'AWRLS-strict': '#ff1744', 'MCA': '#59b5df',
          'FBSE-EWT-LPATV': '#2da875', 'OVME-HHO-REG': '#b23aee',
          'OVME-printed-Eq10': '#b23aee', 'ITMS': '#6B5B4B'}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_one(method, x, fs, cfg, real=False):
    if np.iscomplexobj(x):
        raise ValueError('Expected real-valued input.')
    values = single_vector(x)
    if not np.isfinite(values).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError('A finite vector and positive sampling rate are required.')
    if method == 'AWRLS-strict':
        result = strict_awrls_clean(values, fs, cfg['awrls'],
                                   block_s=cfg['processing']['awrls_block_s'] if real else None,
                                   overlap_s=cfg['processing']['awrls_overlap_s'])
        return result.cleaned
    if method.startswith('OVME-'):
        objective = {'OVME-HHO-REG': 'stated_mse', 'OVME-printed-Eq10': 'printed_mean'}[method]
        return ovme_reg_clean(values, fs, {**cfg['ovme'], 'objective': objective}).cleaned
    function = {'MCA': mca_clean, 'FBSE-EWT-LPATV': fbse_ewt_lpatv_clean}[method]
    process = lambda block: function(block, fs, cfg['baselines'][method]).cleaned
    if real:
        return overlap_add(values, fs, process, cfg['processing']['baseline_block_s'],
                           cfg['processing']['baseline_overlap_s'])
    return process(values)


def simulated_metrics(x, truth, y, fs, cfg):
    metrics = paper.simulated_metrics(x, truth, y, fs, cfg['evaluation']['common_detector'])
    dbp.load_dependencies()
    settings = dbp.build_settings({**cfg['paper_dbp'], 'sampling_rate_hz': fs}, len(x))
    taper = get_window('hann', round(4 * fs), fftbins=True)
    _, clean_log, error = dbp.analyze_epoch(truth, settings, taper)
    if error:
        raise ValueError(error)
    _, output_log, error = dbp.analyze_epoch(y, settings, taper)
    if error:
        raise ValueError(error)
    difference = abs(output_log - clean_log)
    metrics.update({f'dbp_{name}': float(v) for (name, _, _), v in zip(dbp.BANDS, difference)})
    metrics['dbp_total'] = float(sum(difference))
    return metrics


def real_metrics(x, y, fs, cfg):
    reference = paper.prepare_real_reference(x, fs, cfg['evaluation']['common_detector'])
    metrics = paper.real_metrics(x, y, fs, reference)
    outside = ~paper.event_mask(len(x), reference.events, round(.20 * fs))
    if not np.array_equal(outside, reference.outside_mask):
        metrics['outside_window_pearson_r'] = float('nan')
        metrics['outside_window_nrmse_percent'] = float('nan')
    return metrics


def plot_unit(x, outputs, fs, path, truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = min(len(x), round(10 * fs))
    t = np.arange(count) / fs
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.4), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), METHODS):
        ax.plot(t, x[:count], color='#ff956b', lw=.85, label='Input')
        if truth is not None:
            ax.plot(t, truth[:count], color='#64748b', lw=.95, ls='--', label='Ideal')
        ax.plot(t, outputs[method][:count], color=COLORS[method], lw=.9, label='Output')
        ax.set_title({'AWRLS-strict': 'AWRLS', 'OVME-HHO-REG': 'OVME-REG'}.get(method, method),
                     loc='left', color=COLORS[method], fontsize=12, fontweight='bold', pad=25)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.015), ncol=3, frameon=False, fontsize=8)
        ax.set(xlabel='Time (s)', ylabel='Amplitude (a.u.)', xlim=(0, count / fs))
        ax.grid(axis='y', color='#e5e7eb', lw=.65)
    fig.subplots_adjust(left=.075, right=.988, bottom=.09, top=.9, hspace=.43, wspace=.16)
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper.json')
    parser.add_argument('--scope', choices=('synthetic', 'simulated', 'kaya'), default='synthetic')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Optional epoch/file limit; omitted means all.')
    parser.add_argument('--ovme-sensitivity', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['mode'] != 'strict_single_channel_v1' or set(cfg['datasets']) != {'simulated', 'Kaya2018'}:
        parser.error('Only strict v1 with simulated/Kaya datasets is supported.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', scope=args.scope, synthetic=args.scope == 'synthetic',
                    algorithm_channels=1, auxiliary_channels=0, config=cfg,
                    versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                  pywavelets=pywt.__version__),
                    source_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in ROOT.rglob('*.py')},
                    sources={}, units=[])
    manifest['resolved_awrls'] = asdict(awrls_settings(200, cfg['awrls'])[1])
    manifest_path = output / 'manifest.json'
    rows = []
    try:
        if args.scope == 'synthetic':
            fs = 200.0
            t = np.arange(2000) / fs
            truth = 8 * np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 20 * t)
            x = truth + 90 * np.exp(-.5 * ((t - 4.4) / .12) ** 2)
            jobs = [('synthetic_0000', x, fs, truth)]
        elif args.scope == 'simulated':
            mixed_path = args.data_root / 'blinkeeg/mixed.npy'
            truth_path = args.data_root / 'blinkeeg/clean.npy'
            mixed = np.load(mixed_path, mmap_mode='r', allow_pickle=False)
            reference = np.load(truth_path, mmap_mode='r', allow_pickle=False)
            if mixed.ndim != 2 or mixed.shape != reference.shape:
                raise ValueError('Expected matching epochs x samples numeric arrays.')
            for path in (mixed_path, truth_path):
                manifest['sources'][path.relative_to(args.data_root).as_posix()] = digest(path)
            jobs = [(f'epoch_{i:04d}', np.array(mixed[i], dtype=float), cfg['datasets']['simulated']['fs'],
                     (truth_path, i)) for i in range(min(len(mixed), args.limit or len(mixed)))]
        else:
            settings = cfg['datasets']['Kaya2018']
            files = sorted((args.data_root / settings['directory']).glob(settings['glob']))
            if not files:
                raise FileNotFoundError('No Kaya MAT recordings found.')
            jobs = [(f'{p.stem}_{t}', p, t, None) for p in files[:args.limit] for t in settings['targets']]
            for path in files[:args.limit]:
                manifest['sources'][path.relative_to(args.data_root).as_posix()] = digest(path)
        methods = METHODS + (('OVME-printed-Eq10',) if args.ovme_sensitivity else ())
        for index, (unit, x, fs, truth) in enumerate(jobs):
            real = args.scope == 'kaya'
            audit = {}
            if real:
                settings = cfg['datasets']['Kaya2018']
                raw, source_fs, _ = read_target(x, 'Kaya2018', fs)
                prepared, fs, audit = preprocess_target(raw, source_fs, fs, 'Kaya2018',
                    settings['preprocessing'], settings['target_fs'])
                x = prepared[:round(cfg['real_seconds'] * fs)].copy()
            folder = output / 'units' / unit
            folder.mkdir(parents=True)
            np.save(folder / 'input.npy', x, allow_pickle=False)
            original = x.copy()
            outputs, times = {}, {}
            for method in methods:
                start = time.perf_counter()
                y = clean_one(method, x, fs, cfg, real)
                times[method] = time.perf_counter() - start
                if y.shape != x.shape or not np.isfinite(y).all():
                    raise ValueError(f'Invalid output: {method}')
                np.testing.assert_array_equal(x, original)
                outputs[method] = y
                np.save(folder / f'{method}.npy', y, allow_pickle=False)
            # Ground truth amplitudes are accessed after, and never passed to, inference.
            if isinstance(truth, tuple):
                truth = np.array(np.load(truth[0], mmap_mode='r', allow_pickle=False)[truth[1]])
            for method, y in outputs.items():
                metrics = real_metrics(x, y, fs, cfg) if real else simulated_metrics(x, truth, y, fs, cfg)
                rows.append(dict(unit_id=unit, method=method, runtime_seconds=times[method], **metrics))
            if index == 0:
                plot_unit(x, outputs, fs, output / 'first_unit.png', truth)
            manifest['units'].append(dict(unit_id=unit, samples=len(x), fs=fs, preprocessing=audit,
                artifacts={p.name: digest(p) for p in folder.glob('*.npy')}))
            print(f'{index + 1}/{len(jobs)} {unit}', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / 'metrics_by_unit.csv', index=False)
        numbers = frame.select_dtypes(include='number').columns
        frame.groupby('method')[numbers].agg(['mean', 'std']).to_csv(output / 'summary.csv')
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
