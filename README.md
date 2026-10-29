# AWRLS: single-channel EEG blink artifact removal

Code, frozen learned weights, evaluation scripts and result tables for the AWRLS study.
This release supersedes the earlier v1-only default branch; the previous commits remain in Git history.

Repository: https://github.com/Goodenough886/awrls-eeg

## Methods

| Paper name | Implementation / result label | Training |
|---|---|---|
| AWRLS | `awrls2/`, `V2` | None |
| AWRLS-Ddict | `awrls3/`, `V3_dictionary` | Frozen shape dictionary |
| AWRLS-D | `awrls3/`, `V3_full` | Frozen shape dictionary |
| AWRLS-H | `V3_HYBRID` | Dictionary + gated residual TCN |
| Equal-capacity TCN control | `TCN_CONTROL` | Same TCN backbone; proposal channel zeroed |

The primary neural seed is 20261028; the two additional seeds are 20261029 and 20261030.
All seven model files and the numerical core are unchanged from the frozen experiment.
Processing is offline, and ICA is an evaluation reference, not an algorithm input.
Publicly visible code is not an implied open-source license: see [RIGHTS.md](RIGHTS.md).

## Install and run

Python 3.12 was used for validation. From this repository:

```bash
python -m pip install -e .
python -m pip install torch==2.8.0
python verify_release.py
python -m unittest discover -s tests -v
python -m unittest discover -s legacy_tests -v
python run_signal.py input.npy cleaned.npy --fs 200 --method awrls
python run_signal.py input.npy cleaned_hybrid.npy --fs 200 --method hybrid --device cpu
```

Install a platform-appropriate CUDA PyTorch build separately if GPU inference is needed.
Input must be a finite, one-dimensional NPY array containing one target channel, with the true sample rate.
`run_signal.py` also accepts `dictionary`, `awrls-d` and `tcn`. Existing output files are rejected.
The supplied weights do not require clean EEG, EOG, ICA or test-set adaptation at inference.

## Published results

`published_results/` contains all rows of the final internal, public and Kaya tables,
paired statistics, the original v2 ablation and sensitivity tables, and frozen original tables.
Large tables are split into numbered `.csv.gz` parts with repeated headers. Their order,
row counts and original table hashes are recorded in `TABLE_INDEX.json`.

```bash
python restore_tables.py
```

This reconstructs tables under `local_runs/restored_tables/`, leaving the published files untouched.
The `frozen_original/` subdirectory contains the historical inputs to the comparison scripts.
Copy its `test_models.csv` into `results/` when using the frozen baseline input check.

- Public test: 554 source pairs, six physical-power SNR levels and clean input, 13 result labels.
- Complete Kaya collection: 75 sessions, 13 subjects, Fp1/Fp2/Fz, two input recipes,
  12 processed methods, 5,400 full-length outputs. Original 73-session tables are retained separately.
- The missing two official sessions were restored and all 75 files matched the official MD5 identities.
- Internal 300-epoch results are retrospective development evidence; 900 new clean-input baseline outputs were added.
- 13,992 newly generated waveform files were verified locally. Waveform archives are not hosted here.

See [results and limitations](docs/RESULTS_CN.md). The methods show metric-dependent tradeoffs;
this release does not claim that AWRLS dominates every comparator.

## Data and reproducibility boundaries

Obtain data from the providers under their applicable terms:

- [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet)
- [Klados and Bamidis](https://doi.org/10.1016/j.dib.2016.06.032)
- [Kaya et al.](https://doi.org/10.1038/sdata.2018.211)

Set `AWRLS_PUBLIC_DATA` to the directory containing `EEG_all_epochs.npy` and `EOG_all_epochs.npy`.
Set `AWRLS_INTERNAL_DATA` to the directory containing matching `mixed.npy` and `clean.npy` arrays.
Fallback locations are `data/EEGdenoiseNet` and `data/blinkeeg`. The frozen source split includes
indices and data hashes; only its machine-specific directory fields were made portable.
No raw or derived EEG waveform datasets are redistributed. The local 300-by-2000 arrays
lack a verified epoch-to-original-subject mapping and a recovered generation script.
The exact 300 epochs therefore cannot be reconstructed from this repository alone.
Source-disjoint public evaluation is not evidence of subject-disjoint evaluation.
The ICA reference is a silver standard, not physiological ground truth.

`scripts/` contains the frozen training/evaluation implementations and supplemental runners.
Run experiments in a separate checkout because historical scripts write to `results/` and training
can overwrite weights. For example, on a Windows checkout with the original data:

```bash
python scripts/evaluate_models.py --split test --jobs 4 --device cpu
python scripts/run_supplement_full.py --part public --tag public_reproduction --wj-src baseline_source/src --jobs 4
```

Frozen manifests use Windows path separators; the original full experiment runners were validated
on Windows. The portable inference entry point and table reader do not require that original layout.
Some archived full-Kaya/supplement scripts also require the original staging layout and previous
v2 outputs; they are retained as execution evidence, not advertised as a one-command fresh reproduction.
`historical_v2/` supplies the preparation and v2 experiment scripts. ICA dependencies are in
`requirements-ica.txt`. MCA, FBSE-EWT-LPATV and OVME-REG code is in `baseline_source/`.
The optional Windows FP64 VME kernels include source, DLL and numerical validation records;
the default supplemental runner uses the NumPy backend. Recorded concurrent/resumed timings
must not be interpreted as standalone algorithm latency.

ITMS result rows are retained, but its derivative code and empirical template are not distributed;
see [third-party notices](THIRD_PARTY_NOTICES.md). Archive DOI registration has not been claimed.
For a stable citation, record the exact Git commit linked in the revised manuscript.
