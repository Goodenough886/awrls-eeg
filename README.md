# AWRLS single-channel EEG blink artifact removal

Source release accompanying the manuscript on adaptive wavelet pseudo-reference
construction and recursive least squares for single-channel EEG blink removal.

Repository: https://github.com/Goodenough886/awrls-eeg

## Scope

This repository provides the strict single-channel AWRLS v1 core, repaired MCA,
the local FBSE-EWT-LPATV reimplementation, and OVME-HHO-REG. It includes the
manuscript metric evaluators, fixed configurations, synthetic tests and a new
portable command-line runner. Algorithms are research implementations, not
clinical software and not certified reproductions of the baseline authors' code.

ITMS source, its authors' MATLAB supplement and the empirical initial template
are not redistributed: an explicit redistribution permission was not found in
the locally available supplement. The Python port depends on that supplement;
we do not describe it as independently licensed original code. See
[third-party notices](THIRD_PARTY_NOTICES.md). This is not a complete public
release of every historical experiment or every baseline component.

No original-code open-source license is granted with this release, at the
copyright holder's request. The code is public for inspection. Third-party
components retain their existing licenses. See [rights](RIGHTS.md).

## Run

Python 3.12 is the validated Python series. Create an isolated environment,
activate it using the standard command for your operating system, then run:

```bash
python -m pip install -r requirements.txt
python run_tests.py
python run_paper.py --scope synthetic --output results/synthetic
```

The synthetic command needs no dataset, uses the fixed algorithm parameters and
creates a four-method plot, per-unit metrics, a summary and a hash manifest.
Synthetic checks are software validation, not new evidence of paper performance.
Choose a new output directory for each run; existing directories are rejected.

The validated scientific environment used NumPy 2.2.6, SciPy 1.15.3 and
PyWavelets 1.8.0. `requirements.txt` records the project's compatible dependency
ranges. Exact cross-platform floating-point equality is not guaranteed.

## Data access and layout

Acquire the source datasets from their original providers under their terms:

- Klados and Bamidis, EEG/EOG dataset: https://doi.org/10.1016/j.dib.2016.06.032
- Kaya et al., EEG dataset: https://doi.org/10.1038/sdata.2018.211

Raw EEG/EOG, derived arrays, local experiment outputs, manuscripts and PDFs are
not included. The local derived simulation arrays lack a complete epoch-to-source
subject mapping. This release does not reconstruct that missing mapping or
claim that the exact 300 paper epochs can be recreated from the source dataset
alone. Independent subject-level simulation inference is not established.

Provide local inputs in this layout:

```text
data/
  blinkeeg/
    mixed.npy
    clean.npy
  Kalya/
    recording.mat
```

`Kalya` intentionally preserves the original local directory spelling. Set a
different directory in a copy of the configuration when needed. Simulation
arrays must have matching `(epochs, samples)` shapes with 200 Hz sampling and
consistent source amplitude units; the paper used 300 epochs of 2,000 samples.
No amplitude conversion to microvolts is inferred by this runner. The Kaya
reader expects MATLAB struct `o` with fields `data`, `chnames` and `sampFreq`.

```bash
python run_paper.py --scope simulated --data-root data --output results/simulated
python run_paper.py --scope kaya --data-root data --output results/kaya
```

`--limit 1` limits simulation to one epoch or Kaya to one file (all configured
target leads processed separately). `--ovme-sensitivity` also evaluates the
literal printed-Equation-10 objective. Ground truth is used only for evaluation,
never as an algorithm input. All generated results remain local and are ignored
by git.

## Single-channel contract

For each Kaya file, Fp1, Fp2 and Fz are three separate target-only calls, not three
simultaneous algorithm inputs. The MAT container is decoded at the I/O boundary;
one labelled column is copied before interpolation, downsampling and filtering.
Other lead values are not consulted for detection, rejection, timing, wavelet
selection, reference construction, RLS fitting or subtraction. Recorded hardware
reference is retained; this is not a claim of reference-free acquisition.

`reference_mode` is `none`; multichannel arrays and auxiliary-channel settings
are rejected by the public runner. Tests mutate, remove and reorder unselected
lead values while requiring identical target output. Cho is deprecated and is
not included. Historical multichannel AWRLS adapters and the experimental
guarded-v2 algorithm are not part of this release.

## Manuscript protocol and limitations

- AWRLS uses the original 22-wavelet v1 core, minimum similarity 0.85, RLS order
  12, forgetting factor 0.995, delta 0.01 and two passes. The same core is used
  for simulation and real data. No parameter optimization is performed here.
- Real preprocessing selects the target first, then applies the frozen
  preprocessing configuration to its complete recording. The main OVME
  comparison subsequently uses the first 30 seconds. Do not crop the raw file
  before preprocessing or compare these metrics against whole-record results.
- The paper's real pilot has 73 files, 219 target calls and 13 subjects. Target
  calls from a file/subject are not independent observations. The new runner
  processes the supplied files; it does not imply that arbitrary input
  directories reproduce that exact cohort.
- MCA retains the corrected unnormalized SWT dictionaries, db4 level 4,
  14 threshold levels, five inner updates and final threshold 4 sigma.
  `mca_legacy.py` exists only for the normalized-SWT regression test; it is not
  a selectable paper baseline.
- OVME-HHO-REG uses HHO population 30, 20 iterations, alpha 1000-10000,
  initialization frequency 0.5-7.5 Hz, VME maximum 300 states, tolerance 1e-7,
  tau 0, seed 20260926, 3-second blocks with 1-second overlap and positive peaks.
  The primary fitness is `mean(mode**2)` (stated MSE); `mean(mode)` is retained
  as a sensitivity interpretation of printed Equation 10. Neither is selected
  after observing relative AWRLS performance. VME/HHO are component ports, not
  an official OVME-REG implementation. See [reproduction notes](docs/OVME_REG_REPRODUCTION.md).
- `evaluation/metric_core.py` is the paper evaluator, not the older engineering
  evaluator. `evaluation/compute_dbp.py` supplies four-second Hann windows,
  50% overlap, nfft 800 at 200 Hz and the five paper bands. Real low-frequency
  energy reduction is an input-output proxy: it can reflect attenuation of
  physiological slow activity and is not a ground-truth blink removal rate.
  Invalid whole-record outside-window fallbacks are reported as missing values.
- `src/blink_repro/review_statistics.py` includes file/subject aggregation and
  paired effect sizes. The runner summary is descriptive, not an independent
  subject-level significance test. Missing simulation subject mapping is not
  replaced by an artificial random subject grouping.
- Measured per-method wall time includes each method's processing call but
  excludes data loading, preprocessing, scoring and plotting. It is not a
  real-time or wearable latency guarantee. The published processing uses
  acausal filtering and contextual wavelet analysis.

## Source provenance

`SOURCE_PROVENANCE.json` records source SHA-256 values and public-file hashes.
Core AWRLS, MCA and FBSE files were checked against the frozen strict-run
manifest. OVME and its tests/notices come from the frozen OVME code snapshot.
The portable runner is new packaging code. The strict reader excludes the
deprecated Cho branch, the loader excludes historical multichannel adapters,
and the DBP tool's two private default paths were made relative. Those changes
do not alter the retained algorithm/evaluator calculations.

The historical Windows orchestration scripts, local cache import paths and
Word-editing automation are not distributed. No historical data, checkpoints,
tokens, private paths or git history are uploaded. Exact paper reproduction
still requires the same inputs; source availability does not remove the data
and licensing limitations described above. Cite the repository with the exact
commit identifier used, not only the moving `main` branch.
