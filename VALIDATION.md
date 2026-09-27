# Release validation

Validated locally on 2026-09-27 with Python 3.12, NumPy 2.2.6, SciPy 1.15.3 and
PyWavelets 1.8.0.

- 30 synthetic unit tests passed, including MCA regression, VME numerical
  formula agreement, HHO contracts, single-channel rejection, mutation of
  auxiliary leads and identity metric checks.
- The data-free synthetic CLI completed with all four fixed-parameter methods,
  the manuscript metrics, saved outputs, a hash manifest and the comparison plot.
- A locally available frozen simulation epoch and the manuscript's real 30 s
  example were checked against archived waveforms for all four main methods and
  both OVME objective interpretations (five outputs per input, ten in total).
  Waveforms were bit-for-bit identical. Matching stored manuscript metrics passed
  with relative and absolute tolerances of 1e-12 (23 simulation metrics and six
  real metrics per output).
- For the real example, loading and preprocessing its target from the raw MAT
  produced a first-30-second input bit-for-bit identical to the frozen input.

Frozen EEG, generated waveforms, patient/file identifiers and local paths are
not included here. These are representative regression checks, not a rerun of
all 300 simulation epochs or all 219 real target calls. They do not validate
ITMS redistribution, native MATLAB equivalence, external generalization or
clinical use. Runtime from this check is not substituted into the manuscript's
timing results.

Run `python run_tests.py` and `python verify_release.py` after obtaining the
source. Hashes in `SHA256SUMS.json` detect changes against this source inventory;
they are not a cryptographic signature or an independent certification.
