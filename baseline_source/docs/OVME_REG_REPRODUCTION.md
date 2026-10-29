# OVME-HHO-REG reproduction contract

## Status and scope

This is an independently implemented, strict-single-channel reproduction of the
published procedure, with explicitly unresolved ambiguities. It is not the
authors' OVME implementation and does not reproduce their original tables.
The source paper is Silpa and Hota, Medical & Biological Engineering & Computing
62 (2024), 955-972, DOI 10.1007/s11517-023-02976-y.
The user supplied `C:/Users/13665/Downloads/s11517-023-02976-y.pdf`.
Pages 958-960 (PDF pages 4-6) were rendered and inspected, including Algorithm 1.
The experiment manifest records the full PDF SHA-256.

Primary implementation: `src/blink_repro/methods/ovme_reg.py`.
Settings: `config.ovme_reg.json`. Entrypoint: `run_ovme_reg.py`.
No frozen AWRLS, baseline, reader, preprocessing or metric code is changed.
No Cho data are used. ITMS is retained in supplementary comparisons.
Proposed main comparison: AWRLS-guarded-v2, MCA, FBSE-EWT-LPATV, OVME-HHO-REG.
Do not replace the manuscript table until the ambiguity and validation results
have been assessed; an executable implementation is not proof of source fidelity.

## Source-to-code mapping

| Source | Implementation |
| --- | --- |
| Section 3.1, Eq. 11, p.959 | alpha in [1000,10000]; initial frequency in [0.5,7.5] Hz |
| Section 4.4, p.966 | HHO population 30 and 20 iterations |
| Section 3.2, Eq.12, p.959 | median(abs(mode))/0.6745 * sqrt(2*log(N)) |
| Algorithm 1, p.960 | strictly positive local maxima; no minimum peak spacing added |
| Section 3.2, p.959 | peak minus 0.125s through peak plus 0.375s, inclusive endpoints |
| Section 2.3, Eq.8, p.958 | centered EEG-mode covariance / centered mode sum of squares |
| Eq.9, p.958 | EEG minus beta times the uncentered extracted mode within each window |
| Section 4.1, p.961 | authors use three-second single-channel epochs |
| VME author toolbox v2, October 2020 | VMEWorkspace: mirrored FFT, quartic update, centroid and reconstruction |
| HHO author MATLAB implementation | hho_minimize: exploration, four exploitation cases, Levy exponent 1.5 |

VME primary source:
https://www.mathworks.com/matlabcentral/fileexchange/76003-variational-mode-extraction-vme-m

Public v2 package retrieved on 2026-09-26:
https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/9ef79c65-69bb-4f22-ab89-c38fb67b92dd/12f669a4-624c-47c6-8720-85b5c205fc0a/packages/zip

HHO primary source:
https://github.com/aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications

Unmodified downloaded files and licenses are archived in `official_ovme/` and
hashed into the experiment snapshot. The VME code is by Mojtaba Nazari and
S. Mahmoud Sakhaei; redistribution notice is in `VME_LICENSE.txt`.
The HHO code is Copyright (c) 2019 Ali Asghar Heidari, MIT licensed.
These are component sources, not a claim of an official OVME-REG repository.

## Unresolved Equation 10

The original PDF, not merely extracted text, prints an unsquared difference,
although the surrounding text calls the objective mean square error (MSE).
It defines residual f_hat = f - d, using the same input f. Consequently:

- `stated_mse` (primary): mean((f - f_hat)**2) = mean(d**2).
- `printed_mean` (sensitivity): mean(f - f_hat) = mean(d).

Both operate on the target EEG only. Neither uses clean reference EEG, an EOG
channel, labels, artifact masks or outputs from any other method.
Minimizing mode energy may favor a weak or non-blink mode; minimizing a signed
mean can be cancellation-sensitive. Neither objective guarantees blink recovery.
The primary interpretation was fixed from the stated MSE definition, not selected
for favorable AWRLS comparison. Both are reported, and no best-of-two result is
substituted. Authors' clarification is needed for a fully source-verified claim.
No supervised clean-vs-cleaned objective is silently substituted.

## Declared numerical and integration choices

1. VME uses the author's downloadable v2, not an assumed VMD implementation.
   The paper does not identify the toolbox revision. V2 uses kernel
   K=alpha**2*(frequency-omega)**4 and denominator (1+K)*(1+2*K).
   This modified code is not silently replaced by a differently derived VME
   equation or by PySDKit's current port.
2. VME tau=0, tolerance=1e-7 and maximum 300 states (299 updates) follow the
   author's demonstration/code, not a parameter table in OVME-REG.
   The MATLAB stopping expression `delta*conj(delta)'` equals sum(delta**2),
   not sum(abs(delta)**2). The port preserves it literally and tests it against
   an independent full-spectrum translation. This is NOT a MATLAB runtime test.
3. Odd-length inputs are edge-padded once and restored to their original length.
   Constant blocks skip optimization and remain exactly unchanged. A zero-power
   centroid retains its previous frequency instead of dividing by zero.
4. Frequency in Hz is divided by sampling rate inside VME; no extra 2*pi factor.
   Bounds apply to INITIAL frequency. The evolving centroid is not forcibly
   clipped to [0.5,7.5] Hz, matching the VME algorithm.
5. HHO uses a deterministic NumPy RNG, not MATLAB's random sequence. All candidate
   evaluations, including rapid dives, respect Eq.11 bounds. The original HHO
   demonstration only clamps at the population loop. Bounds and RNG differences
   are explicit portability choices. Exact repeated parameter evaluations are
   memoized; the objective and iteration budget are unchanged.
6. Positive peaks only are the paper-faithful default. A negative-polarity blink
   may be missed. `absolute` is exposed as an explicitly different adaptation,
   not enabled in the primary experiment or auto-selected after observing results.
7. Overlapping event windows within a block are merged and regressed once to
   avoid repeated subtraction. Edge windows are clipped; sample positions use
   nearest-integer rounding. No gain clipping, taper, denoising gate or AWRLS
   detector is introduced.
8. For records longer than three seconds, three-second blocks with one-second
   overlap are used. Their CORRECTIONS are uniformly averaged; the last block
   is end-aligned. The paper did not specify this long-record integration rule.
   Samples outside the union of detected windows are exactly unchanged.
9. No extra amplitude normalization is applied. Original source units are retained;
   their conversion to microvolts is not assumed. VME's absolute stopping criterion
   may make amplitude scale relevant.

## Data and evaluation contract

- 300 full, unchanged simulated paired epochs; the existing 150/150 development
  and retrospective internal holdout split is reused without retuning.
- Kaya: 73 source files, Fp1/Fp2/Fz processed as 219 separate one-vector calls.
  Decoding the original multichannel container precedes target-only preprocessing
  in the verified parent; this runner uses those hash-verified target inputs.
- Default validation scope is the first fixed 30 seconds of every Kaya target.
  These are PILOT excerpts, not full-record results and not selected blink windows.
  All comparators are rerun on the identical excerpt. Old full-record outputs
  are never merely cropped and presented as a fair excerpt comparison.
- `--real-seconds 0` requests full-record processing in a NEW output directory.
  Resume verifies fingerprints and saved per-unit artifact hashes.
- Frozen simulated baseline outputs are hash-verified and metrics recomputed.
  Clean reference data are loaded only after inference; both OVME variants also
  receive separate clean-input preservation controls.
- No new p-value claims: retain descriptive paired holdout effects and equal-file,
  equal-subject Kaya effects with pointwise subject bootstrap intervals.
- Candidate-window low-frequency attenuation is not an artifact removal rate.
  Supplemental reference-free metrics preserve undefined no-event values as NaN.
- Per-unit timings obtained under concurrent workers are diagnostic wall times,
  not a serial benchmark or evidence of causal real-time operation.
- The original paper evaluated 30 manually constructed three-second signals on
  different datasets. Its CC/RRMSE/SSIM values are not directly comparable with
  this project's Klados/Kaya metrics, nor guaranteed reproduction targets.

## Commands

Run `run_ovme_reg.ps1` for the fixed 300-epoch / 219-excerpt validation.
Run `run_ovme_reg.ps1 -FullReal -Output <new-directory>` for full real records.
Use the same command/output to resume an interrupted unit-level run.
An interrupted unit is recomputed; completed units are verified and reused.
The experiment writes no manuscript changes and never deletes previous results.
