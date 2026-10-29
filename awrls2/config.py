"""Configuration of AWRLS-2 and the named ablation / baseline variants.

Every design decision of the pipeline is a field of ``Config`` so that an
ablation study is a list of ``Config`` objects produced from the same base.
``V1_REPRODUCTION`` reproduces the published AWRLS v1 behaviour inside the new
code base (same detector rules, periodised DWT, self pseudo-reference, tapped
delay line RLS with two passes, unit gain, Tukey taper).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Tuple

DEFAULT_WAVELET_CANDIDATES: Tuple[str, ...] = (
    "haar",
    "db2", "db3", "db4", "db5", "db6", "db8",
    "sym4", "sym5", "sym6", "sym8",
    "coif1", "coif2", "coif3", "coif5",
    "bior2.2", "bior3.3", "bior3.5", "bior4.4", "bior6.8",
    "rbio4.4", "dmey",
)

# A frequency-matched compact candidate set (speed ablation): one representative
# per family, lengths 4..24 taps.
COMPACT_WAVELET_CANDIDATES: Tuple[str, ...] = ("db2", "db4", "db6", "sym5", "sym8", "coif2", "bior3.5", "dmey")


@dataclass
class Config:
    fs: float = 200.0

    # ------------------------------------------------------------------
    # Stage 1: candidate localisation
    # ------------------------------------------------------------------
    lowpass_hz: float = 12.0
    lowpass_order: int = 4
    energy_window_s: float = 0.08
    min_peak_distance_s: float = 0.45
    peak_refine_window_s: float = 0.12
    min_artifact_duration_s: float = 0.18
    max_artifact_duration_s: float = 1.50
    boundary_mode: str = "envelope_zc"     # "zero_crossing" (v1) | "envelope" | "envelope_zc" | "union"
    boundary_kappa: float = 2.0            # envelope threshold = max(kappa*sigma_bg, boundary_peak_fraction*|peak|)
    boundary_peak_fraction: float = 0.08
    boundary_pad_s: float = 0.08
    boundary_max_left_s: float = 0.60
    boundary_max_right_s: float = 1.10
    surrogate_calibration: bool = True     # phase-randomised surrogates of the clipped background calibrate the thresholds
    surrogate_count: int = 24
    surrogate_clip_sigma: float = 3.0
    surrogate_quantile: float = 0.95
    second_pass: bool = True               # matched-filter rescue of weaker blinks
    second_pass_ncc: float = 0.80
    second_pass_amp_z: float = 3.0
    second_pass_min_seed_events: int = 2

    # ------------------------------------------------------------------
    # Stage 2: adaptive wavelet pseudo-reference
    # ------------------------------------------------------------------
    transform: str = "swt"                 # "dwt" (v1, periodisation) | "swt" (shift invariant)
    wavelet_candidates: Tuple[str, ...] = DEFAULT_WAVELET_CANDIDATES
    wavelet_target_hz: float = 12.0
    max_wavelet_level: int = 6
    minimum_similarity: float = 0.85
    score_mode: str = "full"               # "full" (P*F_E*F_f) | "similarity" | "no_energy" | "no_frequency"
    fixed_wavelet: str | None = None       # e.g. "db4" -> disables adaptive mother-wavelet selection
    fixed_level: int | None = None         # e.g. 4    -> disables adaptive level selection

    # ------------------------------------------------------------------
    # Stage 3: regressor construction (reference purification)
    # ------------------------------------------------------------------
    regressor: str = "template"            # "self_tdl" (v1) | "self" | "template"
    include_self_reference: bool = True    # add the event's own wavelet reference as an extra column
    template_min_events: int = 3           # fewer accepted events -> fall back to "self"
    template_rank: int = 2                 # rank of the leave-one-out blink-shape subspace
    derivative_basis: bool = True          # add d/dn of the leading component (first-order shift/width correction)
    shift_search_s: float = 0.05
    width_scales: Tuple[float, ...] = (0.85, 1.0, 1.18)
    nuisance_trend: bool = True            # unpenalised polynomial columns absorb the local baseline (not subtracted)
    nuisance_order: int = 1                # polynomial order of the nuisance columns (1 = [1, n])

    # ------------------------------------------------------------------
    # Stage 4: estimator
    # ------------------------------------------------------------------
    estimator: str = "ls"                  # "rls" (v1) | "ls" (band-limited ridge LS, proposed) | "gls"
    fit_lowpass_hz: float | None = 12.0    # fit band: "ls" low-passes d and R; "gls" weights only f <= this
    gls_floor: float = 0.05                # spectral floor (fraction of max in-band S_bg) for 1/S_bg weights
    ridge: float = 1e-3                    # relative ridge penalty on blink-basis columns
    ridge_adaptive: bool = False           # penalise the shape columns by kappa / rho, rho = in-window LF energy / background LF energy
    ridge_adaptive_kappa: float = 1.0      # strength of the SNR-adaptive penalty (0 disables)
    rls_order: int = 12                    # taps of the self tapped-delay line (v1)
    rls_forgetting_factor: float = 0.995
    rls_delta: float = 0.01
    rls_passes: int = 2

    # ------------------------------------------------------------------
    # Stage 5: gain, taper, projection and guards
    # ------------------------------------------------------------------
    gain_mode: str = "unit"                # "unit" | "bounded_ls" | "energy_match"
    taper_alpha: float = 0.12
    correction_lowpass_hz: float | None = None   # low-pass the *correction* (not the EEG); None = off
    guard: bool = True                     # reject corrections that do not reduce in-window LF energy
    guard_hf_fraction: float = 0.10        # max fraction of correction energy allowed above lowpass_hz

    def with_updates(self, **updates: Any) -> "Config":
        unknown = set(updates) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"Unknown Config fields: {sorted(unknown)}")
        return replace(self, **updates)

    def validate(self) -> None:
        if not 0 < self.lowpass_hz < self.fs / 2:
            raise ValueError("lowpass_hz must lie in (0, fs/2)")
        if not 0 < self.wavelet_target_hz < self.fs / 2:
            raise ValueError("wavelet_target_hz must lie in (0, fs/2)")
        if not 0 <= self.minimum_similarity <= 1:
            raise ValueError("minimum_similarity must lie in [0, 1]")
        if self.boundary_mode not in ("zero_crossing", "envelope", "envelope_zc", "union"):
            raise ValueError("boundary_mode")
        if self.transform not in ("dwt", "swt"):
            raise ValueError("transform")
        if self.score_mode not in ("full", "similarity", "no_energy", "no_frequency"):
            raise ValueError("score_mode")
        if self.regressor not in ("self_tdl", "self", "template"):
            raise ValueError("regressor")
        if self.estimator not in ("rls", "ls", "gls"):
            raise ValueError("estimator")
        if self.gain_mode not in ("unit", "bounded_ls", "energy_match"):
            raise ValueError("gain_mode")
        if not 0 < self.rls_forgetting_factor <= 1 or self.rls_delta <= 0:
            raise ValueError("RLS parameters")
        if not 0 < self.min_artifact_duration_s <= self.max_artifact_duration_s:
            raise ValueError("artifact duration limits")


# ----------------------------------------------------------------------------
# Named variants
# ----------------------------------------------------------------------------
V1_REPRODUCTION: Dict[str, Any] = dict(
    boundary_mode="zero_crossing", second_pass=False, surrogate_calibration=False, ridge_adaptive=False,
    transform="dwt", score_mode="full",
    regressor="self_tdl", derivative_basis=False, nuisance_trend=False,
    estimator="rls", gain_mode="unit", correction_lowpass_hz=None, guard=False,
)

PROPOSED: Dict[str, Any] = dict()  # == Config() defaults


def ablation_variants() -> Dict[str, Dict[str, Any]]:
    """Ordered dictionary name -> Config overrides.

    The first entry reproduces v1; the last is the full proposed method.
    Each 'A' row removes exactly one component from the proposed method, each
    'B' row adds exactly one component to v1.  'C' rows are alternative design
    choices.  'D' rows are the baseline-style simplifications reviewers ask for.
    """
    v = V1_REPRODUCTION
    out: Dict[str, Dict[str, Any]] = {}
    out["V1_reproduction"] = dict(v)
    # --- B: add one component to v1 -------------------------------------
    out["B1_v1+envelope_zc_boundary"] = dict(v, boundary_mode="envelope_zc")
    out["B2_v1+second_pass"] = dict(v, second_pass=True, boundary_mode="envelope_zc")
    out["B2b_v1+surrogate_calibration"] = dict(v, surrogate_calibration=True)
    out["B3_v1+swt"] = dict(v, transform="swt")
    out["B4_v1+gls"] = dict(v, estimator="gls", regressor="self", nuisance_trend=True)
    out["B4b_v1+ls_in_band"] = dict(v, estimator="ls", regressor="self", nuisance_trend=True)
    out["B5_v1+template"] = dict(v, regressor="template", derivative_basis=True, nuisance_trend=True, estimator="ls")
    out["B6_v1+guard"] = dict(v, guard=True)
    out["B7_v1+energy_gain"] = dict(v, gain_mode="energy_match")
    # --- A: remove one component from the proposed method ------------------
    out["A0_prop-no_surrogate_calibration"] = dict(surrogate_calibration=False)
    out["A1_prop-no_second_pass"] = dict(second_pass=False)
    out["A2_prop-zero_crossing_boundary"] = dict(boundary_mode="zero_crossing")
    out["A2b_prop-envelope_without_zc_snap"] = dict(boundary_mode="envelope")
    out["A3_prop-dwt_instead_of_swt"] = dict(transform="dwt")
    out["A4_prop-self_reference_only"] = dict(regressor="self", derivative_basis=False)
    out["A5_prop-no_derivative_basis"] = dict(derivative_basis=False)
    out["A6_prop-gls_instead_of_ls"] = dict(estimator="gls")
    out["A6b_prop-ls_fullband"] = dict(fit_lowpass_hz=None)
    out["A6c_prop-self+derivative_only"] = dict(regressor="self")
    out["A7_prop-rls_instead_of_ls"] = dict(estimator="rls")
    out["A8_prop-no_guard"] = dict(guard=False)
    out["A9_prop-no_nuisance_trend"] = dict(nuisance_trend=False)
    out["A10_prop+adaptive_ridge"] = dict(ridge_adaptive=True)
    # --- C: scoring / candidate-set alternatives -----------------------------
    out["C1_score_similarity_only"] = dict(score_mode="similarity")
    out["C2_score_no_energy"] = dict(score_mode="no_energy")
    out["C3_score_no_frequency"] = dict(score_mode="no_frequency")
    out["C4_fixed_db4_L4"] = dict(fixed_wavelet="db4", fixed_level=4)
    out["C5_fixed_sym8_L3"] = dict(fixed_wavelet="sym8", fixed_level=3)
    out["C6_compact_candidates"] = dict(wavelet_candidates=COMPACT_WAVELET_CANDIDATES)
    out["C7_template_rank1"] = dict(template_rank=1)
    out["C8_template_rank3"] = dict(template_rank=3)
    out["C9_template_without_self_column"] = dict(include_self_reference=False)
    out["C10_gain_bounded_ls"] = dict(gain_mode="bounded_ls")
    out["C11_correction_lowpass"] = dict(correction_lowpass_hz=12.0)
    # --- D: simplified / online-compatible ---------------------------------
    out["D1_online_rls_one_pass"] = dict(regressor="self_tdl", estimator="rls", rls_passes=1, derivative_basis=False)
    out["D2_direct_subtraction_of_reference"] = dict(regressor="self", estimator="ls", ridge=0.0, derivative_basis=False, nuisance_trend=False, guard=False)
    out["PROPOSED"] = dict(PROPOSED)
    return out


def sensitivity_grid() -> Dict[str, list]:
    """One-at-a-time parameter sweeps around the proposed configuration."""
    return {
        "minimum_similarity": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "wavelet_target_hz": [6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        "gls_floor": [0.01, 0.02, 0.05, 0.1, 0.3, 1.0],
        "fit_lowpass_hz": [8.0, 10.0, 12.0, 15.0, 20.0],
        "ridge": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        "template_min_events": [2, 3, 4, 6],
        "taper_alpha": [0.0, 0.06, 0.12, 0.25, 0.5],
        "second_pass_ncc": [0.70, 0.75, 0.80, 0.85, 0.90],
        "surrogate_quantile": [0.90, 0.95, 0.99, 0.999],
        "surrogate_clip_sigma": [3.0, 4.0, 5.0, 7.0],
        "ridge_adaptive_kappa": [0.0, 0.3, 1.0, 3.0, 10.0],
        "boundary_kappa": [1.0, 1.5, 2.0, 2.5, 3.0],
        "boundary_pad_s": [0.0, 0.04, 0.08, 0.12],
        "nuisance_order": [0, 1, 2, 3],
        "template_rank": [1, 2, 3],
        "lowpass_hz": [8.0, 10.0, 12.0, 15.0],
    }
