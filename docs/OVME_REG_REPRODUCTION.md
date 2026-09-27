# OVME REG reproduction notes

Reference: OVME-REG: Harris hawks optimization algorithm based optimized
variational mode extraction for eye blink artifact removal from EEG signal.
Medical & Biological Engineering & Computing (2024), 62:955-972.
https://doi.org/10.1007/s11517-023-02976-y

This is a local single-input reimplementation informed by the article and the
author VME/HHO components. It is not an official OVME-REG repository, nor has
agreement with the paper authors' complete pipeline been established.

The article describes mean-squared extraction energy but prints an Equation 10
expression without the square. Both interpretations are implemented explicitly:
`stated_mse` is the primary prespecified objective; `printed_mean` is sensitivity
analysis. Clean EEG and concurrent EOG are never used to optimize the extractor.

After VME extraction, the method detects positive peaks using the paper-style
mode threshold, creates windows extending 0.125 s before and 0.375 s after each
peak, and regresses the extracted mode against the contaminated target within
merged windows. Processing of longer inputs uses 3 s blocks and 1 s overlap,
with averaged overlapping corrections. These numerical choices are recorded
in `configs/paper.json`; dataset-specific results from the original publication
are not treated as directly comparable to this different local protocol.

Tests compare the cached VME implementation against a full-spectrum formula
translation, check the published synthetic VME example, HHO bounds and
repeatability, threshold/window rules, regression and block coverage. They are
not native MATLAB validation or confirmation of the full published OVME-REG
performance. No complete end-to-end native MATLAB validation is claimed.
