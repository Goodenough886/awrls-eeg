# Third-party notices

## VME

The VME component in `src/blink_repro/methods/ovme_reg.py` and the independently
structured numerical translation in `tests/test_ovme_reg.py` are based on the
author's VME v2 algorithm implementation. Copyright (c) 2020 Mojtaba Nazari.
The full redistribution conditions and disclaimer are reproduced unchanged in
`official_ovme/VME_LICENSE.txt`. No author endorsement is implied.

## Harris hawks optimization

The HHO component port in `src/blink_repro/methods/ovme_reg.py` is based on the
author implementation. Copyright (c) 2019 Ali Asghar Heidari. Its MIT License is
reproduced unchanged in `official_ovme/HHO_LICENSE.txt`. This license does not
license the repository's unrelated original code.

## ITMS exclusion

The local experiment used a Python port of the MATLAB supplemental implementation
of Valderrama et al., Journal of Neural Engineering 15 (2018) 016008,
https://doi.org/10.1088/1741-2552/aa8d95, and its empirical `h0.mat` template.
No explicit redistribution license was found in the locally available package.
The authors' MATLAB files, template, and derivative Python port are therefore
withheld from this release pending clarification of redistribution permission.
Consult the publisher's supplementary material and its applicable terms.
This release cannot independently reproduce the manuscript's ITMS subsection.
An analytic fallback template is not substituted and represented as the same
experiment.

## Dependencies and data

Dependencies are installed separately and remain subject to their own licenses.
No raw or derived EEG datasets, third-party article PDFs, or authors' MATLAB
source files are bundled. Consult the original dataset providers for access
conditions; public dataset access is not permission inferred for redistribution
of every local derivative.
