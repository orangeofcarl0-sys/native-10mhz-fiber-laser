# Current-state active-history audit

Question: does selecting saved fresh directions by current trust-model prediction improve over rolling last three at the S3 endpoint?

Freeze physical equations, 20.42 m cavity, CNT→OC, pump 27.5 mW, OC 0.8, GDD +0.2 ps², Psat 40 W, grid, normalization and gauge. Start at norm 0.0011883945817279438. Recompute all seven current responses; use one common C/Arnoldi model. Compare H3/G2/G3/G4/G3+A at radii 0.00156, 0.003125, 0.00625. No continuation or parameter release.

Greedy chooses predicted merit decrease only. Force four additions for the audit; separately report the count retained before first marginal improvement below 5%. No slope/cosine screening. Record input and response singular spectra independently. QR compression preserves actual state norm, not coefficient norm; test against the full model.

Accept only positive full-map decrease, Cauchy lower bound, feasible state, three-scale direct model agreement below 5%, and rho above 0.1. Independently replay all candidates. A 95% finite-bank model saturation is not proof of physical curvature dimension. An oracle advantage above 25% is state-specific evidence, not proof of periodic reseeding. No root/stability claim, no attribution of past unequal-start rate ratios as a controlled speedup.


## Observed evidence
- All radii select source outer steps 18,28,14,1. G3/G4 actual ratio 0.32--0.34,
  below 0.95; the proposed early low-dimensional saturation is not supported.
- Relative smallest singular values: D 0.29484, JD 0.02232; numerical ranks 7 at 1%.
- G3+A/G3 actual ratio 11.9--12.5, about four times G4; current oracle usefulness
  is supported, periodic timing and long-run acceleration remain untested.
- Fifteen full-map candidates pass. Independent residual/actual replay differences
  are zero. Maximum candidate model discrepancy 8.49e-7, below 5% threshold.
- 82 CPU tests pass, including full/compressed metric equivalence and uphill
  selection. The current run contains no continuation or physical parameter release.
