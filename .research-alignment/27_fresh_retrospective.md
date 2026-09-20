# Four retrospective fresh-annulus probes

Saved zero-based states 5,11,20,27 are fresh-preconditioner states (report rows
6,12,21,28). Restore each recorded bank by IDs and rebuild its exact current C
and Arnoldi model. Check old seed step and predictions against the archived
trajectory before running the expensive fresh sweep. Use the recorded accepted
radius (equal to initial radius in all four selected states).

Change only old versus fresh annulus direction; no continuation, trigger tuning,
parameter release or bank maintenance. Record Gold/Gfresh, state/response cosine,
fresh descent slope, gradient stencil comparison, and all three full-map controls.
At least3/4 with Gold<1.1 and Gfresh>1.5 is the prespecified model decision gate;
confirmed recovery additionally needs valid true candidates. Direction cosines
and slopes support interpretation; model gain alone is not rotation evidence.

Use the existing Cauchy lower bound, radius, feasible-state, three-scale model
agreement<5%, positive actual decrease and rho>0.1 checks. Preserve all source
bytes, inputs, compressed models and steps. Independent replay follows.
Crossing the gate supports beginning adjoint design, not claiming a measured
adjoint speedup or certifying physical root/stability. Four sampled states do not
establish a universal reseed schedule. No full adjoint implementation is included
in this controlled experiment.

## Recorded outcome
| Saved state | Gold | Gfresh | cos_x | cos_J | Fresh descent |
|---|---|---|---|---|---|
| 5 | 1.00005303 | 1.28422752 | -9.512757e-05 | -0.2482268 | 8.746542e-05 |
| 11 | 1.00278560 | 1.84211947 | -3.943598e-05 | -0.2498778 | 7.418687e-05 |
| 20 | 1.00121208 | 3.01138017 | 8.537639e-06 | -0.1833764 | 9.501684e-05 |
| 27 | 1.00006233 | 8.94955392 | -1.548184e-05 | -0.1648918 | 9.761863e-05 |

Prespecified threshold met: 3/4; validated recovery: 3/4.
Gate to begin adjoint work (at least 3/4): True.
All comparisons concern one frozen model per state; these do not measure an
adjoint speedup, certify a physical root or establish a reseeding period.
