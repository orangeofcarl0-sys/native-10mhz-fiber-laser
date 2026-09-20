# General discrete adjoint validation contract

Question: can J_packed^T v faithfully replace enumerated current gradients, at
lower measured cost, without changing the forward physics or trust solver?
Baseline commit 5d72308; fixed four saved states 5,11,20,27 and physical settings.

Build A shell (scale, phase/time, population, gauges), B linear maps, C Kerr,
D CNT recurrence, E EDF optical/rate/pump reverse. CPU reference first, GPU
same equations with segment checkpoints and local replay. API accepts general v.
No output-monitor, final CNT or final pump cotangents; internal pump dependencies
remain. Existing GMRES, Hookstep, history and guards remain unchanged.

Evidence gates, frozen before numerical results:
- Linear and analytic shell tangent dot products: relative error <1e-11.
- Nonlinear primitive/full residual centered directional derivatives at h=1e-4,
  3e-5,1e-5,3e-6,1e-6: best error <1e-5 for every nonzero block test; retain all
  scales and absolute differences, do not discard near-zero dot products.
- CPU/GPU forward and VJP agreement: relative error <1e-9.
- Four-state annulus projected gradient vs archived central streaming:
  relative error <1e-4; cosine >1-1e-8; slope/Cauchy model agreement <1e-4.
- Core full-output gradient compared with forward streaming from identical
  rebuilt C model and separate central checks if forward error exceeds 1e-4.
- Only after derivative gates, compare H, H+fresh annulus, H+full gradient at
  identical Z,C,bank,radius; report prediction, actual, rho and safeguard checks.

Benchmark identical-state annulus streaming, ordinary forward and general VJP
with v=R including its primal residual construction. Warm up first; synchronize
GPU; report wall times, memory metric definition, speedup and forward cost ratio.
No presumed speedup. Peak memory includes allocator reservations, distinguished
from live tensor memory. Four states are independent counterfactual probes.

Mechanism/evidence correspondence:
- Coordinate scaling/sign/gauges -> shell_vjp -> analytic mock-map block tests.
- Local nonlinear intensity dependence -> Kerr/CNT -> FD scale sweep.
- Pump/downstream equilibrium coupling -> EDF reverse -> population-only v.
- Full current gradient -> projection -> archived annulus and streamed core.
- Better direction vs cheaper direction -> frozen model plus true map controls.

Deliver general CPU/GPU VJP, tests, source/data provenance, benchmark, offline
Chinese report and public repository update. No root or stability claim.

## Observed results

## Discrete adjoint and frozen gradient controls

[Implementation](docs/discrete_adjoint.md), [offline report](docs/discrete_adjoint_report.html),
and [archived results](results/steady_adjoint_20260920).
General `CavityResidual.vjp(x,v)` differentiates the packed real residual;
`DiscreteAdjoint.value_and_vjp(x)` computes R and J^T R with segment replay.
CPU/GPU checks cover shell, linear maps, Kerr, CNT recurrence and EDF pump/rates.
97 CPU unit tests pass, including both OC/CNT orders. Arbitrary cotangent block
tests pass at all four full-resolution states; all five FD scales are recorded.

| State | Annulus error | Core forward-FD error | Stream/adjoint time | Adjoint/forward | Full/annulus predicted | Full/annulus actual |
|---|---|---|---|---|---|---|
| 5 | 2.66e-08 | 9.66e-06 | 386.9 | 6.69 | 1.8405 | 1.8413 |
| 11 | 3.1e-08 | 4.05e-06 | 443.8 | 6.65 | 1.6295 | 1.9238 |
| 20 | 2.44e-08 | 6.04e-06 | 525.2 | 6.48 | 1.7751 | 1.7811 |
| 27 | 2.33e-08 | 6.6e-06 | 349.2 | 7.69 | 2.1014 | 2.1468 |

Warm synchronized timing includes the adjoint primal. Forward/VJP use five
repetitions; streaming is one full 6144-column central sweep per state. Isolated
pool reserved memory is not total process/device memory. Core uses the archived
forward-difference convention; annulus uses central differences.
All candidates preserve Z,C,history and radius. No new trajectory, trigger
change, normal-equation solver or physical stability certification is included.
Full-direction gains do not locate their information to frequencies outside
750 GHz, since core/inversion/gauge components also differ.

Reproduce with an empty LASER_OUTPUT_DIR and configured CuPy:

```text
python run_adjoint_validation.py
python run_adjoint_frozen.py
python check_adjoint.py
python report_adjoint.py
```
