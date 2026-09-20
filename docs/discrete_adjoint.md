# General packed-residual VJP

`CavityResidual.vjp(x, v)` returns the real Euclidean `J(x).T @ v`.
`DiscreteAdjoint(residual).value_and_vjp(x)` returns `(R(x), J.T @ R(x))`
from one checkpointed primal/reverse evaluation. Both use the residual's CPU or
GPU backend, fixed scales and gauge templates. Inputs/outputs are NumPy packed
real vectors, matching the existing residual API. This addition does not change
the solver's gradient source or trust policy.

The supported map is the current frozen-population period-one residual. This is
not a derivative of a dynamic-gain trajectory. Physical parameters, initial CNT
absorption and templates are constants, while optical field, inversion, phase
and scaled delay are differentiable state variables.

## Correspondence

- `adjoint_primitives.shell_vjp`: transpose of packing/scaling, aligned output,
  direct negative input, both gauges, and the population residual. It returns
  cotangents for unaligned engine field and equilibrium inversion plus direct
  packed-state terms. Phase and delay derivatives are explicit.
- `kerr_vjp`: analytic real pullback in the circular polarization basis. No
  coordinate finite differences and no autodiff dependency.
- `cnt_vjp`: reverse physical recurrence with fixed initial q and terminal
  cotangent zero. `adjoint_cnt_gpu.py` executes its scalar sequence on GPU;
  vector field operations remain in CuPy. Monitoring outputs have zero cotangent.
- `DiscreteAdjoint.segment_vjp`: FFT/rotation/dispersion and losses; EDF reverse
  includes both gain halves, intensity weighting, geometric-mean signal power,
  A/B equilibrium and the full longitudinal pump recurrence. The final pump
  output is dead, but intermediate pump dependencies remain live.
- `primal/reverse`: six segment checkpoints plus CNT junction input, then local
  replay/tape for one segment at a time. The entire round-trip tape is not kept.

Complex arrays use `Re(vdot(u, v))`. FFT and inverse FFT adjoints carry N and 1/N;
Jones uses its conjugate transpose, not an assumed unitary inverse. The EDF Kerr
basis remains distinct from the passive-segment basis as in `SpectralEngine`.

## Validation and reproduction

```text
python -m unittest test_discrete_adjoint -v
python run_adjoint_validation.py
python run_adjoint_frozen.py
python check_adjoint.py
python report_adjoint.py
```

Use an empty `LASER_OUTPUT_DIR` and a configured CuPy installation for the four
GPU scripts. The first validation script must finish before frozen controls.
The archived annulus gradient is reconstructed losslessly to roundoff from its
unit direction and stored norm; the frozen benchmark independently re-enumerates
all 6144 central columns. Core comparison uses the exact original forward
difference C rebuild, explicitly distinguished from central annulus differences.

Timing includes VJP primal work. The isolated allocator's reserved-byte peak is
an upper bound on live pool tensors, not total process GPU memory. It excludes
pre-existing arrays and FFT plans. Streaming runs once per state; forward and
adjoint timings use five repetitions. Do not infer a scaling law or performance
on other GPUs from these measurements.

All frozen models retain original Z, C, bank and radius. A full gradient need not
outperform an annulus direction in a constrained subspace. If it does, this alone
does not identify frequencies outside 750 GHz: core, inversion and gauge state
components changed too. Solver integration and new trajectories are deferred.
