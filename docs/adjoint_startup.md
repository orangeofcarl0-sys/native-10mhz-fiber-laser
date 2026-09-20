# Discrete adjoint startup contract

This is the historical startup contract. The implemented API and its validation
workflow are documented in [discrete_adjoint.md](discrete_adjoint.md).

The four-state retrospective experiment meets the frozen 3/4 gate: fresh
annulus gains are 1.2842, 1.8421, 3.0114 and 8.9496 while old gains stay below
1.003. All twelve full-map candidates pass. This motivates developing a current
gradient, not more persisted-seed continuation or trigger tuning. It does not
yet measure adjoint speedup. Near-zero old/fresh cosines establish mismatch to
the persisted seed, not a measured rotation rate between consecutive fresh seeds.

## Exact object to differentiate

Implement `vjp(x, v) = J(x).T @ v` for the real packed/scaled residual in
`steady_state.CavityResidual`, using real inner products (equivalently
Re(conj(u) @ v) for complex optical fields). Accept arbitrary `v` from day one;
the first application is `v=R(x)`. Recover
the audited annulus gradient by restricting the full gradient through the same
orthonormal coordinates in `steady_descent_sources.annulus_coordinates`.
Keep the existing grid, field/population scales and fixed gauges unchanged.

Reverse the actual discrete operations in `spectral_engine.SpectralEngine`:

1. Field packing and output residual subtraction; phase/FFT delay alignment;
   population residual `(p-rates_a/rates_b)/sqrt(cells)`; both fixed-template
   gauges. Include derivatives with respect to phase and delay.
2. Linear FFT propagation and rotations with their actual normalization:
   forward FFT adjoint is N times inverse FFT; inverse FFT adjoint is forward/N.
   Include Jones matrix, OC amplitude factors, losses and PDL.
3. Kerr real derivatives, including conjugate/intensity dependence. Preserve
   the different lab/segment polarization bases used by EDF/passive segments.
4. EDF frozen inversion optical gain and the full longitudinal pump/rate chain.
   Frozen inversion during the map does not remove optical dependence of the
   equilibrium inversion residual. Differentiate geometric-mean powers and
   `rates_a/rates_b`; keep pump attenuation dependencies between cells.
5. CNT temporal recurrence and midpoint absorption. Differentiate the physical
   recurrence underlying the GPU prefix implementation, then verify equivalence
   to the current forward kernel. Initial CNT q is fixed under the existing
   inter-window recovery assumption; internal q depends on earlier optical power.

Use segment/cell checkpointing and deterministic recomputation before retaining
a full propagation tape. Record peak GPU memory and total forward/backward cost.
No differentiable framework or performance factor is assumed here.

## Acceptance before solver integration

Validate each operator and then the assembled VJP with independent centered
Jv dot products, sweeping finite-difference size and including optical,
population, phase and delay directions. Require convergence to a consistent
error floor rather than passing one convenient difference size. At all four
saved states compare the projected VJP with the archived 6144-column gradient:
direction, norm, first-order descent and resulting frozen hookstep prediction.
Recheck actual full-map decline with the existing gate. Define numerical
tolerances from these derivative convergence checks before changing solver use.

Benchmark wall time and memory against the same streaming task at the same
states. Only after agreement and measured benefit should fresh VJP directions
enter the history-bank solver. This document starts the design/validation track;
the retrospective commit contains no adjoint implementation and no new trajectory.
