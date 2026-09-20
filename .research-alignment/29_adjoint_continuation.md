# Forty-step full-adjoint continuation

Start at latest persisted trajectory endpoint R=0.001130642702993049, retaining
its 12 pending history directions and next radius 0.00625. Freeze all physical
parameters, grid and gauges. Forty outer iterations; no new mechanism/parameter.
Current R,g from value_and_vjp every outer iteration. C builds preconditioner
only, no restricted gradient. S=span(Z,full current d,history). Capacity12;
same leave-one-out minimum loss rule conditional on current d. Every accepted
outer direction enters pending history, including reused-preconditioner states.
Old directions retained until ordinary pruning; no persisted annulus seed.

Keep step gate, independent scales, nonlinear rho, feasible bounds, full-gradient
Cauchy bound/fallback and existing trust updates. Record gradient norm, Cauchy
prediction, Gfull/H (H=Z+history), GMRES coverage, bank composition, residual
field/population/gauge blocks and all trials. No slow-progress early stop.

Frozen success: final R<1e-3 strong signal; final five accepted-step mean relative
norm decrease >0.002 improved, 0.0005–0.002 sustained; <0.0002 plus final-five
mean Gfull/H<=1.1 is a plateau diagnostic. Remaining intervals are explicitly
intermediate, not silently assigned to a category. No root/stability claim.
Independent replay of all accepted states, full gradients and guards required.

## Observed outcome
## Full-adjoint forty-step continuation

[Offline report](docs/adjoint_continuation_report.html) and
[records](results/steady_adjoint_continuation_20260920) start at the latest
persisted endpoint, with radius0.00625 and the same12 old history directions.
`solve_augmented(...,value_gradient=...)` computes current R,J^T R each outer
iteration. C supplies only its preconditioner; all other trust/step protections
are unchanged. Every accepted full direction is promoted, also on non-rebuild
steps. Leave-one-out pruning keeps<=12 histories in every actual model.
The terminal pending bank can contain the last accepted direction awaiting
next-state pruning. No annulus gradient or seed is used.

Accepted steps: 40; status: iteration_budget_reached.
Residual: 0.00113064270299 -> 0.00109671297654.
Strong success R<1e-3: False.
Last-five mean norm decrease: 0.0310556%/step.
Last-five mean Gfull/H: 5.34782; regime: intermediate.
H here excludes current C and contains only Z+history; gain ratios are not
directly comparable with the previous four-state H that also contained C.
Field/population/gauge blocks, gradient norm, coverage, Cauchy predictions,
bank composition and every accepted/rejected trial are archived.

Independent full-map/current-gradient/guard replay and98 CPU tests pass.
No parameter release, normal-equation solver or physical pulse certificate.

```text
python run_adjoint_continuation.py
python check_adjoint_continuation.py
python report_adjoint_continuation.py
```

Use a fresh LASER_OUTPUT_DIR and configured CuPy. Raw execution sources and
input hashes are archived separately from the final Git-byte manifest.
