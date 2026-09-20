# One immutable annulus seed across 30 solver steps

Start at the same S3 endpoint (1.1883945817e-3), radius 0.003125; first step must
match prior G7+A. Fixed cavity, parameters, grid, scale and gauges. No annulus
sweep, pump/Psat scan, adjoint or multiple-shooting change during this run.

Current C and all historical/seed responses are recalculated at each state.
The original seven histories are retained initially. An accepted fresh C
direction is promoted into the next model. Cap history at12 using current
leave-one-out model loss with seed included, at the state's initial radius.
Current C is always separate; oldest is only a numerical tie breaker.

Keep step-relevant gate, Cauchy safeguard, multi-scale candidate check and full
nonlinear rho. Every attempted radius records model gain with/without seed.
Every modeled state records seed age, input/response novelty, bank membership,
fresh/cheap source, and radius. Archive accepted states for independent replay.

Success A: final R<1e-3 or last-five mean norm gain>0.1% per accepted step.
Success B: GA>1.15 at strictly more than half of completed fresh states; this is
an operational durability measure, not proof a fresh seed could not improve.
Trigger diagnostic only: GA<1.15 for3consecutive fresh states plus prior-five
mean gain<0.02%/step. No sweep is permitted even if triggered. Actual reseed
frequency, recovered fresh-gradient value and adjoint economics remain untested.

## Recorded outcome
Accepted steps: 30; status: iteration_budget_reached.
Residual: 0.00118839458173 -> 0.00113064270299.
Last-five mean norm gain: 0.0842852%/step.
GA>1.15 at 1/9 fresh states.
Prespecified progress criterion A: False; durability criterion B: False.
No fresh annulus sweep occurred. Hypothetical trigger state indices (zero-based):
[]. These do not establish actual reseeding need,
fresh-seed recovery or adjoint economics. No stable-pulse certificate is claimed.
