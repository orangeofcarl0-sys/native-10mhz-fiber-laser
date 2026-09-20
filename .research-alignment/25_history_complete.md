# Frozen-state complete-history and annulus novelty audit

Keep the S3 endpoint, all equations, grid, parameters, gauges and scales fixed.
Reuse saved seven C histories and annulus direction. Rebuild missing common
C/Arnoldi basis, independently recompute responses, and replay all prior 15 models
before interpreting new comparisons. No fresh annulus sweep or continuation.

Extend greedy through seven without early marginal stopping. Compare G5--G7 and
G4+A--G7+A at the same three radii. Enumerate 128 subsets per radius; compare
greedy against exhaustive optimum at equal cardinality (unrestricted nested
space optimum is trivially the full space). Validate E4 with the full mapping.

Use SVD rank-aware projections for annulus novelty in state and output spaces,
excluding the residual target from the latter. Report tolerance sensitivity and
shared-coefficient normalized stacked fit: separate span membership alone does
not establish paired linear-map redundancy. Preserve physical trust metric.

All candidates need Cauchy lower bound, radius/feasibility, direct three-scale
model agreement under 5%, positive actual merit decrease and rho above 0.1.
Archive inputs, compressed model, steps, source bytes and independent replay.
Only current-state statements are justified; no reseeding timing, adjoint speedup,
physical effective dimension, root/stability or multiple-shooting exclusion claim.


## Observed evidence
- G7+A/G7 actual gain 2.52--2.57 at the three radii, above the 1.25 reference.
- Greedy equals exhaustive model optimum at every equal cardinality and radius.
  E4 full-map results equal G4. No greedy failure demonstrated here.
- nu_x=0.9885605, nu_R=0.9740835, joint error=0.9852629. Rank46 for all tested
  SVD tolerances. Independent least-squares projection checks agree exactly.
- 24 new candidates pass, independent replay differences zero. 384 reduced
  model predictions replay exactly; maximum candidate discrepancy 1.16e-7.
- 85 CPU tests pass. Reusing persisted annulus directions across future states
  remains a next experiment, not a completed run or validated event policy.
