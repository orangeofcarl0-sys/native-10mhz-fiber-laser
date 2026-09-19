# Single-state descent-source audit

Question: does useful descent lie in optical frequencies outside C, historical
fresh directions, or one physically scaled parameter direction at the saved endpoint?
No further current-method search is authorized by this experiment. Reconstruct the
previous run solely to recover omitted vectors; verify all30norms and the endpoint.
S1 current C, S2 C+375--750GHz annulus, S3 C+last3fresh, S4 C+best physical parameter.
Same current C preconditioner, Arnoldi basis, Jv, gauges, residual and radii
.00156/.003125/.00625. S4 alone adds a scaled physical coordinate, no extra equation.
Parameter scales5mW/.02ps²/.05/4W; basepump must remain27.5mW for ALL columns.
Select parameter using one-dimensional model reduction at radius.003125 before
candidate comparisons. Only a valid candidate with actual merit reduction >=1.25
same-radius valid S1 qualifies for30step continuation. No root/stability claim.
Central annulus columns are streamed; no12000-dimensional matrix/factorization.
A small response cosine for one direction does not establish orthogonality to the
whole range. Negative audits of a finite band/history do not prove full stationarity.


## Observed controls and causal limits

| Hypothesis | Implementation / observable | Observation | Bounded inference |
|---|---|---|---|
| Extra spectral support contains useful descent | annulus_coordinates + streamed_gradient; full-output central products, current Jd, actual map | annulus norm1.37305e-4 versus C2.30304e-5; S2 actual5.43--5.48e-9 | useful complement exists; not proof bandwidth alone causes the plateau |
| Rank-one enrichment loses useful combinations | last3 recovered fresh vectors; responses recomputed at endpoint; shared H/Z/V | S3 actual5.29--5.45e-9, near S2; two old directions individually point uphill | C-supported extra directions help the full-grid Krylov model; one response cosine does not diagnose full range stationarity |
| A physical coordinate helps locally | fixed27.5mW base for all columns; scaled scalar trust model | Psat selected; S4 weaker at tested radii, passes25% rule at two radii | scalar ranking is not joint global optimization; large parameter changes are untested |

All12 candidates independently replay exactly. S1 alternate model assembly differs
in state step by at most8.6e-15 relative and prediction by0. The recovery reproduces
all30prior residuals and its endpoint state exactly. S3 at.00625 achieves99.56% of
the best actual one-step decrease, with no recurring6144-column annulus sweep;
select this qualified cheaper route for30steps. Keep pump/other physical parameters
fixed. Compare no long-run speedup factor without an equal-start S1/S2 trajectory.


Implementation caveat: C is NOT a hard cutoff of the entire search. The inverse
uses -I on its complement; original Krylov vectors already span full-grid input.
Saved S1 steps have nonzero annular components. Orthogonal support accounting is
in step_support.json. Historical extras are C-supported, but S3 final steps are
not exclusively C-supported. Do not infer a hard bandwidth barrier or full-space
stationarity from these controls.


S3 continuation completes30accepted steps,7fresh and23cheap current C directions,
no fallback. Norm decreases1.634684% to1.1883945817279438e-3; last5steps average
0.0152447% each. Candidate model discrepancy <=1.986e-6 relative; rho ranges
0.2433--1.0505. Residual and output replay differences are zero. Field residual
1.1882997e-3 falls, but population residual rises to1.4937658e-5: not all blocks
improve. Output0.90297649nJ remains uncertified. No equal-start long-run S1/S2
comparison, no speedup factor, and no full-space stationarity/no-root claim.
