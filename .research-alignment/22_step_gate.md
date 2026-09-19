# Step-relevant gate experiment

Fixed 27.5 mW, saved A/B endpoints, unchanged physics, mesh, gauge and scaling.
First audit normalized central differences at six scales and Richardson half steps.
Then inspect raw augmented and Cauchy candidates at three radii without advancing.
Conditional B continuation: 30 steps, initial radius .00625, original Arnoldi .008/240,
C preconditioner, verified fresh/cheap descent. Candidate model must agree within 5%
at 3e-6 and 1e-6 and actual rho exceed .1. Newton 1% gate only inside radius.
No pump scan, no 27.0 mW repeat, no stability claim without independent certification.
Arnoldi internal residual is distinguished from direct finite-difference action on
its combined Newton vector. Neither smallest h nor Richardson is assumed truth.


Outcome: fixed-state candidates pass all six directional scales (model discrepancy
about0.049%; rho0.977/0.917/0.619). B middle-scale Newton defect remains about1.74%,
while smaller h worsens cancellation; Arnoldi internal residual is not its direct
combined-direction finite difference. Thirty additional accepted steps reduce
norm0.811663%, ending1.2081439204973431e-3. Four steps would be blocked by the old
gate. Nine fresh and21checked cheap directions, zero fallback. Last5 mean improvement
0.004994568%/step. Endpoint replay exact; field residual1.20811788e-3 dominates.
No root or stability certificate. Fresh slopes fluctuate, so no full-space stationary
point claim. Next formulation/gradient enrichment remains a separate experiment.
