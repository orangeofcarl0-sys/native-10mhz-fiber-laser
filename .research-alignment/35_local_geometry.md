# 35 Period-1 local geometry closure audit

Primary question: distinguish insufficient linear resolution, omitted objective curvature, formulation effects and possible nonzero stationarity, without another outer trajectory.
Frozen state: fresh192 arm step20 from phase34, R=0.0009315468344505816. Same physical parameters, grid, scale and gauges.
H1/H2/H3 can coexist. Local stationarity does not prove global root absence; nonzero stationary points may be saddles/maxima. Chi is a scaled gradient diagnostic, not an unweighted projection angle.

## Measurements frozen before execution
- General adjoint g=J^T R; sigma_max estimate combines384-dimensional GKB largest Ritz value and3 independently seeded20-iteration J^TJ power runs (extend to60 if eigen residual>1e-5). Record Ritz residual/start agreement; no claim of a rigorous global upper bound.
- One frozen GKB up to384, checkpoints64,96,128,192,256,320,384. Unconstrained SVD pseudoinverse relative cutoff1e-12; eta from actual saved JV, not just nominal B. Record full ||J^T(R+Js)||/||g||, projected residual, step norm, singular spectrum, Jacobian response consistency. Do not execute a nonlinear LS step.
- Directions: -g, current unconstrained K64/K96/K192 steps, last accepted fresh step evaluated at current state, and3 fixed-seed random combinations of columns321–384. Normalize in existing state metric.
- Full HVP from central adjoint gradients at h=1e-4,3e-5,1e-5,3e-6,1e-6. GN HVP=J^T(Jv). Record signed quadratic curvatures, kappa absolute/GN, vector missing ratio, h-convergence and symmetry.
- Independent cavity-map scalar second differences and R-weighted second derivative, not only adjoint primal. Known polynomial Hessian and ill-conditioned full-rank counterexample tests.
- h=1e-5 is reference; compare3e-6 and1e-6. A curvature classification requires relative change vs GN scale<1% on two fine scales. Report unresolved otherwise.

## Conditional formulation pilot
A small2-interval pilot is triggered if (reliable chi>=1e-2 and eta384<=0.1 and validated kappa>=0.5), OR a high linear residual floor is credibly resolved (eta>=0.5, full normal residual ratio<1e-3, last two eta marginal improvements<1e-3).
These are operational evidence gates, not mathematical theorems. If neither holds, do not claim H3/H4 or implement another formulation from an unresolved finite-K plateau. Record the unmet condition explicitly.
Any pilot must preserve gauge and population coupling and compare a common condensed closure, not just scaled enlarged residual norms. No production switch or long run.

Deliverables: all diagnostic directions/HVPs, linear ladder, norm estimates, numerical health, independent checks, decision evidence, offline Chinese HTML and public code/data. Large GKB bases remain local with reconstruction inputs/hash.

## Measured evidence
Frozen R=9.315468344505816e-4; ||g||=6.94042676604592e-6; sigma estimate2.835289885258; chi0.002627749808. Three power starts converge at20 iterations.
K384 eta0.992744946699; full normal residual ratio0.880967191339, projected ratio1.666329e-7. High eta is not a resolved range-exclusion floor.
K64/K96/K192/last-step omitted quadratic curvature is positive24.6036/34.9746/32.7258/34.4470 percent of GN; all8 directional scale checks pass. Neither operational pilot gate fires. No multiple shooting or outer continuation.
Independent8-direction physical scalar checks: maximum best-scale joint error6.35181e-7 of GN;28 Hessian symmetry pairs maximum scaled error7.02385e-10;7 linear checkpoints with3 FD scales maximum response error4.25293e-7. Two analytical tests pass.
H1-style incomplete linear resolution and H2 direction-dependent curvature coexist. H3 remains untested and H4 is not established; no root/stability claim. Offline report18 sections,6 images, no external assets or page errors,430px mobile width.
