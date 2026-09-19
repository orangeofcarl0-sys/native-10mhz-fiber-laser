# Paired outer-loop research contract

Question: does the fixed-state augmentation gain persist after updating the cavity state?
Control: 27.5 mW published down_275, same physical map/template/scales, C forward 1e-6 builder, centered Jv 1e-5 and independent 1e-6 check, 240 vectors, relative gate <1%, 40 outer steps, initial radius0.003125, max radius2. Both arms share radius and linear/model/rho refresh rules. Only B adds verified descent, augmentation and Cauchy fallback.
Mechanism: rebuild columns B=JE yield full-output B^T R; interim A_C^T P R is cheap but checked by current full Jd. Hypothesis: adding verified missing directions increases accepted merit reduction and may reach a faster root-solving regime. Expected observables: R, radius, rho, G_C/G_H, span error, alpha_C, step and Newton norms, refresh reason, raw model pass/fallback. Counterexample: same slow decay or independent linear gate fails despite fresh builder.
Scope: no stale unchecked direction, no forced energy, no period-p, no broad pump sweep. A failed linear gate stops safely after fresh rebuild, with the unused outer budget explicit. 27.0 confirmation only if 27.5 cumulative advantage is substantial. No root/stability inference from mere improvement.

Before seeing paired outcomes, define substantial cumulative improvement operationally as B residual at least20% below A at the same accepted-step count after at least30 accepted steps. This is a deterministic effect-size trigger for27.0 replication, not statistical significance. If either arm cannot meet the linear gate, report that limitation rather than relaxing it silently.

An early B numerical root also qualifies for replication if its matched-step residual is at least20% lower; a successful early solve need not artificially continue to30 steps. This rule is recorded before B starts.


Observed final: A18acceptedsteps, R=1.2758381004591417e-3; B30acceptedsteps, R=1.2180302219795991e-3. Both stop at independent linear gate, budgets not completed. Common18step residual decrease factor3.45; B total norm decrease5.9458%, last5mean0.04235%/step. All30B raw augmented,9full and21cheap valid directions, no fallback/raw model failures. This supports cumulative benefit, not fast convergence or existence/stability claims. The predeclared27.0replication trigger is not met.

Discriminating endpoint check: force240Arnoldi directions after fresh builders, without advancing states. A model4.09e-11 vs independent0.0300918; B model7.68e-12 vs independent0.0186085. Thus raising the internal basis budget alone does not resolve the failed gate. Huge Newton norms27278.8/6134.0 predominantly in field block motivate directional finite-difference/linearity verification; cause not established as solely difference noise or condition amplification.

Endpoint replay identical; accepted history continuous/monotone, all accepted gates pass. B field1.21795e-3, population1.31612e-5, gauge5.67669e-6; gauge worsens from3.42e-7, so not all blocks improve.71CPU tests passed across the full suite and two added targeted tests. Numerical improvement remains distinct from physical pulse certification.
