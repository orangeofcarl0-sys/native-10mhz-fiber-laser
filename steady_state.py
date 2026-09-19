"""Period-one relative equilibria; solver iterations are not physical time.

Field units sqrt(W), time ps, inversion dimensionless. CNT is eliminated only
when its inter-window recovery factor is below 1e-14. Keep the grid fixed.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
from spectral_engine import SpectralEngine


class CavityResidual:
    """Stateful propagator plus fixed template for smooth phase/time gauges."""

    def __init__(self, c, dt, template, pump, oc, gpu=False):
        self.engine = SpectralEngine(dict(c, gain_mode="frozen"), dt,
                                     template.shape[-1], [pump], [oc],
                                     gpu=gpu, fused_edf=False)
        self.n, self.dt = template.shape[-1], dt
        self.cells = self.engine.ops[1][0]
        gap = 1e12 / self.engine.e.rep - self.n * dt
        if gap <= 0 or np.exp(-gap / c["sa_recovery_ps"]) > 1e-14:
            raise ValueError("CNT state cannot be eliminated at this window/recovery time")
        self.scale = np.linalg.norm(template)
        if self.scale == 0:
            raise ValueError("A nonzero pulse template is required")
        self.template = template / self.scale
        self.omega = 2 * np.pi * np.fft.fftfreq(self.n, dt)
        tangent = np.fft.ifft(1j * self.omega * np.fft.fft(self.template), axis=-1)
        tangent -= 1j * self.template * np.vdot(1j * self.template, tangent).real
        self.time_scale = 1 / np.linalg.norm(tangent)
        self.time_tangent = tangent * self.time_scale
        self.evaluations = 0

    def pack(self, a, pop, phase=0., shift_ps=0.):
        z = (a / self.scale).ravel()
        return np.r_[z.real, z.imag, np.ravel(pop) / np.sqrt(self.cells),
                     phase, shift_ps / self.time_scale]

    def unpack(self, x):
        m = 2 * self.n
        a = (x[:m] + 1j * x[m:2*m]).reshape(2, self.n) * self.scale
        pop = x[2*m:-2] * np.sqrt(self.cells)
        return a, pop, x[-2], x[-1] * self.time_scale

    def feasible(self, x):
        pop = x[4*self.n:-2] * np.sqrt(self.cells)
        return np.isfinite(x).all() and np.all((pop > 0) & (pop < 1))

    def __call__(self, x):
        a, pop, phase, shift = self.unpack(x)
        e, xp = self.engine, self.engine.xp
        aa = xp.asarray(a[None]).copy()
        pp = xp.asarray(pop[None]).copy()
        q = xp.full(1, e.c["sa_modulation"], dtype=xp.float64)
        b, out, _, _, _ = e.step(aa, pp, q)
        host = (lambda v: xp.asnumpy(v)) if xp is not np else np.asarray
        b, self.output = host(b[0]), host(out[0])
        self.neq = host(e.rates_a[0] / e.rates_b[0])
        self.rates_b = host(e.rates_b[0]).copy()
        # T_-shift: b(t+shift); phase and delay remain unknowns of the solve.
        b = np.fft.ifft(np.fft.fft(b) * np.exp(1j*self.omega*shift)) * np.exp(-1j*phase)
        r = ((b-a) / self.scale).ravel()
        delta = a / self.scale - self.template
        gauge = [np.vdot(1j*self.template, delta).real,
                 np.vdot(self.time_tangent, delta).real]
        self.evaluations += 1
        return np.r_[r.real, r.imag, (pop-self.neq)/np.sqrt(self.cells), gauge]


def solve(residual, x, max_steps=25, tolerance=1e-7, inner=25):
    """Damped matrix-free Newton; physical bounds enforced by line search.

    LGMRES approximates J dx=-r. Record residuals and calls, not fictitious RT.
    A small residual alone may describe zero field, CW, or multiple pulses.
    """
    x = x.copy()
    r = residual(x)
    history = []
    for step in range(max_steps+1):
        norm = np.linalg.norm(r)
        history.append(dict(step=step, residual=float(norm), calls=residual.evaluations))
        if norm < tolerance:
            return x, history, "residual_converged"
        if step == max_steps:
            break
        def jv(v):
            h = 1e-7 / max(np.linalg.norm(v), 1e-100)
            return (residual(x+h*v)-r)/h
        operator = LinearOperator((x.size, x.size), matvec=jv, dtype=float)
        dx, info = lgmres(operator, -r, rtol=0.03, atol=0,
                          maxiter=2, inner_m=inner, outer_k=3)
        history[-1]["linear_info"] = int(info)
        history[-1]["linear_relative_residual"] = float(np.linalg.norm(jv(dx)+r)/norm)
        accepted = False
        for k in range(14):
            alpha = 0.5**k
            trial = x + alpha*dx
            if not residual.feasible(trial):
                continue
            rr = residual(trial)
            if np.isfinite(rr).all() and np.linalg.norm(rr) < (1-1e-4*alpha)*norm:
                x, r, accepted = trial, rr, True
                history[-1]["alpha"] = alpha
                break
        if not accepted:
            return x, history, "line_search_stalled"
    return x, history, "iteration_budget_reached"
