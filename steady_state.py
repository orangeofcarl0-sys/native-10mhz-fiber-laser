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
        self.gpu, self.pump, self.oc = gpu, pump, oc
        self.batch_engines = {}

    def batch(self, states):
        """Independent finite-difference probes; same template and fine grid.

        Count every cavity evaluation, including simultaneous GPU cases.
        Main one-state diagnostics remain unchanged by these probe maps.
        """
        count, n = len(states), self.n
        if count not in self.batch_engines:
            self.batch_engines[count] = SpectralEngine(
                self.engine.c, self.dt, n, [self.pump]*count, [self.oc]*count,
                gpu=self.gpu, fused_edf=False)
        e = self.batch_engines[count]
        xp = e.xp
        a = (states[:,:2*n]+1j*states[:,2*n:4*n]).reshape(count,2,n)*self.scale
        p = states[:,4*n:-2]*np.sqrt(self.cells)
        b,_,_,_,_ = e.step(xp.asarray(a).copy(),xp.asarray(p).copy(),
                          xp.full(count,e.c['sa_modulation'],dtype=xp.float64))
        host = xp.asnumpy if self.gpu else np.asarray
        b,neq = host(b),host(e.rates_a/e.rates_b)
        factor=np.exp(1j*states[:,-1,None]*self.time_scale*self.omega-1j*states[:,-2,None])
        b=np.fft.ifft(np.fft.fft(b)*factor[:,None])
        r=((b-a)/self.scale).reshape(count,-1)
        delta=a/self.scale-self.template
        gauges=np.column_stack([np.sum(np.conj(v)*delta,axis=(1,2)).real
                                for v in [1j*self.template,self.time_tangent]])
        self.evaluations += count
        return np.concatenate([r.real,r.imag,(p-neq)/np.sqrt(self.cells),gauges],axis=1)

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


def solve(residual, x, max_steps=25, tolerance=1e-7, inner=25, precondition=False,
          coarse_cutoff=32, rebuild_every=1, progress=None, observer=None):
    """Damped matrix-free Newton; physical bounds enforced by line search.

    LGMRES approximates J dx=-r. Record residuals and calls, not fictitious RT.
    A small residual alone may describe zero field, CW, or multiple pulses.
    """
    x = x.copy()
    r = residual(x)
    history = []
    preconditioner = None
    if rebuild_every < 1:
        raise ValueError('Preconditioner rebuild interval must be positive')
    for step in range(max_steps+1):
        norm = np.linalg.norm(r)
        history.append(dict(step=step, residual=float(norm), calls=residual.evaluations))
        if progress is not None:
            progress(history[-1])
        if norm < tolerance:
            return x, history, "residual_converged"
        if step == max_steps:
            break
        def jv(v):
            h = 1e-7 / max(np.linalg.norm(v), 1e-100)
            return (residual(x+h*v)-r)/h
        operator = LinearOperator((x.size, x.size), matvec=jv, dtype=float)
        if precondition:
            from steady_preconditioner import build, build_coarse
            rebuild = step % rebuild_every == 0 or (
                step > 0 and history[-2].get('linear_relative_residual',1) > .1)
            if rebuild:
                if precondition == 'coarse':
                    preconditioner, diagnostics = build_coarse(residual,x,r,cutoff=coarse_cutoff)
                else:
                    preconditioner, diagnostics = build(residual,x,r)
            history[-1]['precondition_rebuilt'] = rebuild
            history[-1].update(diagnostics)
        # Right preconditioning preserves the original residual norm in Krylov.
        linear = operator if preconditioner is None else LinearOperator(
            operator.shape, matvec=lambda v: operator @ (preconditioner @ v), dtype=float)
        direction, info = lgmres(linear, -r, rtol=0.03, atol=0,
                                 maxiter=2, inner_m=inner, outer_k=3)
        dx = direction if preconditioner is None else preconditioner @ direction
        history[-1]["linear_info"] = int(info)
        history[-1]["linear_relative_residual"] = float(np.linalg.norm(jv(dx)+r)/norm)
        if observer is not None:
            observer(step, x.copy(), r.copy(), dx.copy())
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
