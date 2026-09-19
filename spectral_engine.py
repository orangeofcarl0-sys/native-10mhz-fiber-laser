"""Domain-aware FP64 cavity map; preserve the reference Strang split exactly.

State F[case, polarization, frequency] uses the unnormalized forward FFT.
Passive Kerr is evaluated in the segment basis, EDF Kerr in the lab basis,
matching the reference implementation (these conventions must not be swapped).
"""

from python_engine import Engine


class SpectralEngine:
    def __init__(self, c, dt, n, pumps, oc, gpu=False, fused_cnt=True, fused_edf=True):
        if gpu:
            from gpu_setup import cp as xp
            from cupyx.scipy.fft import fft, ifft
            from gpu_batch_core import kerr, absorber

            if fused_cnt:
                from gpu_kernels import absorber_prefix as absorber
        else:
            import numpy as xp
            from scipy.fft import fft, ifft
            from batch_engine import kerr, absorber
        self.xp, self.fft, self.ifft = xp, fft, ifft
        self.kerr, self.absorber = kerr, absorber
        self.c, self.dt, self.n = c, dt, n
        self.fused_edf = gpu and fused_edf
        if self.fused_edf:
            from gpu_kernels import weighted_power, gain_half, rate_update, kerr_inplace

            self.weighted_gpu, self.gain_half, self.rate_update = (
                weighted_power,
                gain_half,
                rate_update,
            )
            self.kerr = kerr_inplace
        self.e = Engine(c, dt, n)
        e = self.e
        self.ops = [(s, d, xp.asarray(h), xp.asarray(r), g) for s, d, h, r, g in e.ops]
        self.full = [h * h for _, _, h, _, _ in self.ops]
        self.profile, self.J = xp.asarray(e.profile), xp.asarray(e.J)
        self.pumps, self.oc = xp.asarray(pumps, dtype=xp.float64), xp.asarray(
            oc, dtype=xp.float64
        )
        self.last_gap, self.last_tau = xp.zeros(len(oc)), xp.zeros(len(oc))
        self.rates_b = xp.zeros((len(oc), self.ops[1][0]), dtype=xp.float64)
        # Available in the unfused path for joint optical/inversion root solving.
        self.rates_a = None if self.fused_edf else xp.zeros_like(self.rates_b)
        self.transform_count = 0

    def forward(self, a):
        self.transform_count += 1
        return self.fft(a, axis=-1)

    def inverse(self, f):
        self.transform_count += 1
        return self.ifft(f, axis=-1)

    def weighted(self, f):
        xp, e = self.xp, self.e
        if self.fused_edf:
            return self.weighted_gpu(f, self.profile, self.dt * 1e-12 * e.rep / self.n)
        return (
            xp.sum(abs(f) ** 2 * self.profile, axis=(1, 2))
            * self.dt
            * 1e-12
            * e.rep
            / self.n
        )

    def fiber_spectrum(self, f, index, pump, pop):
        xp, c, e = self.xp, self.c, self.e
        steps, dz, half, r, gamma = self.ops[index]
        if index != 1:
            f = half * (r.T @ f)
            for j in range(steps):
                a = self.kerr(self.inverse(f), gamma * dz)
                f = (half if j == steps - 1 else self.full[index]) * self.forward(a)
            return r @ f, pump, pop
        f = r.T @ f
        old = self.weighted(f)
        if len(self.last_gap) != len(pop):
            self.last_gap = xp.zeros(len(pop))
            self.last_tau = xp.zeros(len(pop))
            self.rates_b = xp.empty_like(pop)
            if not self.fused_edf:
                self.rates_a = xp.empty_like(pop)
        self.last_gap.fill(0)
        self.last_tau.fill(0)
        for j in range(steps):
            if self.fused_edf:
                first = self.gain_half(f, half, self.profile, pop, j, c, dz)
                a = self.kerr(r @ self.inverse(first), gamma * dz)
                f = self.gain_half(
                    self.forward(r.T @ a), half, self.profile, pop, j, c, dz
                )
                new = self.weighted(f)
                self.rate_update(
                    old,
                    new,
                    pump,
                    pop,
                    self.last_gap,
                    self.last_tau,
                    j,
                    c,
                    e,
                    dz,
                    rates_b=self.rates_b,
                )
                old = new
                continue
            inv = pop[:, j].copy()
            gain = (c["emission_s_m"] * inv - c["alpha_s_m"] * (1 - inv))[
                :, None
            ] * self.profile
            hh = half[None] * xp.exp(gain[:, None] * dz / 4)
            a = r @ self.inverse(hh * f)
            a = self.kerr(a, gamma * dz)
            f = hh * self.forward(r.T @ a)
            new = self.weighted(f)
            signal = xp.sqrt(old * new)
            pmid = pump * xp.exp(-c["alpha_p_m"] * (1 - inv) * dz / 2)
            A = (c["alpha_p_m"] * pmid / e.hp + c["alpha_s_m"] * signal / e.hs) / e.ions
            B = (
                1 / c["upper_lifetime_s"]
                + (
                    c["alpha_p_m"] * pmid / e.hp
                    + (c["alpha_s_m"] + c["emission_s_m"]) * signal / e.hs
                )
                / e.ions
            )
            self.rates_b[:, j] = B
            self.rates_a[:, j] = A
            if c.get("gain_mode", "dynamic") != "frozen":
                pop[:, j] = inv + (A / B - inv) * (-xp.expm1(-B / e.rep))
            pump *= xp.exp(-c["alpha_p_m"] * (1 - inv) * dz)
            self.last_gap = xp.maximum(self.last_gap, abs(inv - A / B))
            self.last_tau = xp.maximum(self.last_tau, 1 / B)
            old = new
        return r @ f, pump, pop

    def split(self, a):
        a = a * 10 ** (-self.c["oc_excess_dB"] / 20)
        return (
            a * self.xp.sqrt(1 - self.oc)[:, None, None],
            a * self.xp.sqrt(self.oc)[:, None, None],
        )

    def step_spectrum(self, f, pop, q):
        c, xp = self.c, self.xp
        batch = len(self.oc)
        if (
            f.shape != (batch, 2, self.n)
            or pop.shape != (batch, self.ops[1][0])
            or q.shape != (batch,)
            or self.pumps.shape != (batch,)
        ):
            raise ValueError(
                "Field, longitudinal population, pump and CNT shapes disagree"
            )
        if f.dtype != xp.complex128 or pop.dtype != xp.float64 or q.dtype != xp.float64:
            raise ValueError(
                "Spectral map requires complex128 field and float64 population/CNT"
            )
        if not (
            f.flags.c_contiguous and pop.flags.c_contiguous and q.flags.c_contiguous
        ):
            raise ValueError("Spectral state arrays must be contiguous")
        pump = self.pumps * 10 ** (-c["pump_path_dB"] / 10)
        for j in range(6):
            f, pump, pop = self.fiber_spectrum(f, j, pump, pop)
            if j == 3:
                a = self.J @ self.inverse(f)
                if c["topology"] == "OC_CNT":
                    a, out = self.split(a)
                a, q, sa = self.absorber(a, q, c, self.dt)
                if c["topology"] == "CNT_OC":
                    a, out = self.split(a)
                f = self.forward(a) * 10 ** (-c["splice_connector_dB"] / 20)
        f *= 10 ** (-c["hybrid_IL_dB"] / 20)
        f[:, 1] *= 10 ** (-c["hybrid_PDL_dB"] / 20)
        q = c["sa_modulation"] + (q - c["sa_modulation"]) * xp.exp(
            -max(0, 1e12 / self.e.rep - self.n * self.dt) / c["sa_recovery_ps"]
        )
        return f, out, pop, q, sa

    def step(self, a, pop, q):
        f, out, pop, q, sa = self.step_spectrum(self.forward(a), pop, q)
        self.last_spectrum = f
        return self.inverse(f), out, pop, q, sa
