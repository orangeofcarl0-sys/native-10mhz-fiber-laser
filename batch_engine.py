"""Vectorized independent cavities; batch axis never mixes physical fields."""

import numpy as np
from scipy.fft import fft, ifft
from python_engine import Engine


def kerr(a, g):
    p = (a[:, 0] + 1j * a[:, 1]) / np.sqrt(2)
    m = (a[:, 0] - 1j * a[:, 1]) / np.sqrt(2)
    pp = abs(p) ** 2
    mm = abs(m) ** 2
    p *= np.exp(2j / 3 * g * (pp + 2 * mm))
    m *= np.exp(2j / 3 * g * (mm + 2 * pp))
    return np.stack([(p + m) / np.sqrt(2), (p - m) / (1j * np.sqrt(2))], axis=1)


def absorber(a, q, c, dt):
    power = np.sum(abs(a) ** 2, axis=1)
    rate = 1 / c["sa_recovery_ps"] + power / c["sa_saturation_energy_pJ"]
    eq = c["sa_modulation"] / (c["sa_recovery_ps"] * rate)
    decay = np.exp(-rate * dt)
    qm = np.empty_like(power)
    for start in range(0, power.shape[1], 16):
        sl = slice(start, start + 16)
        aa = decay[:, sl]
        bb = eq[:, sl] * (1 - aa)
        product = np.cumprod(aa, axis=1)
        if np.all(product[:, -1] > 1e-200):
            after = product * (q[:, None] + np.cumsum(bb / product, axis=1))
            before = np.concatenate([q[:, None], after[:, :-1]], axis=1)
            qm[:, sl] = eq[:, sl] + (before - eq[:, sl]) * np.sqrt(aa)
            q = after[:, -1]
        else:
            for j in range(start, min(start + 16, power.shape[1])):
                qm[:, j] = eq[:, j] + (q - eq[:, j]) * np.sqrt(decay[:, j])
                q = eq[:, j] + (q - eq[:, j]) * decay[:, j]
    transmission = 1 - c["sa_nonsaturable"] - qm
    assert np.min(transmission) > 0
    metrics = np.stack(
        [power.sum(axis=1) * dt, power.max(axis=1), qm.min(axis=1)], axis=1
    )
    return a * np.sqrt(transmission[:, None, :]), q, metrics


class BatchEngine:
    def __init__(self, c, dt, n, pumps, oc):
        self.c = c
        self.dt = dt
        self.n = n
        self.e = Engine(c, dt, n)
        self.pumps = np.asarray(pumps)
        self.oc = np.asarray(oc)
        self.last_gap = np.zeros(len(oc))
        self.last_tau = np.zeros(len(oc))

    def fiber(self, a, index, pump, pop):
        e = self.e
        c = self.c
        steps, dz, half, R, gamma = e.ops[index]
        if index != 1:
            a = ifft(half * fft(R.T @ a, axis=-1), axis=-1)
            for j in range(steps):
                a = kerr(a, gamma * dz)
                a = ifft(
                    (half if j == steps - 1 else half**2) * fft(a, axis=-1), axis=-1
                )
            return R @ a, pump, pop
        gaps = []
        taus = []
        rates_a = []
        rates_b = []
        for j in range(steps):
            inv = pop[:, j].copy()
            old = (
                np.sum(np.sum(abs(fft(a, axis=-1)) ** 2, axis=1) * e.profile, axis=1)
                * self.dt
                * 1e-12
                * e.rep
                / e.n
            )
            gain = (c["emission_s_m"] * inv - c["alpha_s_m"] * (1 - inv))[
                :, None
            ] * e.profile
            hh = half[None, :, :] * np.exp(gain[:, None, :] * dz / 4)
            a = R @ ifft(hh * fft(R.T @ a, axis=-1), axis=-1)
            a = kerr(a, gamma * dz)
            a = R @ ifft(hh * fft(R.T @ a, axis=-1), axis=-1)
            pmid = pump * np.exp(-c["alpha_p_m"] * (1 - inv) * dz / 2)
            new = (
                np.sum(np.sum(abs(fft(a, axis=-1)) ** 2, axis=1) * e.profile, axis=1)
                * self.dt
                * 1e-12
                * e.rep
                / e.n
            )
            smid = np.sqrt(old * new)
            A = (c["alpha_p_m"] * pmid / e.hp + c["alpha_s_m"] * smid / e.hs) / e.ions
            B = (
                1 / c["upper_lifetime_s"]
                + (
                    c["alpha_p_m"] * pmid / e.hp
                    + (c["alpha_s_m"] + c["emission_s_m"]) * smid / e.hs
                )
                / e.ions
            )
            equilibrium = A / B
            if c.get("gain_mode", "dynamic") != "frozen":
                pop[:, j] = inv + (equilibrium - inv) * (-np.expm1(-B / e.rep))
            pump *= np.exp(-c["alpha_p_m"] * (1 - inv) * dz)
            gaps.append(abs(inv - equilibrium))
            taus.append(1 / B)
            rates_a.append(A)
            rates_b.append(B)
        self.last_gap = np.max(gaps, axis=0)
        self.last_tau = np.max(taus, axis=0)
        self.rates_a = np.array(rates_a).T
        self.rates_b = np.array(rates_b).T
        return a, pump, pop

    def split(self, a):
        a = a * 10 ** (-self.c["oc_excess_dB"] / 20)
        return (
            a * np.sqrt(1 - self.oc)[:, None, None],
            a * np.sqrt(self.oc)[:, None, None],
        )

    def step(self, a, pop, q):
        c = self.c
        e = self.e
        pump = self.pumps * 10 ** (-c["pump_path_dB"] / 10)
        for j in range(6):
            a, pump, pop = self.fiber(a, j, pump, pop)
            if j == 3:
                a = e.J @ a
                if c["topology"] == "OC_CNT":
                    a, out = self.split(a)
                a, q, sa = absorber(a, q, c, self.dt)
                if c["topology"] == "CNT_OC":
                    a, out = self.split(a)
                a *= 10 ** (-c["splice_connector_dB"] / 20)
        a *= 10 ** (-c["hybrid_IL_dB"] / 20)
        a[:, 1] *= 10 ** (-c["hybrid_PDL_dB"] / 20)
        q = c["sa_modulation"] + (q - c["sa_modulation"]) * np.exp(
            -max(0, 1e12 / e.rep - self.n * self.dt) / c["sa_recovery_ps"]
        )
        return a, out, pop, q, sa


def aligned_residual(previous, current):
    p = np.sum(abs(previous) ** 2, axis=1)
    q = np.sum(abs(current) ** 2, axis=1)
    n = p.shape[1]
    row = np.arange(len(p))
    corr = ifft(fft(p, axis=1) * np.conj(fft(q, axis=1)), axis=1).real
    k = np.argmax(corr, axis=1)
    den = corr[row, (k - 1) % n] - 2 * corr[row, k] + corr[row, (k + 1) % n]
    fraction = np.divide(
        0.5 * (corr[row, (k - 1) % n] - corr[row, (k + 1) % n]),
        den,
        out=np.zeros(len(p)),
        where=den != 0,
    )
    shift = np.where(k < n / 2, k, k - n) + fraction
    shifted = ifft(
        fft(current, axis=-1)
        * np.exp(-2j * np.pi * np.fft.fftfreq(n)[None, None, :] * shift[:, None, None]),
        axis=-1,
    )
    error = np.linalg.norm(np.sum(abs(shifted) ** 2, axis=1) - p, axis=1) / np.maximum(
        np.linalg.norm(p, axis=1), 1e-100
    )
    return error, shift
