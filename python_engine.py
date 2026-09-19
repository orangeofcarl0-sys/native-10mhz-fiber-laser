"""Independent translation of the audited MATLAB map; no output normalization."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.fft import fft, ifft


def kerr(a, g):
    p = (a[0] + 1j * a[1]) / np.sqrt(2)
    m = (a[0] - 1j * a[1]) / np.sqrt(2)
    pp = abs(p) ** 2
    mm = abs(m) ** 2
    p *= np.exp(2j / 3 * g * (pp + 2 * mm))
    m *= np.exp(2j / 3 * g * (mm + 2 * pp))
    return np.array([(p + m) / np.sqrt(2), (p - m) / (1j * np.sqrt(2))])


def absorber(a, q, c, dt):
    P = np.sum(abs(a) ** 2, axis=0)
    rate = 1 / c["sa_recovery_ps"] + P / c["sa_saturation_energy_pJ"]
    eq = c["sa_modulation"] / (c["sa_recovery_ps"] * rate)
    decay = np.exp(-rate * dt)
    qm = np.empty(len(P))
    qmin = q
    # Exact variable-coefficient recurrence in short blocks; scalar fallback avoids underflow.
    for st in range(0, len(P), 16):
        sl = slice(st, st + 16)
        aa = decay[sl]
        bb = eq[sl] * (1 - aa)
        prod = np.cumprod(aa)
        if prod[-1] > 1e-200:
            after = prod * (q + np.cumsum(bb / prod))
            before = np.r_[q, after[:-1]]
            qm[sl] = eq[sl] + (before - eq[sl]) * np.sqrt(aa)
            q = after[-1]
        else:
            for j in range(st, min(st + 16, len(P))):
                qm[j] = eq[j] + (q - eq[j]) * np.sqrt(decay[j])
                q = eq[j] + (q - eq[j]) * decay[j]
    T = 1 - c["sa_nonsaturable"] - qm
    assert np.min(T) > 0
    return (
        a * np.sqrt(T),
        q,
        np.array(
            [
                np.sum(P * T) / np.sum(P),
                min(qmin, float(qm.min())),
                P.max(),
                P.sum() * dt,
            ]
        ),
    )


class Engine:
    def __init__(self, c, dt, n):
        self.c = c
        self.dt = dt
        self.n = n
        self.rep = 299792458 / (c["group_index"] * 20.42)
        self.w = np.fft.fftfreq(n, dt) * 2 * np.pi
        self.profile = 1 / (
            1
            + (
                2
                * self.w
                / (2 * np.pi * 299792.458 * c["gain_fwhm_nm"] / c["lambda_s_nm"] ** 2)
            )
            ** 2
        )
        self.hs = 6.62607015e-34 * 299792458 / (c["lambda_s_nm"] * 1e-9)
        self.hp = 6.62607015e-34 * 299792458 / (c["lambda_p_nm"] * 1e-9)
        self.ions = c["ion_density_m3"] * c["doped_area_m2"]
        self.ops = []
        for index, seg in enumerate(c["segments"]):
            L, b2, b3, g, beat, angle, dgd = seg
            limit = (
                c["max_step_m"]
                if index == 1
                else c.get("passive_step_m", c["max_step_m"])
            )
            steps = int(np.ceil(L / limit))
            dz = L / steps
            D = 1j * b2 * self.w**2 / 2 - 1j * b3 * self.w**3 / 6
            b = 1j * (beat + dgd * self.w) / 2
            half = np.exp(np.array([D + b, D - b]) * dz / 2)
            R = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            self.ops.append((steps, dz, half, R, g))
        J = np.eye(2, dtype=complex)
        for angle, ret in zip(
            c["pc_angles_rad"], c["pc_turns"] * c["pc_retardance_per_turn_rad"]
        ):
            R = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            J = R @ np.diag(np.exp(1j * np.array([-1, 1]) * ret / 2)) @ R.T @ J
        self.J = J

    def fiber(self, a, idx, pump, pop):
        steps, dz, half, R, gamma = self.ops[idx]
        phase = 0
        aud = []
        c = self.c
        if idx != 1:
            a = ifft(half * fft(R.T @ a, axis=1), axis=1)
            for j in range(steps):
                phase = max(phase, gamma * dz * np.sum(abs(a) ** 2, axis=0).max())
                a = kerr(a, gamma * dz)
                a = ifft((half if j == steps - 1 else half**2) * fft(a, axis=1), axis=1)
            return R @ a, pump, pop, phase, np.empty((0, 10))
        for j in range(steps):
            pin = pump
            sin = np.sum(abs(a) ** 2) * self.dt * 1e-12 * self.rep
            inv = pop[j]
            old = (
                np.sum(np.sum(abs(fft(a, axis=1)) ** 2, axis=0) * self.profile)
                * self.dt
                * 1e-12
                * self.rep
                / self.n
            )
            gain = (c["emission_s_m"] * inv - c["alpha_s_m"] * (1 - inv)) * self.profile
            hh = half * np.exp(gain * dz / 4)
            a = R @ ifft(hh * fft(R.T @ a, axis=1), axis=1)
            phase = max(phase, gamma * dz * np.sum(abs(a) ** 2, axis=0).max())
            a = kerr(a, gamma * dz)
            a = R @ ifft(hh * fft(R.T @ a, axis=1), axis=1)
            pmid = pump * np.exp(-c["alpha_p_m"] * (1 - inv) * dz / 2)
            new = (
                np.sum(np.sum(abs(fft(a, axis=1)) ** 2, axis=0) * self.profile)
                * self.dt
                * 1e-12
                * self.rep
                / self.n
            )
            smid = np.sqrt(old * new)
            A = (
                c["alpha_p_m"] * pmid / self.hp + c["alpha_s_m"] * smid / self.hs
            ) / self.ions
            B = (
                1 / c["upper_lifetime_s"]
                + (
                    c["alpha_p_m"] * pmid / self.hp
                    + (c["alpha_s_m"] + c["emission_s_m"]) * smid / self.hs
                )
                / self.ions
            )
            eq = A / B
            nex = (
                inv
                if c["gain_mode"] == "frozen"
                else (
                    eq
                    if c["gain_mode"] == "rate_equilibrium"
                    else inv + (eq - inv) * (-np.expm1(-B / self.rep))
                )
            )
            pop[j] = nex
            pump *= np.exp(-c["alpha_p_m"] * (1 - inv) * dz)
            sout = np.sum(abs(a) ** 2) * self.dt * 1e-12 * self.rep
            aud.append(
                [
                    inv,
                    nex,
                    eq,
                    B,
                    (pin - pump) / self.hp,
                    (sout - sin) / self.hs,
                    self.ions * dz * inv / c["upper_lifetime_s"],
                    self.ions * dz * (nex - inv) * self.rep,
                    pin,
                    pump,
                ]
            )
        return a, pump, pop, phase, np.array(aud)

    def step(self, a, pop, q):
        c = self.c
        pump = c["pump_ld_W"] * 10 ** (-c["pump_path_dB"] / 10)
        ledger = []
        local = []
        gain = None
        energy = lambda x: np.sum(abs(x) ** 2) * self.dt
        for j in range(6):
            before = energy(a)
            a, pp, nn, phase, ga = self.fiber(a, j, pump, pop)
            if j == 1:
                pump = pp
                pop = nn
                gain = ga
            ledger.append([j + 1, before, energy(a), int(j == 1)])
            P = np.sum(abs(a) ** 2, axis=0)
            t = (np.arange(self.n) - self.n / 2) * self.dt
            ct = sum(t * P) / sum(P)
            var = sum((t - ct) ** 2 * P) / sum(P)
            deriv = ifft(1j * self.w * fft(a, axis=1), axis=1)
            chirp = np.imag(
                np.sum(np.sum(np.conj(a) * deriv, axis=0) * (t - ct))
                * self.dt
                / energy(a)
            ) / max(var, 1e-100)
            local.append([energy(a), P.max(), np.sqrt(var), chirp, phase])
            if j == 3:
                before = energy(a)
                a = self.J @ a
                ledger.append([7, before, energy(a), 0])
                if c["topology"] == "OC_CNT":
                    a, out, le = self.oc(a)
                    ledger.extend(le)
                before = energy(a)
                a, q, sa = absorber(a, q, c, self.dt)
                ledger.append([8, before, energy(a), 2])
                if c["topology"] == "CNT_OC":
                    a, out, le = self.oc(a)
                    ledger.extend(le)
                before = energy(a)
                a *= 10 ** (-c["splice_connector_dB"] / 20)
                ledger.append([9, before, energy(a), 2])
        before = energy(a)
        a *= 10 ** (-c["hybrid_IL_dB"] / 20)
        a[1] *= 10 ** (-c["hybrid_PDL_dB"] / 20)
        ledger.append([12, before, energy(a), 2])
        q = c["sa_modulation"] + (q - c["sa_modulation"]) * np.exp(
            -max(0, 1e12 / self.rep - self.n * self.dt) / c["sa_recovery_ps"]
        )
        return a, out, pop, q, np.array(ledger), gain, sa, np.array(local)

    def oc(self, a):
        c = self.c
        E = np.sum(abs(a) ** 2) * self.dt
        a = a * 10 ** (-c["oc_excess_dB"] / 20)
        Ea = np.sum(abs(a) ** 2) * self.dt
        out = np.sqrt(c["output_power_fraction"]) * a
        a = np.sqrt(1 - c["output_power_fraction"]) * a
        return (
            a,
            out,
            [
                [10, E, Ea, 2],
                [11, Ea, (np.sum(abs(a) ** 2) + np.sum(abs(out) ** 2)) * self.dt, 0],
            ],
        )
