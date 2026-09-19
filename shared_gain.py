"""Separated temporal windows in ONE cavity, sharing one EDF population.

Average-power gain approximation; windows must be separated by many CNT recovery
times. This does not resolve within-round-trip population depletion or ASE.
"""

import numpy as np
from scipy.fft import fft, ifft
from batch_engine import BatchEngine, kerr


class SharedGainEngine(BatchEngine):
    def __init__(self, c, dt, n, pump, oc, windows=2):
        super().__init__(c, dt, n, [pump], np.full(windows, oc))

    def fiber(self, a, index, pump, pop):
        if index != 1:
            return super().fiber(a, index, pump, pop)
        if pop.shape != (1, self.e.ops[1][0]):
            raise ValueError("All windows require ONE longitudinal inversion array")
        e, c = self.e, self.c
        steps, dz, half, rotation, gamma = e.ops[1]
        gaps, taus = [], []
        scale = self.dt * 1e-12 * e.rep / e.n
        for j in range(steps):
            inv = pop[0, j].copy()
            f0 = fft(rotation.T @ a, axis=-1)
            old = np.sum(abs(f0) ** 2 * e.profile, axis=(1, 2)) * scale
            gain = (c["emission_s_m"] * inv - c["alpha_s_m"] * (1 - inv)) * e.profile
            hh = half * np.exp(gain * dz / 4)
            a = rotation @ ifft(hh * f0, axis=-1)
            a = kerr(a, gamma * dz)
            f1 = hh * fft(rotation.T @ a, axis=-1)
            a = rotation @ ifft(f1, axis=-1)
            new = np.sum(abs(f1) ** 2 * e.profile, axis=(1, 2)) * scale
            # Sum the midpoint powers, not independent population updates.
            signal = np.sqrt(old * new).sum()
            pmid = pump[0] * np.exp(-c["alpha_p_m"] * (1 - inv) * dz / 2)
            A = (c["alpha_p_m"] * pmid / e.hp + c["alpha_s_m"] * signal / e.hs) / e.ions
            B = (
                1 / c["upper_lifetime_s"]
                + (
                    c["alpha_p_m"] * pmid / e.hp
                    + (c["alpha_s_m"] + c["emission_s_m"]) * signal / e.hs
                )
                / e.ions
            )
            if c.get("gain_mode", "dynamic") != "frozen":
                pop[0, j] = inv + (A / B - inv) * (-np.expm1(-B / e.rep))
            pump *= np.exp(-c["alpha_p_m"] * (1 - inv) * dz)
            gaps.append(abs(inv - A / B))
            taus.append(1 / B)
        self.last_gap = np.array([max(gaps)])
        self.last_tau = np.array([max(taus)])
        return a, pump, pop


def satellite_windows(primary, energy_ratio):
    """Remote replica, without renormalizing the primary; ratio is energy ratio."""
    if energy_ratio < 0:
        raise ValueError("Energy ratio must be nonnegative")
    return np.stack([primary, np.sqrt(energy_ratio) * primary])


def integrated_stokes(a):
    """Normalized integrated Stokes, S3=+2 Im(Ex Ey*), matching existing convention."""
    x, y = a[..., 0, :], a[..., 1, :]
    energy = np.sum(abs(x) ** 2 + abs(y) ** 2, axis=-1)
    cross = np.sum(x * y.conj(), axis=-1)
    values = np.stack(
        [np.sum(abs(x) ** 2 - abs(y) ** 2, axis=-1), 2 * cross.real, 2 * cross.imag],
        axis=-1,
    )
    return values / np.maximum(energy[..., None], 1e-100)


def close_satellite(primary, dt, delay_ps, energy_ratio, phase=0.0):
    """Coherent same-window perturbation; caller must check wraparound/edges.

    The ratio describes the added replica alone; interference changes total energy.
    """
    if energy_ratio < 0:
        raise ValueError("Energy ratio must be nonnegative")
    shifted = np.fft.ifft(
        np.fft.fft(primary, axis=-1)
        * np.exp(-2j * np.pi * np.fft.fftfreq(primary.shape[-1], dt) * delay_ps),
        axis=-1,
    )
    return primary + np.sqrt(energy_ratio) * np.exp(1j * phase) * shifted
