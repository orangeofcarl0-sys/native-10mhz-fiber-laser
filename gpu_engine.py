"""Host interface compatible with the tested CPU batch engine."""

from gpu_setup import cp
from gpu_batch_core import BatchEngine as GPUCore
from spectral_engine import SpectralEngine
from python_engine import Engine
import numpy as np


def make_core(c, dt, n, pumps, oc):
    if c.get("gpu_pipeline", "spectral") == "reference":
        return GPUCore(c, dt, n, pumps, oc)
    if c.get("gpu_pipeline", "spectral") != "spectral":
        raise ValueError("gpu_pipeline must be spectral or reference")
    return SpectralEngine(c, dt, n, pumps, oc, gpu=True)


class BatchEngine:
    def __init__(self, c, dt, n, pumps, oc):
        self.core = make_core(c, dt, n, pumps, oc)
        self.e = Engine(c, dt, n)
        self.pumps = np.asarray(pumps)
        self.oc = np.asarray(oc)

    def step(self, a, pop, q):
        self.core.pumps = cp.asarray(self.pumps)
        self.core.oc = cp.asarray(self.oc)
        result = self.core.step(cp.asarray(a), cp.asarray(pop), cp.asarray(q))
        self.last_gap = cp.asnumpy(self.core.last_gap)
        self.last_tau = cp.asnumpy(self.core.last_tau)
        if hasattr(self.core, "rates_b"):
            self.rates_b = cp.asnumpy(self.core.rates_b)
        result = tuple(cp.asnumpy(x) for x in result)
        if not all(np.isfinite(x).all() for x in result):
            raise FloatingPointError("Nonfinite GPU state or invalid CNT transmission")
        return result


class ResidentEngine:
    """FP64 fixed-grid screening; no full-field host transfer between round trips.

    Failed cases retain their last clean state. Flags are permanent, and must be
    inspected before interpreting results. Adaptive replay remains a CPU feature.
    """

    def __init__(self, c, dt, a, pop, q, pumps, oc):
        self.core = make_core(c, dt, a.shape[-1], pumps, oc)
        self.a, self.pop, self.q = (cp.asarray(x).copy() for x in (a, pop, q))
        self.spectrum = cp.fft.fft(self.a, axis=-1)
        self.frequency_edge = abs(cp.fft.fftfreq(a.shape[-1])) > 0.45
        self.out = cp.zeros_like(self.a)
        self.failed = cp.zeros(len(a), dtype=bool)
        self.completed = cp.zeros(len(a), dtype=cp.int64)
        self.rounds = 0

    def step(self):
        if isinstance(self.core, SpectralEngine):
            f, out, pop, q, sa = self.core.step_spectrum(
                self.spectrum.copy(), self.pop.copy(), self.q.copy()
            )
            a = self.core.inverse(f)
        else:
            a, out, pop, q, sa = self.core.step(
                self.a.copy(), self.pop.copy(), self.q.copy()
            )
            f = cp.fft.fft(a, axis=-1)
        power = cp.sum(abs(a) ** 2, axis=1)
        energy = cp.sum(power, axis=1)
        spectrum = abs(f) ** 2
        spectral_edge = cp.sum(
            spectrum[:, :, self.frequency_edge], axis=(1, 2)
        ) / cp.maximum(cp.sum(spectrum, axis=(1, 2)), 1e-100)
        width = max(1, a.shape[-1] // 20)
        time_edge = (
            power[:, :width].sum(axis=1) + power[:, -width:].sum(axis=1)
        ) / cp.maximum(energy, 1e-100)
        finite = (
            cp.all(cp.isfinite(a), axis=(1, 2))
            & cp.all(cp.isfinite(pop), axis=1)
            & cp.isfinite(q)
            & cp.all(cp.isfinite(out), axis=(1, 2))
        )
        self.failed |= ~finite | (spectral_edge > 1e-8) | (time_edge > 1e-6)
        good = ~self.failed
        self.completed += good
        self.a = cp.where(good[:, None, None], a, self.a)
        self.spectrum = cp.where(good[:, None, None], f, self.spectrum)
        self.out = cp.where(good[:, None, None], out, self.out)
        self.pop = cp.where(good[:, None], pop, self.pop)
        self.q = cp.where(good, q, self.q)
        self.rounds += 1
        output_power = cp.sum(abs(out) ** 2, axis=1)
        local_peaks = cp.sum(
            (output_power[:, 1:-1] > output_power[:, :-2])
            & (output_power[:, 1:-1] > output_power[:, 2:])
            & (output_power[:, 1:-1] > 0.1 * output_power.max(axis=1)[:, None]),
            axis=1,
        )
        return cp.stack(
            [
                energy * self.core.dt,
                spectral_edge,
                time_edge,
                self.failed,
                output_power.sum(axis=1) * self.core.dt,
                local_peaks,
                self.core.last_gap,
            ],
            axis=1,
        )

    def run(self, rounds):
        """Return [round, case, intracavity E/edges/failure/output E/rough peaks/gap].

        Rough peaks omit prominence and plateau handling; never a certificate.
        """
        if rounds < 1:
            raise ValueError("rounds must be positive")
        return cp.asnumpy(cp.stack([self.step() for _ in range(rounds)]))

    def checkpoint(self):
        return {
            k: cp.asnumpy(getattr(self, k))
            for k in ("a", "out", "pop", "q", "failed", "completed")
        }
