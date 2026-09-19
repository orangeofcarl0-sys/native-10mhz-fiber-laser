"""Host interface compatible with the tested CPU batch engine."""

from gpu_setup import cp
from gpu_batch_core import BatchEngine as GPUCore
from python_engine import Engine
import numpy as np


class BatchEngine:
    def __init__(self, c, dt, n, pumps, oc):
        self.core = GPUCore(c, dt, n, pumps, oc)
        self.e = Engine(c, dt, n)
        self.pumps = np.asarray(pumps)
        self.oc = np.asarray(oc)

    def step(self, a, pop, q):
        self.core.pumps = cp.asarray(self.pumps)
        self.core.oc = cp.asarray(self.oc)
        result = self.core.step(cp.asarray(a), cp.asarray(pop), cp.asarray(q))
        self.last_gap = cp.asnumpy(self.core.last_gap)
        self.last_tau = cp.asnumpy(self.core.last_tau)
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
        self.core = GPUCore(c, dt, a.shape[-1], pumps, oc)
        self.a, self.pop, self.q = (cp.asarray(x).copy() for x in (a, pop, q))
        self.out = cp.zeros_like(self.a)
        self.failed = cp.zeros(len(a), dtype=bool)
        self.rounds = 0

    def step(self):
        a, out, pop, q, sa = self.core.step(
            self.a.copy(), self.pop.copy(), self.q.copy()
        )
        power = cp.sum(abs(a) ** 2, axis=1)
        energy = cp.sum(power, axis=1)
        spectrum = abs(cp.fft.fft(a, axis=-1)) ** 2
        frequency = abs(cp.fft.fftfreq(a.shape[-1]))
        spectral_edge = cp.sum(
            spectrum[:, :, frequency > 0.45], axis=(1, 2)
        ) / cp.maximum(cp.sum(spectrum, axis=(1, 2)), 1e-100)
        width = max(1, a.shape[-1] // 20)
        time_edge = (
            power[:, :width].sum(axis=1) + power[:, -width:].sum(axis=1)
        ) / cp.maximum(energy, 1e-100)
        finite = (
            cp.all(cp.isfinite(a), axis=(1, 2))
            & cp.all(cp.isfinite(pop), axis=1)
            & cp.isfinite(q)
        )
        self.failed |= ~finite | (spectral_edge > 1e-8) | (time_edge > 1e-6)
        good = ~self.failed
        self.a = cp.where(good[:, None, None], a, self.a)
        self.out = cp.where(good[:, None, None], out, self.out)
        self.pop = cp.where(good[:, None], pop, self.pop)
        self.q = cp.where(good, q, self.q)
        self.rounds += 1
        return cp.stack(
            [energy * self.core.dt, spectral_edge, time_edge, self.failed], axis=1
        )

    def run(self, rounds):
        """Return compact diagnostics [round, case, energy/edges/failure]."""
        return cp.asnumpy(cp.stack([self.step() for _ in range(rounds)]))

    def checkpoint(self):
        return {
            k: cp.asnumpy(getattr(self, k)) for k in ("a", "out", "pop", "q", "failed")
        }
