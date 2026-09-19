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
        return tuple(cp.asnumpy(x) for x in result)
