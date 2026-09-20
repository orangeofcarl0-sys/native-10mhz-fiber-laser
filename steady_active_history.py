"""Current-response greedy enrichment in the actual scaled-state trust metric."""
import numpy as np
from steady_hookstep import hookstep


class HistoryModel:
    """Two orthogonal compressions preserve state norms and residual norms."""
    def __init__(self, state_basis, responses, residual, base_count):
        self.q, self.state = np.linalg.qr(state_basis, mode='reduced')
        target = -residual / np.linalg.norm(residual)
        qo, ro = np.linalg.qr(np.column_stack([target, responses]), mode='reduced')
        if qo[:, 0] @ target < 0:
            ro[0] *= -1
        self.response = ro[:, 1:]
        self.beta = np.linalg.norm(residual)
        self.base_count = base_count
        self.cache = {}

    def solve(self, selected, radius):
        key = (tuple(sorted(selected)), float(radius))
        if key not in self.cache:
            indices = list(range(self.base_count)) + [self.base_count+i for i in key[0]]
            step, prediction, lam = hookstep(
                self.response[:, indices], self.state[:, indices], self.beta,
                radius, np.ones(self.state.shape[0]))
            self.cache[key] = (step, prediction, lam)
        return self.cache[key]

    def full_step(self, selected, radius):
        step, prediction, lam = self.solve(selected, radius)
        return self.q @ step, prediction, lam


def greedy_history(model, count, radius, maximum=4, stop_gain=.05):
    """Force a full audit ladder; separately record a prospective stopping policy.

    Selection uses predicted merit only, with chronological tie breaking.
    Neither direction slope nor cosine is an eligibility filter.
    """
    selected = []
    previous = model.solve([], radius)[1]
    rows = []
    policy_count = None
    for level in range(min(maximum, count)):
        trials = [dict(index=i, prediction=float(model.solve(selected+[i], radius)[1]))
                  for i in range(count) if i not in selected]
        winner = max(trials, key=lambda row: row['prediction'])
        gain = (winner['prediction']-previous)/max(abs(previous), 1e-100)
        if policy_count is None and gain < stop_gain:
            policy_count = level
        selected.append(winner['index'])
        rows.append(dict(selected=selected.copy(), prediction=winner['prediction'],
                         marginal_gain=float(gain), candidates=trials))
        previous = winner['prediction']
    return dict(ladder=rows, policy_count=len(rows) if policy_count is None else policy_count,
                policy_rule='reject first addition whose relative marginal gain is below 5%')
