"""Matrix-free local multipliers of a verified, fixed-grid real return map.

Caller supplies nondimensional real coordinates (field real/imaginary components,
longitudinal inversion, CNT state) and a gauge-fixed one-period return map.
For a period-p orbit the map must compose p round trips. No accelerated gain map.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs


def multipliers(
    return_map, state, neutral_vectors, count=4, epsilon=1e-6, residual_tolerance=1e-7
):
    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("State must be a finite, scaled real vector")
    residual = np.linalg.norm(return_map(state.copy()) - state) / max(
        np.linalg.norm(state), 1.0
    )
    if not np.isfinite(residual) or residual > residual_tolerance:
        raise ValueError(f"Base orbit is not converged: residual={residual:g}")
    neutral = np.asarray(neutral_vectors, dtype=float).reshape(len(state), -1)
    if neutral.shape[1]:
        u, s, _ = np.linalg.svd(neutral, full_matrices=False)
        basis = u[:, s > max(s.max(), 1e-100) * 1e-12]
    else:
        basis = neutral

    def project(v):
        return v - basis @ (basis.T @ v)

    def derivative(v):
        v = project(v)
        norm = np.linalg.norm(v)
        if norm < 1e-100:
            return np.zeros_like(v)
        h = epsilon / norm
        return project(
            (return_map(state + h * v) - return_map(state - h * v)) / (2 * h)
        )

    if not 0 < count < len(state) - 1 or epsilon <= 0:
        raise ValueError("Invalid Arnoldi size or finite-difference step")
    operator = LinearOperator((len(state), len(state)), matvec=derivative, dtype=float)
    values = eigs(
        operator,
        k=count,
        which="LM",
        return_eigenvectors=False,
        v0=np.random.default_rng(0).normal(size=len(state)),
        tol=1e-9,
    )
    return values
