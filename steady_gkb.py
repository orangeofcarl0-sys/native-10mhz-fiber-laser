"""Fully reorthogonalized Golub–Kahan basis and SVD trust-region solve."""
import time
import numpy as np
from scipy.optimize import brentq


def reorthogonalize(w, basis):
    w = w.copy()
    for _ in range(2):
        w -= basis @ (basis.T @ w)
    return w


def gkb(jv, jtv, b, maximum=32, radius=None):
    """Return U,V,JV,B; retain actual responses to audit inexact Jv.

    B is the nominal bidiagonal recurrence. Full reorthogonalization
    corrections are deliberately measured, not silently discarded.
    """
    beta = np.linalg.norm(b)
    if beta == 0:
        raise ValueError('A nonzero right hand side is required')
    begin = time.perf_counter()
    u = (b / beta)[:, None]
    first = jtv(u[:, 0])
    v = np.empty((len(first), 0))
    responses = []
    diagonal = []; subdiagonal = []; rows = []
    previous = None; streak = 0
    for k in range(maximum):
        w = first.copy() if k == 0 else jtv(u[:, k])
        if k:
            w -= subdiagonal[-1] * v[:, -1]
        w = reorthogonalize(w, v)
        alpha = np.linalg.norm(w)
        if alpha < 1e-14:
            break
        v = np.column_stack([v, w / alpha])
        av = jv(v[:, -1]); responses.append(av)
        w = reorthogonalize(av - alpha*u[:, k], u)
        beta_next = np.linalg.norm(w)
        diagonal.append(alpha); subdiagonal.append(beta_next)
        rows.append(dict(k=k+1, seconds=time.perf_counter()-begin))
        if beta_next < 1e-14:
            break
        u = np.column_stack([u, w/beta_next])
        if radius is not None and (k+1) % 8 == 0:
            small = np.zeros((k+2,k+1))
            for j in range(k+1):
                small[j,j]=diagonal[j];small[j+1,j]=subdiagonal[j]
            target=np.zeros(k+2);target[0]=beta
            prediction=svd_trust(small,target,radius)[1]
            gain=None if previous is None else (prediction-previous)/max(prediction,1e-100)
            streak=streak+1 if gain is not None and 0<=gain<.01 else 0
            rows[-1].update(prediction=prediction,marginal_gain=gain,low_gain_streak=streak)
            previous=prediction
            if streak>=2:
                rows[-1]['saturated']=True
                break
    count = v.shape[1]
    B = np.zeros((u.shape[1], count))
    for k in range(count):
        B[k, k] = diagonal[k]
        if k+1 < len(B):
            B[k+1, k] = subdiagonal[k]
    if not responses:
        raise ValueError('GKB has no nonzero descent direction')
    return u, v, np.column_stack(responses), B, rows


def svd_trust(matrix, target, radius):
    """Solve min ||target-matrix@y|| with ||y||<=radius, without normal LU."""
    if radius <= 0:
        raise ValueError('Positive radius required')
    p, sigma, qt = np.linalg.svd(matrix, full_matrices=False)
    rhs = p.T @ target
    def coefficients(lam):
        return np.divide(sigma*rhs, sigma*sigma+lam,
                         out=np.zeros_like(sigma), where=sigma*sigma+lam>1e-30)
    lam = 0.
    if np.linalg.norm(coefficients(0)) > radius:
        high = max(float(sigma[0]**2), 1.)
        while np.linalg.norm(coefficients(high)) > radius:
            high *= 4
        lam = brentq(lambda t: np.linalg.norm(coefficients(t))-radius,
                     0, high, xtol=1e-14, rtol=1e-12)
    y = qt.T @ coefficients(lam)
    residual = target-matrix@y
    prediction = .5*(target@target-residual@residual)
    return y, float(prediction), float(lam)
