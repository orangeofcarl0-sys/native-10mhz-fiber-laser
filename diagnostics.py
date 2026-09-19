"""Scalar reference for translation and global-phase alignment."""

import numpy as np


def align(reference, current):
    p = np.sum(abs(reference) ** 2, axis=0)
    q = np.sum(abs(current) ** 2, axis=0)
    n = len(p)
    corr = np.fft.ifft(np.fft.fft(p) * np.conj(np.fft.fft(q))).real
    k = int(np.argmax(corr))
    den = corr[(k - 1) % n] - 2 * corr[k] + corr[(k + 1) % n]
    frac = 0.5 * (corr[(k - 1) % n] - corr[(k + 1) % n]) / den if den else 0
    shift = (k if k < n / 2 else k - n) + frac
    a = np.fft.ifft(
        np.fft.fft(current, axis=1) * np.exp(-2j * np.pi * np.fft.fftfreq(n) * shift),
        axis=1,
    )
    intensity = np.linalg.norm(np.sum(abs(a) ** 2, axis=0) - p) / max(
        np.linalg.norm(p), 1e-100
    )
    a *= np.exp(-1j * np.angle(np.vdot(reference, a)))
    field = np.linalg.norm(a - reference) / max(np.linalg.norm(reference), 1e-100)
    return float(intensity), float(field), float(shift)
