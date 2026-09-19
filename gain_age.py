"""Conditional EDF relaxation along a given rate history, not a cavity Jacobian.

Only accepted physical round trips count. Reset the age for a new parameter
point, or preserve it when resuming the same trajectory. Unknown prehistory
must not be inferred from a terminal tau. Fixed EDF cell identities required.
"""

import numpy as np


def advance(age, rates_b, round_trip_s):
    """Accumulate Gamma[cell] = sum B[cell,k] T_R on an unchanged EDF mesh."""
    age, rates_b = np.asarray(age, float), np.asarray(rates_b, float)
    if age.shape != rates_b.shape:
        raise ValueError("Gain age and rates must have identical cell shapes")
    if not (np.isfinite(age).all() and np.isfinite(rates_b).all()):
        raise ValueError("Nonfinite age or rate")
    if np.any(age < 0) or np.any(rates_b <= 0) or not np.isfinite(round_trip_s) or round_trip_s <= 0:
        raise ValueError("Positive physical rates and time, nonnegative age required")
    return age + rates_b * round_trip_s


def summary(age):
    minimum = float(np.min(age))
    return dict(gain_age_min=minimum, gain_memory_max=float(np.exp(-minimum)))
