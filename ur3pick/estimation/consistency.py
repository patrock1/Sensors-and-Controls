"""
consistency.py
Is the filter telling the truth about its own uncertainty
"""
import numpy as np

def nees(err, P):
    """Normalised estimation error squared; E[NEES] = n for a consistent filter."""
    return float(err @ np.linalg.solve(P, err))


def mean_nees(errs, Ps):
    return float(np.mean([nees(e, P) for e, P in zip(errs, Ps)]))

def coverage_2sigma(errs, sigmas):
    return float(np.mean(np.abs(np.asarray(errs)) < 2 * np.asarray(sigmas)))

def verdict(mean_nees_value, n):
    r = mean_nees_value / n
    return "OVERCONFIDENT" if r > 1.2 else "conservative" if r < 0.85 else "consistent"