"""
Summary statistics shared by the measurement and report layers.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def median(values: Sequence[float]) -> float:
    """
    Return the median of a non-empty sequence.
    """

    if not values:
        raise ValueError("median() requires at least one value.")

    return statistics.median(values)


def p95(values: Sequence[float]) -> float:
    """
    Return the 95th percentile using linear interpolation.

    Uses the "inclusive" method, which matches NumPy's default. With few runs
    (e.g. 5) the result is close to the maximum; that is expected.
    """

    if not values:
        raise ValueError("p95() requires at least one value.")

    if len(values) == 1:
        return float(values[0])

    return statistics.quantiles(values, n=100, method="inclusive")[94]


def p5(values: Sequence[float]) -> float:
    """
    Return the 5th percentile using linear interpolation.

    For tokens/s this gives the worst-case tail (slowest runs)
    """
    if not values:
        raise ValueError("p5() requires at least one value.")

    if len(values) == 1:
        return float(values[0])

    return statistics.quantiles(values, n=100, method="inclusive")[4]
