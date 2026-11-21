"""
Unit tests for summary statistics (median, p5, p95).
Runs without GPU.
"""

from __future__ import annotations

import math

import pytest

from quantbench.stats import median, p5, p95


def test_median_odd_and_even() -> None:
    assert math.isclose(median([1.0, 3.0, 2.0]), 2.0)
    assert math.isclose(median([1.0, 2.0, 3.0, 4.0]), 2.5)


def test_single_element() -> None:
    val = 34.0
    assert math.isclose(median([val]), val)
    assert math.isclose(p95([val]), val)
    assert math.isclose(p5([val]), val)


def test_empty_sequence_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        median([])
    with pytest.raises(ValueError, match="at least one value"):
        p95([])
    with pytest.raises(ValueError, match="at least one value"):
        p5([])


def test_percentiles_order() -> None:

    values = [float(x) for x in range(1, 101)]
    m = median(values)
    high = p95(values)
    low = p5(values)

    assert low < m < high
    assert math.isclose(low, 5.95)  # 5th percentile of 1..100 inclusive is 5.95
    assert math.isclose(high, 95.05)  # 95th percentile of 1..100 inclusive is 95.05
