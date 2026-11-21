"""
Unit tests for latency measurement using MockBackend.
"""

from __future__ import annotations

import pytest

from quantbench.backends.mock import MockBackend
from quantbench.measure.latency import time_generation


def test_time_generation_metrics() -> None:

    # 50 ms prefill delay, 10ms per token decode delay
    prefill_s = 0.050
    per_token_s = 0.010
    n_tokens = 6

    backend = MockBackend(prefill_s=prefill_s, per_token_s=per_token_s)
    prompt_ids = [1, 2, 3]

    timing = time_generation(backend, prompt_ids, max_new_tokens=n_tokens)

    assert timing.n_tokens == n_tokens

    # TTFT should be around 50ms (allow reasonable margin for OS thread scheduling)
    assert 40.0 <= timing.ttft_ms <= 90.0

    # Expected decode speed: (6-1) tokens / 0.050 s = 100 token/s
    assert timing.tokens_per_s is not None
    assert 70.0 <= timing.tokens_per_s <= 130.0


def test_time_generation_single_token() -> None:
    # When max_new_tokens is 1, tokens_per_s cannot be computed (decode time is 0)
    backend = MockBackend(prefill_s=0.010, per_token_s=0.010)
    timing = time_generation(backend, [1], max_new_tokens=1)
    assert timing.n_tokens == 1
    assert timing.tokens_per_s is None
    assert timing.ttft_ms > 0.0


def test_time_generation_zero_tokens_raises() -> None:
    backend = MockBackend()
    with pytest.raises(ValueError, match="produced no tokens"):
        time_generation(backend, [1], max_new_tokens=0)
