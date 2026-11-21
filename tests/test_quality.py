"""
Unit tests for perplexity calculation.
"""

from __future__ import annotations

import math

import pytest

from quantbench.backends.mock import MockBackend
from quantbench.measure.quality import compute_multi_perplexity, compute_perplexity


class ScoringMockBackend(MockBackend):
    """
    MockBackend with scoring capability returning deterministic log-probabilities.
    """

    supports_scoring: bool = True

    def __init__(self, fixed_log_prob: float) -> None:
        super().__init__()
        self._fixed_log_prob = fixed_log_prob

    def score(self, token_ids: list[int]) -> list[float]:
        # Return N - 1 log-probabilities
        return [self._fixed_log_prob] * (len(token_ids) - 1)


def test_compute_perplexity_perfect_prediction() -> None:
    # If log P = 0 for all tokens (probability = 1.0), PPL must be exp(0) = 1.0
    backend = ScoringMockBackend(fixed_log_prob=0.0)
    ppl = compute_perplexity(backend, [1, 2, 3, 4, 5])

    assert math.isclose(ppl, 1.0)


def test_compute_perplexity_known_value() -> None:
    # If mean log P = -2.0, PPL must be exp(2.0) ~= 7.389
    log_prob = -2.0
    backend = ScoringMockBackend(fixed_log_prob=log_prob)
    ppl = compute_perplexity(backend, [1, 2, 3, 4])

    assert math.isclose(ppl, math.exp(-log_prob))


def test_compute_perplexity_unsupported_backend() -> None:
    backend = MockBackend()
    with pytest.raises(ValueError, match="does not support scoring"):
        compute_perplexity(backend, [1, 2, 3])


def test_compute_multi_perplexity_breakdown_and_weighting() -> None:
    backend = ScoringMockBackend(fixed_log_prob=-1.0)
    prompts = {
        "short_text": [1, 2, 3],
        "long_text": [1, 2, 3, 4, 5],
    }
    overall_ppl, breakdown = compute_multi_perplexity(backend, prompts)

    assert math.isclose(overall_ppl, math.e)

    assert len(breakdown) == 2
    by_name = {b.prompt_name: b for b in breakdown}
    assert by_name["short_text"].n_tokens == 3
    assert math.isclose(by_name["short_text"].perplexity, math.e)
    assert by_name["long_text"].n_tokens == 5
    assert math.isclose(by_name["long_text"].perplexity, math.e)


def test_compute_multi_perplexity_empty_raises() -> None:
    backend = ScoringMockBackend(fixed_log_prob=-1.0)
    with pytest.raises(ValueError, match="must not be empty"):
        compute_multi_perplexity(backend, {})
