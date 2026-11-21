from __future__ import annotations

import math
from collections.abc import Mapping

from quantbench.backends.base import Backend
from quantbench.schema import PerplexityBreakdown


def compute_perplexity(backend: Backend, token_ids: list[int]) -> float:
    """
    Compute perplexity of token_ids using backend's score() method.
    """

    if not backend.supports_scoring:
        raise ValueError(f"Backend {backend.name} does not support scoring.")

    log_probs = backend.score(token_ids)

    if not log_probs:
        raise ValueError("score() returned an empty log-probabilities list.")

    mean_log_prob = sum(log_probs) / len(log_probs)

    return float(math.exp(-mean_log_prob))


def compute_multi_perplexity(
    backend: Backend, eval_prompts: Mapping[str, list[int]]
) -> tuple[float, list[PerplexityBreakdown]]:
    """
    Compute overall token-weighted perplexity and per-prompt breakdown across multiple prompts.
    """
    if not backend.supports_scoring:
        raise ValueError(f"Backend {backend.name} does not support scoring.")
    if not eval_prompts:
        raise ValueError("eval_prompts must not be empty.")
    breakdown: list[PerplexityBreakdown] = []
    total_log_probs_sum = 0.0
    total_predicted_tokens = 0
    for name, token_ids in eval_prompts.items():
        if len(token_ids) < 2:
            continue
        log_probs = backend.score(token_ids)
        if not log_probs:
            continue
        n_pred = len(log_probs)
        prompt_mean_log = sum(log_probs) / n_pred
        prompt_ppl = float(math.exp(-prompt_mean_log))
        breakdown.append(
            PerplexityBreakdown(
                prompt_name=name,
                n_tokens=len(token_ids),
                perplexity=prompt_ppl,
            )
        )
        total_log_probs_sum += sum(log_probs)
        total_predicted_tokens += n_pred
    if total_predicted_tokens == 0:
        raise ValueError("No tokens were successfully evaluated for perplexity.")
    overall_ppl = float(math.exp(-(total_log_probs_sum / total_predicted_tokens)))
    return overall_ppl, breakdown
