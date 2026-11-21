"""
Timing of a single generation run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from quantbench.backends.base import Backend


@dataclass(frozen=True)
class Timing:
    ttft_ms: float
    tokens_per_s: float | None
    n_tokens: int


def time_generation(
    backend: Backend, prompt_ids: list[int], max_new_tokens: int
) -> Timing:
    """
    Run  one generation and measure TTFT and decode speed.
    """

    start = time.perf_counter()

    first = last = start

    n_tokens = 0

    for _ in backend.generate(prompt_ids, max_new_tokens):
        last = time.perf_counter()

        if n_tokens == 0:
            first = last

        n_tokens += 1

    if n_tokens == 0:
        raise ValueError("backend produced no tokens.")

    decode_s = last - first

    tokens_per_s = (n_tokens - 1) / decode_s if n_tokens > 1 and decode_s > 0 else None

    return Timing(
        ttft_ms=(first - start) * 1000, tokens_per_s=tokens_per_s, n_tokens=n_tokens
    )
