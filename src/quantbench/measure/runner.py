"""
Runs the benchmark protocol for one backend and builds a ModelResult.
"""

from __future__ import annotations

import logging
import time

from quantbench.backends.base import Backend, BackendOOMError
from quantbench.measure.latency import time_generation
from quantbench.measure.quality import compute_multi_perplexity
from quantbench.measure.vram import GpuSampler, measure_baseline_vram_mb
from quantbench.schema import (
    ModelResult,
    PerplexityBreakdown,
    RunMetrics,
    Stat,
    Summary,
)
from quantbench.stats import median, p5, p95

logger = logging.getLogger(__name__)


def run_benchmark(
    backend: Backend,
    *,
    model: str,
    format: str,
    engine: str,
    prompt_ids: list[int],
    eval_prompts: dict[str, list[int]] | None = None,
    max_new_tokens: int,
    warmup_runs: int = 4,
    timed_runs: int = 5,
    cooldown_s: float = 0.0,
    raise_errors: bool = False,
) -> ModelResult:

    baseline_mb, _spread_mb = measure_baseline_vram_mb()
    identity = {
        "model": model,
        "format": format,
        "engine": engine,
        "baseline_vram_mb": baseline_mb,
    }

    try:
        backend.load()

        warmup: list[RunMetrics] = []

        for _ in range(warmup_runs):
            warmup.append(_timed_run(backend, prompt_ids, max_new_tokens))

        runs: list[RunMetrics] = []

        for _ in range(timed_runs):
            runs.append(_timed_run(backend, prompt_ids, max_new_tokens))
            time.sleep(cooldown_s)

        perplexity: float | None = None
        breakdown: list[PerplexityBreakdown] = []

        if backend.supports_scoring and eval_prompts:
            perplexity, breakdown = compute_multi_perplexity(backend, eval_prompts)

    except BackendOOMError as oom_error:
        logger.warning("%s/%s/%s: out of memory: %s", model, format, engine, oom_error)
        return ModelResult(status="oom", **identity)

    except Exception as e:
        logger.exception("%s/%s/%s: benchmark failed", model, format, engine)

        if raise_errors:
            raise

        return ModelResult(
            status="error", error_message=f"{type(e).__name__}: {e}", **identity
        )

    finally:
        _unload_quietly(backend)

    return ModelResult(
        status="ok",
        runs=runs,
        warmup=warmup,
        summary=_summarize(runs),
        perplexity=perplexity,
        perplexity_breakdown=breakdown,
        **identity,
    )


def _timed_run(
    backend: Backend, prompt_ids: list[int], max_new_tokens: int
) -> RunMetrics:

    with GpuSampler() as sampler:
        timing = time_generation(backend, prompt_ids, max_new_tokens)

    return RunMetrics(
        ttft_ms=timing.ttft_ms,
        tokens_per_s=timing.tokens_per_s,
        n_tokens=timing.n_tokens,
        peak_vram_mb=sampler.peak_vram_mb,
        gpu_clock_mhz=sampler.median_clock_mhz,
        gpu_temp_c=sampler.max_temp_c,
        gpu_util_pct=sampler.mean_util_pct,
    )


def _stat(values: list[float]) -> Stat:
    return Stat(median=median(values), p95=p95(values), p5=p5(values))


def _summarize(runs: list[RunMetrics]) -> Summary:
    speeds = [r.tokens_per_s for r in runs if r.tokens_per_s is not None]

    return Summary(
        ttft_ms=_stat([r.ttft_ms for r in runs]),
        tokens_per_s=_stat(speeds) if speeds else None,
        peak_vram_mb=_stat([r.peak_vram_mb for r in runs]),
    )


def _unload_quietly(backend: Backend) -> None:

    try:
        backend.unload()
    except Exception:
        logger.warning("unload failed", exc_info=True)
