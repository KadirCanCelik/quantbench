"""
Report schema shared by the measurement and report layers.

It lives at the package root so that quantbench.measure ( which writes reports)
and quantbench.report ( which reads them) can both depend on it without depending
on each other.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from quantbench.stats import median, p5, p95


class StrictModel(BaseModel):
    """
    Base for all schema models.

    Unknown fields are rejected so that a typo or a report written
    by a different schema version fails loudly instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class Environment(StrictModel):
    """
    Hardware and software the benchmark ran on.
    """

    gpu_name: str
    driver_version: str
    vram_total_mb: int
    power_limit_w: float | None = None
    python_version: str
    library_versions: dict[str, str] = {}
    cpu_model: str
    cpu_threads: int


class RunMetrics(StrictModel):
    """
    Raw measurements from a single timed run (warmup runs are no stored).
    """

    ttft_ms: float
    tokens_per_s: float | None = None
    n_tokens: int
    peak_vram_mb: float
    gpu_clock_mhz: float | None = None
    gpu_temp_c: float | None = (
        None  # Maximum temperature seen during the run (throttling indicator).
    )
    gpu_util_pct: float | None = None


class Stat(StrictModel):
    """
    Median and 95th percentile of one metric across timed runs.
    """

    median: float
    p95: float
    p5: float


class Summary(StrictModel):
    """
    Statistics derived from the raw runs. Raw runs are the source of truth.
    """

    ttft_ms: Stat
    tokens_per_s: Stat | None = None
    peak_vram_mb: Stat


class PerplexityBreakdown(StrictModel):
    """
    Per-prompt perplexity measurement and evaluated token count.
    """

    prompt_name: str
    n_tokens: int
    perplexity: float


class ModelResult(StrictModel):
    """
    Result for one (model, format, engine) combination.
    """

    model: str
    format: str
    engine: str
    status: Literal["ok", "oom", "error"]
    baseline_vram_mb: float | None = None
    runs: list[RunMetrics] = []
    warmup: list[RunMetrics] = []
    summary: Summary | None = None
    #: None when the backend does not support scoring or the run failed.
    perplexity: float | None = None
    perplexity_breakdown: list[PerplexityBreakdown] = []
    error_message: str | None = None
    model_revision: str | None = None
    order_index: int | None = None
    start_temp_c: float | None = None
    repeat_index: int | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelResult:
        if self.status == "ok":
            if not self.runs:
                raise ValueError("status 'ok' requires at least one run.")

            if self.summary is None:
                raise ValueError("status 'ok' requires a summary.")

            if self.baseline_vram_mb is None:
                raise ValueError("status 'ok' requires baseline_vram_mb.")

            if self.perplexity is not None and (
                math.isnan(self.perplexity)
                or math.isinf(self.perplexity)
                or self.perplexity < 1.0
            ):
                raise ValueError(
                    f"perplexity must be a finite number >= 1.0 got {self.perplexity}"
                )

            self._check_summary_matches_runs(self.summary)

        else:
            if (
                self.runs
                or self.warmup
                or self.summary is not None
                or self.perplexity is not None
                or self.perplexity_breakdown
            ):
                raise ValueError(
                    f"status '{self.status}' must have no runs, summary, perplexity or breakdown."
                )

            if self.status == "error" and not self.error_message:
                raise ValueError("status 'error' requires error_message")

        return self

    def _check_summary_matches_runs(self, summary: Summary) -> None:
        """
        Reject a summary that disagrees with the raw runs.
        """

        expected = {
            "ttft_ms": [r.ttft_ms for r in self.runs],
            "peak_vram_mb": [r.peak_vram_mb for r in self.runs],
        }

        speeds = [r.tokens_per_s for r in self.runs if r.tokens_per_s is not None]
        if speeds:
            expected["tokens_per_s"] = speeds
        elif summary.tokens_per_s is not None:
            raise ValueError("summary has tokens_per_s but no run has one")

        for name, values in expected.items():
            stat = getattr(summary, name)
            if stat is None:
                raise ValueError(f"summary is missing {name}")
            if not (
                math.isclose(stat.median, median(values), rel_tol=1e-9)
                and math.isclose(stat.p95, p95(values), rel_tol=1e-9)
                and math.isclose(stat.p5, p5(values), rel_tol=1e-9)
            ):
                raise ValueError(f"summary.{name} does not match the raw runs")


class Report(StrictModel):
    """
    Top-level benchmark report: one environment, many results.
    """

    schema_version: int = 1
    environment: Environment
    results: list[ModelResult]
