"""
Unit tests for Pydantic report schemas and validators.
Runs without GPU.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantbench.schema import (
    ModelResult,
    RunMetrics,
    Stat,
    Summary,
)


def _sample_run(
    ttft: float = 20.0, tps: float = 50.0, vram: float = 1500.0
) -> RunMetrics:
    return RunMetrics(ttft_ms=ttft, tokens_per_s=tps, n_tokens=128, peak_vram_mb=vram)


def _sample_summary(
    ttft: float = 20.0, tps: float = 50.0, vram: float = 1500.0
) -> Summary:
    return Summary(
        ttft_ms=Stat(median=ttft, p95=ttft, p5=ttft),
        tokens_per_s=Stat(median=tps, p95=tps, p5=tps),
        peak_vram_mb=Stat(median=vram, p95=vram, p5=vram),
    )


def test_strict_model_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunMetrics(
            ttft_ms=10.0, n_tokens=10, peak_vram_mb=100.0, unexpected_field="disallowed"
        )


def test_model_result_ok_validation() -> None:
    run = _sample_run()
    summary = _sample_summary()

    # Valid OK result
    res = ModelResult(
        model="test-model",
        format="fp16",
        engine="transformers",
        status="ok",
        baseline_vram_mb=500.0,
        runs=[run],
        summary=summary,
    )
    assert res.status == "ok"

    with pytest.raises(ValidationError, match="requires a summary"):
        ModelResult(
            model="test-model",
            format="fp16",
            engine="transformers",
            status="ok",
            baseline_vram_mb=500.0,
            runs=[run],
            summary=None,
        )

    # OK status missing baseline must fail
    with pytest.raises(ValidationError, match="requires baseline_vram_mb"):
        ModelResult(
            model="test-model",
            format="fp16",
            engine="transformers",
            status="ok",
            baseline_vram_mb=None,
            runs=[run],
            summary=summary,
        )


def test_model_result_summary_mismatch_detected() -> None:
    run = _sample_run(ttft=20.0)
    # Summary claims median is 99.0 while run has 20.0.
    fraudulent_summary = _sample_summary(ttft=99.0)
    with pytest.raises(ValidationError, match="does not match the raw runs"):
        ModelResult(
            model="test-model",
            format="fp16",
            engine="transformers",
            status="ok",
            baseline_vram_mb=500.0,
            runs=[run],
            summary=fraudulent_summary,
        )


def test_model_result_error_status_validation() -> None:
    # Error without message must fail
    with pytest.raises(ValidationError, match="requires error_message"):
        ModelResult(
            model="test-model",
            format="fp16",
            engine="transformers",
            status="error",
            error_message=None,
        )
    # Error status must not have runs or summary
    with pytest.raises(ValidationError, match="must have no runs"):
        ModelResult(
            model="test-model",
            format="fp16",
            engine="transformers",
            status="error",
            error_message="failed",
            runs=[_sample_run()],
        )
