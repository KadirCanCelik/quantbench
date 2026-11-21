"""
Unit tests for configuration loading and validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantbench.config import BenchmarkConfig, ModelSpec
from quantbench.orchestrator import select_by_name


def test_model_spec_revision_regex() -> None:
    # Exactly 40 lowercase hexadecimal characters must pass
    valid_hash = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    spec = ModelSpec(name="test", repo="test/repo", revision=valid_hash)
    assert spec.revision == valid_hash

    # Invalid: 39 characters (too short)
    with pytest.raises(ValidationError):
        ModelSpec(name="test", repo="test/repo", revision=valid_hash[:-1])

    # Invalid: non-hex characters (contains 'z')
    invalid_hash = "060db6499f32faf8b98477b0a26969ef7d8b998z"
    with pytest.raises(ValidationError):
        ModelSpec(name="test", repo="test/repo", revision=invalid_hash)


def test_benchmark_config_defaults() -> None:
    valid_hash = "a" * 40
    data = {
        "models": [{"name": "m1", "repo": "r1", "revision": valid_hash}],
        "variants": [{"name": "v1", "backend": "b1", "format": "f1", "engine": "e1"}],
    }

    cfg = BenchmarkConfig.model_validate(data)
    assert len(cfg.models) == 1
    assert len(cfg.variants) == 1

    # Default run settings must be populated automatically
    assert cfg.run.prompt_tokens == 128
    assert cfg.run.max_new_tokens == 128
    assert cfg.run.warmup_runs == 4
    assert cfg.run.timed_runs == 5
    assert cfg.run.repeats == 3


def test_benchmark_config_missing_required_fields() -> None:
    # Missing models list must fail
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate({"variants": []})


def test_select_by_name_filtering() -> None:
    models = [
        ModelSpec(name="m1", repo="r1", revision="a" * 40),
        ModelSpec(name="m2", repo="r2", revision="b" * 40),
    ]

    # None filter returns everything
    assert select_by_name(models, None, "model") == models

    # Filter single item
    selected = select_by_name(models, ["m1"], "model")
    assert len(selected) == 1
    assert selected[0].name == "m1"

    # Unknown model name must raise ValueError
    with pytest.raises(ValueError, match="unknown model"):
        select_by_name(models, ["non_existent"], "model")
