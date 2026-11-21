"""
Loading and validating the benchmark configuration.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from quantbench.schema import StrictModel


class ModelSpec(StrictModel):
    """
    A model to benchmark, pinned to an exact Hugging Face revision.
    """

    name: str
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # Full 40-character commit hash.


class VariantSpec(StrictModel):
    name: str
    backend: str
    format: str
    engine: str


class RunSettings(StrictModel):
    """
    Measurement protocol parameters.
    """

    prompt_file: str = "data/prompt.txt"
    prompt_tokens: int = 128
    max_new_tokens: int = 128
    warmup_runs: int = 4
    timed_runs: int = 5
    pause_between_jobs: float = 15.0
    timeout_s: int = 1200
    repeats: int = 3
    eval_files: list[str] = Field(
        default_factory=lambda: [
            "data/prompt.txt",
            "data/code.txt",
            "data/reasoning.txt",
            "data/nature.txt",
        ]
    )


class BenchmarkConfig(StrictModel):
    models: list[ModelSpec]
    variants: list[VariantSpec]
    run: RunSettings = RunSettings()


def load_config(path: Path) -> BenchmarkConfig:
    """
    Read and validate the benchmark config file.
    """
    with path.open() as f:
        data = yaml.safe_load(f)

    return BenchmarkConfig.model_validate(data)


def load_models(path: Path) -> list[ModelSpec]:
    return load_config(path).models
