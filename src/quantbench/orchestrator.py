"""
Runs the benchmark matrix. One worker process per (model, variant) pair.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from transformers import AutoTokenizer

from quantbench.config import BenchmarkConfig, ModelSpec, VariantSpec
from quantbench.job import Job
from quantbench.measure.environment import collect_environment
from quantbench.measure.vram import read_gpu_temperature_c
from quantbench.schema import Environment, ModelResult, Report

logger = logging.getLogger(__name__)


def run_matrix(
    config: BenchmarkConfig,
    *,
    models_dir: Path = Path("models"),
    work_dir: Path = Path(".tmp/jobs"),
    results_dir: Path = Path("results"),
    only_models: list[str] | None = None,
    only_variants: list[str] | None = None,
    reverse: bool = False,
    repeats: int | None = None,
) -> Path:
    """
    Run the selected (model, variant) pairs for several passes, writing one JSON report.
    """
    models = select_by_name(config.models, only_models, "model")
    variants = select_by_name(config.variants, only_variants, "variant")
    base = [(m, v) for m in models for v in variants]
    passes = repeats if repeats is not None else config.run.repeats

    schedule: list[tuple[int, ModelSpec, VariantSpec]] = []

    for repeat in range(passes):
        backwards = reverse != (repeat % 2 == 1)
        for model, variant in reversed(base) if backwards else base:
            schedule.append((repeat, model, variant))

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"report-{datetime.now():%Y%m%d-%H%M%S}.json"
    environment = collect_environment()

    results: list[ModelResult] = []

    for index, (repeat, model, variant) in enumerate(schedule):
        if index > 0:
            time.sleep(config.run.pause_between_jobs)

        results.append(
            _run_job(model, variant, index, repeat, config, models_dir, work_dir)
        )

        # Rewritten after every job, so a crash loses at most the running job.

        _write_report(path, environment, results)

    logger.info("report written to %s", path)
    return path


def _write_report(
    path: Path, environment: Environment, results: list[ModelResult]
) -> None:
    report = Report(environment=environment, results=results)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(report.model_dump_json(indent=2))
    tmp.replace(path)


def select_by_name(items: list, names: list[str] | None, kind: str) -> list:
    if names is None:
        return items
    known = {item.name for item in items}
    unknown = set(names) - known
    if unknown:
        raise ValueError(
            f"unknown {kind}(s) {sorted(unknown)}; available: {sorted(known)}"
        )
    return [item for item in items if item.name in names]


def _run_job(
    model: ModelSpec,
    variant: VariantSpec,
    index: int,
    repeat: int,
    config: BenchmarkConfig,
    models_dir: Path,
    work_dir: Path,
) -> ModelResult:
    label = f"{model.name}/{variant.name} (pass {repeat + 1})"
    local_dir = models_dir / model.name
    eval_prompts = _tokenize_eval_prompts(local_dir, config.run.eval_files)

    job = Job(
        model=model.name,
        local_dir=str(local_dir.resolve()),
        backend=variant.backend,
        format=variant.format,
        engine=variant.engine,
        prompt_ids=_tokenize_prompt(
            local_dir, config.run.prompt_file, config.run.prompt_tokens
        ),
        eval_prompts=eval_prompts,
        max_new_tokens=config.run.max_new_tokens,
        warmup_runs=config.run.warmup_runs,
        timed_runs=config.run.timed_runs,
    )
    stem = f"{index:03d}-{model.name}-{variant.name}"
    job_path = work_dir / f"{stem}.job.json"
    result_path = work_dir / f"{stem}.result.json"
    log_path = work_dir / f"{stem}.log"
    job_path.write_text(job.model_dump_json(indent=2))
    result_path.unlink(missing_ok=True)

    start_temp_c = read_gpu_temperature_c()
    logger.info("%s: starting (GPU at %.0f C)", label, start_temp_c)
    result = _launch_worker(job, job_path, result_path, log_path, config.run.timeout_s)
    logger.info("%s: %s", label, result.status)
    return result.model_copy(
        update={
            "model_revision": model.revision,
            "order_index": index,
            "start_temp_c": start_temp_c,
            "repeat_index": repeat,
        }
    )


def _tokenize_prompt(model_dir: Path, prompt_file: str, n_tokens: int) -> list[int]:
    """
    Tokenize the fixed prompt once and cut it to exactly n_tokens ids.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tokenizer(Path(prompt_file).read_text()).input_ids
    if len(ids) < n_tokens:
        raise ValueError(
            f"{prompt_file} has {len(ids)} tokens, need at least {n_tokens}"
        )
    return ids[:n_tokens]


def _tokenize_eval_prompts(
    model_dir: Path, eval_files: list[str]
) -> dict[str, list[int]]:
    """Tokenize all evaluation files into full token ID lists."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts: dict[str, list[int]] = {}
    for file_str in eval_files:
        path = Path(file_str)
        if path.exists():
            name = path.stem  # e.g. "code", "reasoning", "factual", "prompt"
            prompts[name] = tokenizer(path.read_text()).input_ids
    return prompts


def _launch_worker(
    job: Job, job_path: Path, result_path: Path, log_path: Path, timeout_s: int
) -> ModelResult:

    def failed(message: str) -> ModelResult:
        return ModelResult(
            model=job.model,
            format=job.format,
            engine=job.engine,
            status="error",
            error_message=message,
        )

    command = [
        sys.executable,
        "-m",
        "quantbench.worker",
        str(job_path),
        str(result_path),
    ]

    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )

        except subprocess.TimeoutExpired:
            return failed(f"timeout after {timeout_s} s")

    if proc.returncode != 0 or not result_path.exists():
        return failed(
            f"worker exited with code {proc.returncode}: {_log_tail(log_path)}"
        )

    try:
        return ModelResult.model_validate_json(result_path.read_text())
    except ValidationError as e:
        return failed(f"invalied result file ({e.error_count()} validation errors)")


def _log_tail(log_path: Path, lines: int = 3, limit: int = 500) -> str:
    text = [line for line in log_path.read_text().splitlines() if line.strip()]
    return " | ".join(text[-lines:])[-limit:]
