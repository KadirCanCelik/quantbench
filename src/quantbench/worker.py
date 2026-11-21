"""
Runs one job and writes the result as JSON.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from quantbench.backends.registry import build_backend
from quantbench.job import Job
from quantbench.measure.runner import run_benchmark


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python -m quantbench.worker JOB.json RESULT.json", file=sys.stderr
        )
        return 2

    job_path, result_path = Path(argv[1]), Path(argv[2])

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    job = Job.model_validate_json(job_path.read_text())
    backend = build_backend(job.backend, Path(job.local_dir))
    result = run_benchmark(
        backend,
        model=job.model,
        format=job.format,
        engine=job.engine,
        prompt_ids=job.prompt_ids,
        eval_prompts=job.eval_prompts,
        max_new_tokens=job.max_new_tokens,
        warmup_runs=job.warmup_runs,
        timed_runs=job.timed_runs,
        cooldown_s=job.cooldown_s,
    )

    result_path.write_text(result.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
