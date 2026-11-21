from __future__ import annotations

from pydantic import Field

from quantbench.schema import StrictModel


class Job(StrictModel):
    model: str
    local_dir: str
    backend: str
    format: str
    engine: str
    prompt_ids: list[int]
    eval_prompts: dict[str, list[int]] = Field(default_factory=dict)
    max_new_tokens: int
    warmup_runs: int = 4
    timed_runs: int = 5
    cooldown_s: float = 0.0
