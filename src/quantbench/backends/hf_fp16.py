"""FP16 backend: Hugging Face transformers with a hand-written greedy loop.

The loop is written by hand (instead of model.generate()) so that tokens are
yielded one at a time without threads or queues, and so that the number of
generated tokens is exactly max_new_tokens: EOS is simply never checked.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from quantbench.backends.base import Backend, BackendOOMError


class HfFp16Backend(Backend):
    def __init__(self, model_dir: Path, device: str = "cuda") -> None:
        self._model_dir = model_dir
        self._device = device
        self._model: AutoModelForCausalLM | None = None

    @property
    def name(self) -> str:
        return "fp16 (transformers, greedy loop)"

    def load(self) -> None:
        model = None

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_dir, dtype=torch.float16, local_files_only=True
            )
            self._model = model.to(self._device).eval()

        except torch.cuda.OutOfMemoryError as e:
            model = None
            raise BackendOOMError(str(e)) from e

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Iterator[int]:
        if self._model is None:
            raise RuntimeError("load() must be called before generate()")

        input_ids = torch.tensor([prompt_ids], device=self._device)
        attention_mask = torch.ones_like(input_ids)
        cache = DynamicCache(config=self._model.config)
        try:
            for _ in range(max_new_tokens):
                next_ids = self._step(input_ids, attention_mask, cache)
                yield int(next_ids.item())
                input_ids = next_ids
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, 1))], dim=-1
                )
        except torch.cuda.OutOfMemoryError as e:
            raise BackendOOMError(str(e)) from e

    @torch.inference_mode()
    def _step(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, cache: DynamicCache
    ) -> torch.Tensor:
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        return out.logits[:, -1:].argmax(-1)

    def unload(self) -> None:
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    supports_scoring: bool = True

    @torch.inference_mode()
    def score(self, token_ids: list[int]) -> list[float]:
        """
        Compute the log-probability of each token given preceding context.

        In an autoregressive causal language model, predicting the initial token (t=0)
        is undefined because it lacks preceding context. Therefore, for a sequence of
        length N, predictions are made for tokens 1 through N-1, yielding exactly N - 1
        log-probabilities.

        """

        if self._model is None:
            raise RuntimeError("load() must be called before score().")

        if len(token_ids) < 2:
            raise ValueError("score() requires at least 2 tokens.")

        try:
            inputs = torch.tensor([token_ids], device=self._device)
            outputs = self._model(inputs, use_cache=False)
            logits = outputs.logits

            shift_logits = logits[0, :-1, :].contiguous()
            shift_targets = inputs[0, 1:].contiguous()

            nll = torch.nn.functional.cross_entropy(
                shift_logits, shift_targets, reduction="none"
            )
            log_probs = -nll
            return log_probs.tolist()

        except torch.cuda.OutOfMemoryError as e:
            raise BackendOOMError(str(e)) from e
