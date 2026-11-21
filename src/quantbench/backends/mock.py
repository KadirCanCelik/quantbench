from __future__ import annotations

import time
from collections.abc import Iterator

from quantbench.backends.base import Backend


class MockBackend(Backend):
    """
    Simulates prefill and devoce delays without touching a GPU.
    """

    def __init__(
        self,
        prefill_s: float = 0.100,
        per_token_s: float = 0.010,
        eager_prefill: bool = False,
    ) -> None:

        self._prefill_s = prefill_s
        self._per_token_s = per_token_s
        self._eager_prefill = eager_prefill

    @property
    def name(self) -> str:
        return "mock"

    def load(self) -> None:
        pass

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Iterator[int]:
        if self._eager_prefill:
            time.sleep(self._prefill_s)

        return self._stream(max_new_tokens, do_prefill=not self._eager_prefill)

    def _stream(self, max_new_tokens: int, do_prefill: bool) -> Iterator[int]:

        if do_prefill:
            time.sleep(self._prefill_s)

        for i in range(max_new_tokens):
            if i > 0:
                time.sleep(self._per_token_s)

            yield i

    def unload(self) -> None:
        pass
