"""
Common interface that every (format, engine) backend implements.

Backend only load a model and produce tokens. They never time or measure anything: measurement lives in quantbench.measure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BackendOOMError(Exception):
    """
    Raised by a backend when the device runs out of memory.

    Each backend translates its engine's own out-of-memory error into this one,
    so the measurement layer can record an 'oom' result without knowing any
    engine-spesific exception types
    """


class Backend(ABC):
    """A model served by one specific (format, engine) pair."""

    supports_scoring: bool = False

    @property
    @abstractmethod
    def name(self) -> None:
        """
        Load the model onto its target device.
        """

    @abstractmethod
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Iterator[int]:
        """
        Yield generated token ids one at a time.

        The prompt arrives pre-tokenized so every backend processes exactly the same
        input, which keeps prefill work and TTFT comparable across (format, engine) pairs.

        Streaming lets the measurement layer capture the arrival time of the firt token (TTFT, including prompt processing)
        without the backend doing any timing itself. Each yield must be exactly one token, so the number of yields equals
        the number of generated tokens.
        """

    def score(self, token_ids: list[int]) -> list[float]:
        """
        Return the log-probability of each token given the ones before it.

        Optional: Only available if supports_scoring is True.
        """

        raise NotImplementedError(f"{self.name} does not support scoring.")

    @abstractmethod
    def unload(self) -> None:
        """
        Release the model and free device memory.
        """
