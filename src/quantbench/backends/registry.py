"""
Maps backend names used in configs to backend instances.

A benchmark config refers to a backend by name ( for example 'hf_int8').
The worker process calls 'build_backend' with that name to get a ready-to-use
'Backend'. This module is the only place that knows which name belongs to which class.

Imports happen inside the factories on purpose. Importing a backend pulls in its heavy
dependencies (torch, transformers ...) doing it lazily means a process only loads
the libraries of the backend it actually runs, and a missing optional library only breaks
the backends that need it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from quantbench.backends.base import Backend

BackendFactory = Callable[[Path], Backend]


def _hf_fp16(model_dir: Path) -> Backend:
    from quantbench.backends.hf_fp16 import HfFp16Backend

    return HfFp16Backend(model_dir)


def _bnb(mode: Literal["int8", "nf4"]) -> BackendFactory:
    """
    Return a factory for the bitsandbytes backend in the given mode.
    """

    def factory(model_dir: Path) -> Backend:
        from quantbench.backends.hf_bnb import HfBnbBackend

        return HfBnbBackend(model_dir, mode)

    return factory


_BACKENDS: dict[str, BackendFactory] = {
    "hf_fp16": _hf_fp16,
    "hf_int8": _bnb("int8"),
    "hf_nf4": _bnb("nf4"),
}


def build_backend(name: str, model_dir: Path) -> Backend:
    """
    Build the backend registered under 'name' for the model in 'model_dir'.
    """
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)}")

    return factory(model_dir)
