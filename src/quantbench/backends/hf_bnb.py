"""
bitsandbytes backends (INT8 and NF4), reusing the FP16 backend's greedy loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from quantbench.backends.base import BackendOOMError
from quantbench.backends.hf_fp16 import HfFp16Backend

Mode = Literal["int8", "nf4"]


def _quantization_config(mode: Mode) -> BitsAndBytesConfig:
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


class HfBnbBackend(HfFp16Backend):
    def __init__(self, model_dir: Path, mode: Mode, device_index: int = 0) -> None:
        super().__init__(model_dir, device=f"cuda:{device_index}")
        self._mode = mode
        self._device_index = device_index

    @property
    def name(self) -> str:
        return f"{self._mode} (transformers + bitsandbytes, greedy loop)"

    def load(self) -> None:
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_dir,
                dtype=torch.float16,  # non-quantized modules, same as the FP16 variant
                quantization_config=_quantization_config(self._mode),
                device_map={"": self._device_index},
                local_files_only=True,
            )
            self._model = model.eval()
        except torch.cuda.OutOfMemoryError as e:
            model = None
            raise BackendOOMError(str(e)) from e
