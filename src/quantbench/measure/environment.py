"""
Collects the hardware and software description stored in every report.
"""

from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path

import pynvml

from quantbench.schema import Environment

_TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "nvidia-ml-py",
    "pydantic",
    "huggingface_hub",
]
_MIB = 1024**2


def _text(
    value: str | bytes,
) -> str:  # Older NVML bindings return bytes, newer ones return str.
    return value.decode() if isinstance(value, bytes) else value


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()

    except OSError:
        pass
    return platform.processor() or "unknown"


def collect_environment(device_index: int = 0) -> Environment:
    """
    Describe the GPU, driver and key library versions.
    """
    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = _text(pynvml.nvmlDeviceGetName(handle))
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
        vram_total_mb = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total / _MIB)

        try:
            power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        except pynvml.NVMLError:
            power_limit_w = None  # not every GPU or driver reports it

    finally:
        pynvml.nvmlShutdown()

    versions: dict[str, str] = {}

    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)

        except metadata.PackageNotFoundError:
            pass

    return Environment(
        gpu_name=name,
        driver_version=driver,
        vram_total_mb=vram_total_mb,
        power_limit_w=power_limit_w,
        python_version=platform.python_version(),
        library_versions=versions,
        cpu_model=_cpu_model(),
        cpu_threads=os.cpu_count() or 0,
    )
