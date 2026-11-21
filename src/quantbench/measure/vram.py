"""
Background GPU sampling through NVML
"""

from __future__ import annotations

import statistics
import threading
import time
from types import TracebackType
from typing import Self

import pynvml

SAMPLE_INTERVAL_S = 0.020
_MIB = 1024**2


class GpuSampler:
    """
    Context manager that samples VRAM, SM clock and temperature.
    """

    def __init__(
        self, device_index: int = 0, interval_s: float = SAMPLE_INTERVAL_S
    ) -> None:
        self._device_index = device_index
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._vram_mb: list[float] = []
        self._clock_mhz: list[float] = []
        self._temp_c: list[float] = []
        self._util_pct: list[float] = []

    def __enter__(self) -> Self:

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        self._sample_once()  # guarantees at least one sample for very short blocks
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join()

        self._sample_once()  # final sample after the work finished
        pynvml.nvmlShutdown()

    def _sample_once(self) -> None:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self._vram_mb.append(mem.used / _MIB)

        self._clock_mhz.append(
            float(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM))
        )
        self._temp_c.append(
            float(
                pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        )

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            self._util_pct.append(float(util))

        except pynvml.NVMLError:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample_once()

    @property
    def n_samples(self) -> int:
        return len(self._vram_mb)

    @property
    def peak_vram_mb(self) -> float:
        return max(self._vram_mb)

    @property
    def median_clock_mhz(self) -> float:
        return statistics.median(self._clock_mhz)

    @property
    def max_temp_c(self) -> float:
        return max(self._temp_c)

    @property
    def median_vram_mb(self) -> float:
        return statistics.median(self._vram_mb)

    @property
    def vram_spread_mb(self) -> float:
        """
        Max - min sample, large values mean other gpu users are fluctuating.
        """
        return max(self._vram_mb) - min(self._vram_mb)

    @property
    def mean_util_pct(self) -> float | None:
        return statistics.mean(self._util_pct) if self._util_pct else None


BASELINE_WINDOW_S = 0.5


def measure_baseline_vram_mb(
    window_s: float = BASELINE_WINDOW_S,
) -> tuple[float, float]:
    """
    Return of the GPU while idle. Call this befor loading model or creating a CUDA context
    in this process, so the baseline reflects only what other programs are using.
    """

    with GpuSampler() as sampler:
        time.sleep(window_s)

    return sampler.median_vram_mb, sampler.vram_spread_mb


def read_gpu_temperature_c(device_index: int = 0) -> float:
    """
    One-shot GPU temperature through NVML (does not create a CUDA context).
    """

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )

    finally:
        pynvml.nvmlShutdown()
