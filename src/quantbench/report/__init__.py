"""
Report layer: reads JSON reports and renders them as a rich terminal table.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from quantbench.schema import ModelResult, Report

_SPILL_THRESHOLD = 0.97  # peak VRAM / total VRAM


def load_report(path: Path) -> Report:
    """Read and validate a JSON report file."""
    return Report.model_validate_json(path.read_text())


def render_report(report: Report, console: Console | None = None) -> None:
    """Print a rich table summary of a report to the terminal."""
    if console is None:
        console = Console()
    env = report.environment

    console.print(
        f"\n[bold]GPU:[/bold] {env.gpu_name}  |  "
        f"[bold]VRAM:[/bold] {env.vram_total_mb} MiB  |  "
        f"[bold]Driver:[/bold] {env.driver_version}  |  "
        f"[bold]Python:[/bold] {env.python_version}"
    )
    lib_line = "  •  ".join(f"{k}: {v}" for k, v in env.library_versions.items())
    if lib_line:
        console.print(f"  {lib_line}\n", style="dim")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        pad_edge=True,
        padding=(0, 2),  # left-right 2 spaces padding between columns
    )

    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("Format", style="magenta", no_wrap=True)
    table.add_column("Engine", style="dim white", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("TTFT (med · p95)", justify="right", min_width=18)
    table.add_column("tok/s (med · p5)", justify="right", min_width=18)
    table.add_column("Peak VRAM (MiB)", justify="right", min_width=16)
    table.add_column("PPL", justify="right", min_width=8)
    table.add_column("Warnings", justify="center", no_wrap=True)
    footer_warnings: list[str] = []
    for r in report.results:
        _add_row(table, r, env.vram_total_mb, footer_warnings)
    console.print(table)

    if footer_warnings:
        console.print("\n[yellow bold]Warnings[/yellow bold]")
        for w in footer_warnings:
            console.print(Text(f"  ⚠  {w}", style="yellow"))
    console.print()


def _add_row(
    table: Table, r: ModelResult, total_vram_mb: int, warnings: list[str]
) -> None:
    status_text, status_style = _status_cell(r)

    if r.status != "ok" or r.summary is None:
        table.add_row(
            r.model,
            r.format,
            r.engine,
            Text(status_text, style=status_style),
            "—",
            "—",
            "—",
            "—",
            "",
        )
        return

    s = r.summary

    ttft_cell = f"{s.ttft_ms.median:.1f} · {s.ttft_ms.p95:.1f}"

    tps_cell = f"{s.tokens_per_s.median:.1f} · {s.tokens_per_s.p5:.1f}"

    vram_cell = f"{s.peak_vram_mb.median:.0f}"

    ppl_cell = f"{r.perplexity:.3f}" if r.perplexity is not None else "—"

    row_flags: list[str] = []

    if s.peak_vram_mb.median >= _SPILL_THRESHOLD * total_vram_mb:
        row_flags.append("SPILL")
        warnings.append(
            f"[SPILL] {r.model}/{r.format}: "
            f"peak {s.peak_vram_mb.median:.0f} MiB ≥ {_SPILL_THRESHOLD:.0%} of "
            f"total {total_vram_mb} MiB — model may be spilling to system RAM"
        )
    if r.runs:
        clocks = [run.gpu_clock_mhz for run in r.runs if run.gpu_clock_mhz is not None]
        if clocks and statistics.median(clocks) < 300:
            row_flags.append("SLOW")
            warnings.append(
                f"[SLOW] {r.model}/{r.format}: median GPU clock {statistics.median(clocks):.0f} MHz "
                f"— GPU may be throttling (check power mode)"
            )
    warn_cell = " ".join(row_flags)
    warn_style = "yellow" if row_flags else ""
    table.add_row(
        r.model,
        r.format,
        r.engine,
        Text(status_text, style=status_style),
        ttft_cell,
        tps_cell,
        vram_cell,
        ppl_cell,
        Text(warn_cell, style=warn_style),
    )


def _status_cell(r: ModelResult) -> tuple[str, str]:
    if r.status == "ok":
        return "ok", "green"
    if r.status == "oom":
        return "OOM", "red"
    return "error", "red bold"
