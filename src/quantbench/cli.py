"""
Command Line Interface
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from quantbench.config import load_config
from quantbench.fetch import MODELS_DIR, fetch_model
from quantbench.orchestrator import run_matrix, select_by_name

app = typer.Typer(
    help="Benchmark LLm quantization formats on a GPU.", no_args_is_help=True
)

DEFAULT_CONFIG = Path("configs/benchmark.yaml")
ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Benchmark config file.")
]
DebugOpt = Annotated[bool, typer.Option("--debug", help="Verbose logging.")]
ModelOpt = Annotated[
    list[str] | None,
    typer.Option("--model", "-m", help="Only these models (repeatable)."),
]


def _setup_logging(debug: bool) -> None:
    handler = RichHandler(show_path=False, rich_tracebacks=True)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[handler])
    logging.getLogger("quantbench").setLevel(logging.DEBUG if debug else logging.INFO)


@app.command()
def fetch(
    config: ConfigOpt = DEFAULT_CONFIG, model: ModelOpt = None, debug: DebugOpt = False
) -> None:
    """
    Download the pinned models into models/
    """
    _setup_logging(debug)
    cfg = load_config(config)
    for spec in select_by_name(cfg.models, model, "model"):
        typer.echo(f"fetching {spec.name} ({spec.repo} @ {spec.revision[:8]})")
        typer.echo(f"  -> {fetch_model(spec)}")


@app.command()
def run(
    config: ConfigOpt = DEFAULT_CONFIG,
    model: ModelOpt = None,
    variant: Annotated[
        list[str] | None, typer.Option("--variant", "-v", help="Only these variants.")
    ] = None,
    reverse: Annotated[
        bool, typer.Option("--reverse", help="Run the matrix in reverse order.")
    ] = False,
    debug: DebugOpt = False,
    repeats: Annotated[
        int | None,
        typer.Option("--repeats", "-r", min=1, help="Passes over the matrix."),
    ] = None,
) -> None:
    """
    Run the benchmark matrix and write a JSON report to results/
    """
    _setup_logging(debug)
    cfg = load_config(config)

    missing = [
        m.name
        for m in select_by_name(cfg.models, model, "model")
        if not (MODELS_DIR / m.name / "config.json").exists()
    ]

    if missing:
        typer.echo(
            f"models not downloaded: {', '.join(missing)}. Run `quantbench fetch` first."
        )
        raise typer.Exit(1)

    report = run_matrix(
        cfg, only_models=model, only_variants=variant, reverse=reverse, repeats=repeats
    )
    typer.echo(f"report:{report}")


@app.command()
def show(
    report: Annotated[Path, typer.Argument(help="JSON report file to display.")],
    debug: DebugOpt = False,
) -> None:
    """
    Display a benchmark report as a rich terminal table.
    """
    _setup_logging(debug)

    if not report.exists():
        typer.echo(f"report not found: {report}", err=True)
        raise typer.Exit(1)

    from pydantic import ValidationError

    from quantbench.report import load_report, render_report

    try:
        r = load_report(report)
    except ValidationError as e:
        typer.echo(
            f"invalid report ({e.error_count()} validation errors): {report}",
            err=True,
        )
        raise typer.Exit(1) from e

    render_report(r)
