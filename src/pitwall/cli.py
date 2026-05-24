"""Pit Wall CLI — `pitwall ingest`, `pitwall build`, `pitwall train`."""

from __future__ import annotations

import typer
from rich.console import Console

from pitwall.ingest.fastf1_client import ingest_round, ingest_season, setup_cache

app = typer.Typer(help="Pit Wall Intelligence — F1 race-strategy analytics CLI")
console = Console()


@app.command()
def ingest(
    year: int = typer.Option(..., help="Season year"),
    round_: int = typer.Option(None, "--round", "-r", help="Specific round (omit for full season)"),
) -> None:
    """Pull FastF1 data for a season or single round."""
    setup_cache()
    if round_ is None:
        console.print(f"[bold cyan]Ingesting full {year} season[/]")
        ingest_season(year)
    else:
        console.print(f"[bold cyan]Ingesting {year} round {round_}[/]")
        ingest_round(year, round_)


@app.command()
def build() -> None:
    """Run the dbt build pipeline."""
    import subprocess

    subprocess.run(["dbt", "build", "--project-dir", "dbt"], check=False)


@app.command()
def train() -> None:
    """Fit degradation curves and the undercut classifier on processed data."""
    console.print("[bold cyan]Training degradation + undercut models...[/]")
    console.print("Not yet implemented — see notebooks/03_model_validation.ipynb")


if __name__ == "__main__":
    app()
