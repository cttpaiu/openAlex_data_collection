"""Command: export-format — export DB to formatted CSVs (coming soon)."""

import click
from rich.console import Console

console = Console()


@click.command("export-format")
def export_format_command() -> None:
    """Export database to formatted CSVs for analysis (coming soon)."""
    console.print("[yellow]⚠ export-format is not yet implemented.[/yellow]")
    console.print("[dim]This command will export the DuckDB database to analysis-ready CSVs.[/dim]")
