"""Command: validate — check keywords.txt and topics.txt before hitting the API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openalex.config import load_config
from openalex.validator import (
    check_and_print_keyword_errors,
    validate_topic_format,
    validate_topics_exist,
)

console = Console()


@click.command("validate")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--no-api", is_flag=True, help="Skip API existence check for topic IDs (format check only)")
def validate_command(config_path: str, no_api: bool) -> None:
    """Validate keywords.txt and topics.txt before running any collection command."""
    cfg = load_config(config_path)
    all_ok = True

    # ── 1. Keywords ──────────────────────────────────────────────────────────
    console.rule("[bold]Keywords[/bold]")
    kw_path = Path(cfg.keywords_file)

    if not kw_path.exists():
        console.print(f"[bold red]✗ File not found:[/bold red] {cfg.keywords_file}")
        all_ok = False
    else:
        keywords = kw_path.read_text(encoding="utf-8").strip()
        console.print(f"[dim]File:[/dim] [cyan]{cfg.keywords_file}[/cyan]  "
                      f"[dim]({len(keywords)} chars, ~{len(keywords.split())} words)[/dim]")

        from openalex.validator import validate_keywords
        errors = validate_keywords(keywords)

        if errors:
            console.print("[bold red]✗ Keyword validation failed:[/bold red]")
            for e in errors:
                console.print(f"  [red]• {e}[/red]")
            all_ok = False
        else:
            console.print("[bold green]✓ Keywords are valid[/bold green]")
            # Show a preview of the query (first 200 chars)
            preview = keywords[:200].replace("\n", " ").strip()
            if len(keywords) > 200:
                preview += "…"
            console.print(f"  [dim]Preview:[/dim] {preview}")

    # ── 2. Topics ─────────────────────────────────────────────────────────────
    console.rule("[bold]Topics[/bold]")
    topics_path = Path(cfg.topics_file)

    if not topics_path.exists():
        console.print(f"[yellow]⚠ Topics file not found:[/yellow] {cfg.topics_file}  [dim](optional)[/dim]")
    else:
        raw_lines = topics_path.read_text(encoding="utf-8").splitlines()
        topic_ids = [
            line.strip()
            for line in raw_lines
            if line.strip() and not line.strip().startswith("#")
        ]

        console.print(f"[dim]File:[/dim] [cyan]{cfg.topics_file}[/cyan]  "
                      f"[dim]({len(topic_ids)} topic IDs)[/dim]")

        if not topic_ids:
            console.print("[yellow]⚠ No topic IDs found (file is empty or only comments)[/yellow]")
        else:
            # Stage 1: format check
            valid_ids = [t for t in topic_ids if validate_topic_format(t)]
            bad_format = [t for t in topic_ids if not validate_topic_format(t)]

            _print_format_results(valid_ids, bad_format)

            if bad_format:
                all_ok = False

            # Stage 2: API existence check (optional)
            if valid_ids and not no_api:
                cfg.validate_api_key()
                console.print(f"\n[dim]Checking {len(valid_ids)} IDs against OpenAlex API...[/dim]")
                existence = asyncio.run(
                    validate_topics_exist(valid_ids, cfg.email, cfg.api_key)
                )
                _print_api_results(existence)
                not_found = [tid for tid, ok in existence.items() if not ok]
                if not_found:
                    all_ok = False
            elif no_api:
                console.print("[dim]  (API check skipped via --no-api)[/dim]")

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule()
    if all_ok:
        console.print("[bold green]✓ All checks passed — ready to run collection commands.[/bold green]")
    else:
        console.print("[bold red]✗ Validation failed — fix the errors above before collecting.[/bold red]")
        raise SystemExit(1)


def _print_format_results(valid: list[str], bad: list[str]) -> None:
    if bad:
        console.print(f"[bold red]✗ {len(bad)} malformed topic ID(s):[/bold red]")
        for t in bad:
            console.print(f"  [red]• {t!r}[/red]  [dim](expected format: T + 5 digits, e.g. T10020)[/dim]")
    else:
        console.print(f"[green]✓ All {len(valid)} topic IDs have correct format (T + 5 digits)[/green]")


def _print_api_results(existence: dict[str, bool]) -> None:
    found = [tid for tid, ok in existence.items() if ok]
    missing = [tid for tid, ok in existence.items() if not ok]

    console.print(f"[green]✓ {len(found)} topic IDs confirmed on OpenAlex[/green]")

    if missing:
        console.print(f"[bold red]✗ {len(missing)} topic ID(s) not found on OpenAlex:[/bold red]")
        for tid in missing:
            console.print(f"  [red]• {tid}[/red]")
