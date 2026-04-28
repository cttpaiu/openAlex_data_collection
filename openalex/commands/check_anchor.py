"""Command: check-anchor — verify anchor papers exist in current result set."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from openalex.anchors import AnchorCheckResult, check_anchor_coverage, parse_anchor_entries, print_anchor_summary
from openalex.config import load_config
from openalex.utils import build_filter
from openalex.validator import check_and_print_keyword_errors, validate_topic_format

console = Console()


@click.command("check-anchor")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--no-topics", is_flag=True, help="Check anchors against keyword-only filter")
def check_anchor_command(config_path: str, no_topics: bool) -> None:
    """Check whether anchor papers are present in the active OpenAlex filter result set."""
    cfg = load_config(config_path)
    cfg.validate_api_key()

    try:
        keywords = cfg.get_keywords()
    except FileNotFoundError as e:
        console.print(f"[bold red]✗ {e}[/bold red]")
        raise SystemExit(1)

    if not check_and_print_keyword_errors(keywords):
        raise SystemExit(1)

    topics: list[str] = []
    context = "Keyword search"
    if not no_topics:
        topics = [t for t in cfg.get_topics() if validate_topic_format(t)]
        context = "Filtered search (keywords + topics)"

    api_filter = build_filter(
        keywords=keywords,
        topics=topics if topics else None,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    enforce_anchor_coverage(cfg, api_filter, context_name=context)


def enforce_anchor_coverage(cfg, api_filter: str, context_name: str) -> AnchorCheckResult:
    raw_entries = cfg.get_anchors()
    if not raw_entries:
        console.print(
            f"[bold red]✗ No anchor entries found.[/bold red]\n"
            f"Edit [cyan]{cfg.anchor_file}[/cyan] and add DOI/title anchors."
        )
        raise SystemExit(1)

    anchors, invalid_entries = parse_anchor_entries(raw_entries)
    if not anchors and invalid_entries:
        result = AnchorCheckResult(found=[], missing=[], invalid=invalid_entries)
        print_anchor_summary(result, context_name=context_name)
        raise SystemExit(1)

    result = asyncio.run(check_anchor_coverage(cfg, api_filter, anchors))
    result = AnchorCheckResult(
        found=result.found,
        missing=result.missing,
        invalid=invalid_entries,
    )
    print_anchor_summary(result, context_name=context_name)

    if result.invalid or result.missing:
        raise SystemExit(1)

    return result
