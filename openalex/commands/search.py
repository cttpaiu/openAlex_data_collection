"""Commands: search (keyword-only) and search-filtered (keyword + topics)."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from openalex.api_client import AsyncOpenAlexClient
from openalex.commands.check_anchor import enforce_anchor_coverage
from openalex.config import load_config
from openalex.utils import build_filter, print_search_result_panel
from openalex.validator import check_and_print_keyword_errors, validate_topic_format

console = Console()


@click.command("search")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--dry-run", is_flag=True, help="Validate inputs only, skip API call")
def search_command(config_path: str, dry_run: bool) -> None:
    """Count papers matching keyword query (no topic filter)."""
    cfg = load_config(config_path)
    cfg.validate_api_key()

    try:
        keywords = cfg.get_keywords()
    except FileNotFoundError as e:
        console.print(f"[bold red]✗ {e}[/bold red]")
        raise SystemExit(1)

    if not check_and_print_keyword_errors(keywords):
        raise SystemExit(1)

    if dry_run:
        console.print("[green]✓ Dry run: keywords are valid.[/green]")
        return

    api_filter = build_filter(
        keywords=keywords,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    count = asyncio.run(_get_count(cfg, api_filter))
    print_search_result_panel(
        title="OpenAlex Keyword Search",
        count=count,
        keywords_file=cfg.keywords_file,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )
    enforce_anchor_coverage(cfg, api_filter, context_name="Keyword search")


@click.command("search-filtered")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--dry-run", is_flag=True, help="Validate inputs only, skip API call")
def search_filtered_command(config_path: str, dry_run: bool) -> None:
    """Count papers matching both keywords AND topics filter."""
    cfg = load_config(config_path)
    cfg.validate_api_key()

    try:
        keywords = cfg.get_keywords()
    except FileNotFoundError as e:
        console.print(f"[bold red]✗ {e}[/bold red]")
        raise SystemExit(1)

    if not check_and_print_keyword_errors(keywords):
        raise SystemExit(1)

    topics = cfg.get_topics()
    if not topics:
        console.print(
            "[bold red]✗ No topic IDs found in topics file.[/bold red]\n"
            f"Edit [cyan]{cfg.topics_file}[/cyan] or run [cyan]openalex get-topics[/cyan] first."
        )
        raise SystemExit(1)

    invalid = [t for t in topics if not validate_topic_format(t)]
    if invalid:
        console.print(f"[yellow]⚠ Malformed topic IDs (skipped): {', '.join(invalid)}[/yellow]")
        topics = [t for t in topics if validate_topic_format(t)]

    if dry_run:
        console.print(f"[green]✓ Dry run: keywords valid, {len(topics)} topic IDs loaded.[/green]")
        return

    api_filter = build_filter(
        keywords=keywords,
        topics=topics,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    count = asyncio.run(_get_count(cfg, api_filter))
    print_search_result_panel(
        title="OpenAlex Filtered Search",
        count=count,
        keywords_file=cfg.keywords_file,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
        topics_count=len(topics),
    )
    enforce_anchor_coverage(cfg, api_filter, context_name="Filtered search (keywords + topics)")


async def _get_count(cfg, api_filter: str) -> int:
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        per_page=1,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        return await client.get_total_count(api_filter)



    