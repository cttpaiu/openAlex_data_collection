"""Command: get-topics — list research topics from keyword search results."""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from openalex.api_client import AsyncOpenAlexClient
from openalex.config import load_config
from openalex.utils import build_filter
from openalex.validator import check_and_print_keyword_errors

console = Console()


@click.command("get-topics")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--details", is_flag=True, help="Show paper count per topic")
@click.option("--csv", "save_csv", is_flag=True, help="Save results to CSV")
@click.option("--filter-topics", is_flag=True, help="Only show topics that are in topics.txt")
@click.option("--output", "-o", default=None, help="Output CSV filename (skips prompt if set)")
def get_topics_command(config_path: str, details: bool, save_csv: bool, filter_topics: bool, output: str) -> None:
    """List research topics appearing in keyword search results."""
    cfg = load_config(config_path)
    cfg.validate_api_key()

    try:
        keywords = cfg.get_keywords()
    except FileNotFoundError as e:
        console.print(f"[bold red]✗ {e}[/bold red]")
        raise SystemExit(1)

    if not check_and_print_keyword_errors(keywords):
        raise SystemExit(1)

    api_filter = build_filter(
        keywords=keywords,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    console.print("[dim]Fetching topic groups from OpenAlex...[/dim]")
    groups = asyncio.run(_fetch_all_groups(cfg, api_filter))

    if not groups:
        console.print("[yellow]⚠ No topics returned.[/yellow]")
        return

    # Filter to topics.txt if requested
    if filter_topics:
        allowed = set(cfg.get_topics())
        if allowed:
            groups = [g for g in groups if g["key"] in allowed]

    total_papers = sum(g["count"] for g in groups)

    # Enrich with display names if needed (and not already in group_by response)
    if details or save_csv:
        groups = asyncio.run(_enrich_topic_names(cfg, groups))

    # Auto-switch to CSV for large lists
    if len(groups) > 50 and not save_csv and not details:
        console.print(
            f"[yellow]⚠ {len(groups)} topics found — showing top 20. Use --details --csv to save all.[/yellow]"
        )
        _print_topic_table(groups[:20], total_papers, title=f"Top 20 of {len(groups)} Topics")
        return

    if save_csv:
        _save_to_csv(groups, total_papers, output)
        return

    if details:
        if len(groups) > 50:
            _save_to_csv(groups, total_papers)
        else:
            _print_topic_table(groups, total_papers, title=f"Topics ({len(groups)} found)")
    else:
        _print_topic_table(groups[:20], total_papers, title=f"Topics ({len(groups)} found)")
        if len(groups) > 20:
            console.print(
                f"[dim](+ {len(groups) - 20} more — use --details --csv to see all)[/dim]"
            )


def _print_topic_table(groups: list[dict], total_papers: int, title: str) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Topic ID", style="cyan", no_wrap=True)
    table.add_column("Topic Name", style="white")
    table.add_column("Papers", justify="right", style="green")
    table.add_column("%", justify="right", style="dim")

    for g in groups:
        pct = f"{g['count'] / total_papers * 100:.1f}%" if total_papers else "-"
        table.add_row(g["key"], g.get("display_name", g.get("key_display_name", "—")), f"{g['count']:,}", pct)

    console.print(table)


def _save_to_csv(groups: list[dict], total_papers: int, output: str = None) -> None:
    if output:
        filename = output
    else:
        import questionary
        from datetime import datetime

        default_name = f"topics_{datetime.now().strftime('%Y%m%d')}.csv"
        filename = questionary.text("Output filename:", default=default_name).ask()
        if not filename:
            filename = default_name

    path = Path(filename)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic_id", "display_name", "paper_count", "percentage"])
        writer.writeheader()
        for g in groups:
            pct = round(g["count"] / total_papers * 100, 2) if total_papers else 0
            writer.writerow({
                "topic_id": g["key"],
                "display_name": g.get("display_name", g.get("key_display_name", "")),
                "paper_count": g["count"],
                "percentage": pct,
            })

    console.print(f"[green]✓ Saved {len(groups)} topics to [cyan]{filename}[/cyan][/green]")


async def _fetch_all_groups(cfg: Any, api_filter: str) -> list[dict]:
    """Fetch all topic groups using cursor pagination (OpenAlex returns max 200 per page)."""
    all_groups: list[dict] = []
    cursor = "*"

    async with AsyncOpenAlexClient(
        api_key=cfg.api_key,
        email=cfg.email,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        while cursor:
            data = await client.fetch_group_by(api_filter, "primary_topic.id", cursor=cursor)
            if not data:
                break
            batch = data.get("group_by", [])
            if not batch:
                break
            all_groups.extend(batch)
            # Get next cursor from meta
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            # If no next_cursor, we've fetched all groups
            if not next_cursor:
                break
            cursor = next_cursor

    return sorted(all_groups, key=lambda g: g["count"], reverse=True)


async def _enrich_topic_names(cfg: Any, groups: list[dict]) -> list[dict]:
    """Fetch display_name for each topic in parallel (if not already present)."""
    needs_enrichment = [g for g in groups if not g.get("display_name") and g.get("key_display_name") is None]

    if not needs_enrichment:
        # Use key_display_name as display_name if already present
        for g in groups:
            if not g.get("display_name"):
                g["display_name"] = g.get("key_display_name", g["key"])
        return groups

    async with AsyncOpenAlexClient(
        api_key=cfg.api_key,
        email=cfg.email,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        tasks = [client.fetch_topic_details(g["key"]) for g in groups]
        details_list = await asyncio.gather(*tasks)

    for g, details in zip(groups, details_list):
        if details:
            g["display_name"] = details.get("display_name", g["key"])
        else:
            g["display_name"] = g.get("key_display_name", g["key"])

    return groups
