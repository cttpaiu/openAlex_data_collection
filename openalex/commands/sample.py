"""Command: sample — random reservoir sample of matching papers."""

from __future__ import annotations

import asyncio
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openalex.api_client import AsyncOpenAlexClient
from openalex.config import load_config
from openalex.utils import build_filter, reconstruct_abstract
from openalex.validator import check_and_print_keyword_errors, validate_topic_format

console = Console()

SAMPLE_FIELDS = (
    "id,doi,title,publication_year,type,primary_topic,"
    "abstract_inverted_index,cited_by_count,fwci,"
    "institutions_distinct_count,countries_distinct_count"
)


@click.command("sample")
@click.option("--size", "-n", required=True, type=int, help="Number of papers to sample")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--no-topics", is_flag=True, help="Use keyword filter only (ignore topics file)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--csv", "save_csv", is_flag=True, help="Save sample to CSV automatically")
@click.option("--output", "-o", default=None, help="Output CSV filename (used with --csv)")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--fast", is_flag=True, help="Use OpenAlex sample API (fast but BIASED - not recommended)")
def sample_command(size: int, config_path: str, no_topics: bool, yes: bool, save_csv: bool, output: str, seed: int, fast: bool) -> None:
    """Take a random validation sample from matching papers.

    Uses reservoir sampling for statistically valid random sampling.
    Every paper has equal probability of selection.

    Note: For large datasets (500k+ papers), this may take 5-10 minutes.
    Use --fast for quick exploration (but sample may be biased).
    """
    cfg = load_config(config_path)
    cfg.validate_api_key()

    try:
        keywords = cfg.get_keywords()
    except FileNotFoundError as e:
        console.print(f"[bold red]✗ {e}[/bold red]")
        raise SystemExit(1)

    if not check_and_print_keyword_errors(keywords):
        raise SystemExit(1)

    topics: Optional[list[str]] = None
    if not no_topics:
        raw_topics = cfg.get_topics()
        if raw_topics:
            topics = [t for t in raw_topics if validate_topic_format(t)]

    # Show active filters before fetching
    _print_filter_summary(cfg, topics, size)

    if not yes:
        import questionary
        if not questionary.confirm("Proceed?", default=True).ask():
            console.print("[dim]Cancelled.[/dim]")
            return

    api_filter = build_filter(
        keywords=keywords,
        topics=topics,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    if fast:
        console.print(f"\n[yellow]⚠ Using FAST sampling (OpenAlex sample API)[/yellow]")
        console.print(f"[dim]Warning: This may produce BIASED samples that don't represent the population.[/dim]")
        console.print(f"[dim]For statistically valid sampling, omit --fast to use reservoir sampling.[/dim]\n")
        sampled = asyncio.run(_fetch_sample_fast(cfg, api_filter, size, seed))
    else:
        console.print(f"\n[dim]Using reservoir sampling for statistically valid random sample...[/dim]")
        sampled = asyncio.run(_reservoir_sample(cfg, api_filter, size, seed))

    if not sampled:
        console.print("[yellow]⚠ No papers returned.[/yellow]")
        return

    _print_sample_table(sampled)

    if save_csv:
        filename = output or f"sample_{datetime.now().strftime('%Y%m%d')}.csv"
        _save_sample_csv(sampled, filename)
    elif not yes and questionary.confirm("\nSave sample to CSV?", default=True).ask():
        default_name = f"sample_{datetime.now().strftime('%Y%m%d')}.csv"
        filename = questionary.text("Filename:", default=default_name).ask() or default_name
        _save_sample_csv(sampled, filename)


def _print_filter_summary(cfg: Any, topics: Optional[list[str]], size: int) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="cyan")
    table.add_row("Keywords", cfg.keywords_file)
    table.add_row("Topics", f"{len(topics)} IDs" if topics else "None (keyword-only)")
    table.add_row("Date range", f"{cfg.date_from} → {cfg.date_to}")
    table.add_row("Doc types", ", ".join(cfg.doc_types))
    table.add_row("Sample size", str(size))
    console.print(Panel(table, title="[bold]Active filters[/bold]", expand=False))


def _print_sample_table(papers: List[Dict]) -> None:
    table = Table(title=f"Random Sample ({len(papers)} papers)", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Year", width=6)
    table.add_column("Title", style="white", max_width=55)
    table.add_column("Topic", style="cyan", max_width=30)
    table.add_column("Cited", justify="right", style="green")

    for i, p in enumerate(papers, 1):
        topic = (p.get("primary_topic") or {}).get("display_name", "—")
        title = (p.get("title") or "Untitled")[:54]
        table.add_row(
            str(i),
            str(p.get("publication_year") or "—"),
            title,
            topic[:29],
            str(p.get("cited_by_count") or 0),
        )

    console.print(table)


def _save_sample_csv(papers: List[Dict], filename: str) -> None:
    path = Path(filename)
    fieldnames = [
        "id", "doi", "title", "publication_year", "type",
        "topic_id", "topic_name", "abstract_text",
        "cited_by_count", "fwci", "institutions_count", "countries_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in papers:
            topic = p.get("primary_topic") or {}
            writer.writerow({
                "id": p.get("id", ""),
                "doi": p.get("doi", ""),
                "title": p.get("title", ""),
                "publication_year": p.get("publication_year", ""),
                "type": p.get("type", ""),
                "topic_id": (topic.get("id") or "").split("/")[-1],
                "topic_name": topic.get("display_name", ""),
                "abstract_text": reconstruct_abstract(p.get("abstract_inverted_index")),
                "cited_by_count": p.get("cited_by_count", ""),
                "fwci": p.get("fwci", ""),
                "institutions_count": p.get("institutions_distinct_count", ""),
                "countries_count": p.get("countries_distinct_count", ""),
            })
    console.print(f"[green]✓ Saved to [cyan]{filename}[/cyan][/green]")


async def _reservoir_sample(cfg: Any, api_filter: str, k: int, seed: Optional[int] = None) -> List[Dict]:
    """Reservoir sampling — statistically valid random sample from ALL results.

    Algorithm: https://en.wikipedia.org/wiki/Reservoir_sampling
    Every paper has equal probability (k/n) of being selected, regardless of order.

    WARNING: This scans ALL matching papers, so it's slow for large datasets.
    For 500k+ papers, expect 5-10 minutes runtime.

    Args:
        cfg: Configuration
        api_filter: OpenAlex API filter string
        k: Sample size
        seed: Random seed for reproducibility (optional)

    Returns:
        List of k randomly sampled papers
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(int(datetime.now().timestamp() * 1000))

    reservoir: List[Dict] = []
    n = 0  # Total papers seen

    async with AsyncOpenAlexClient(
        api_key=cfg.api_key,
        email=cfg.email,
        per_page=cfg.per_page,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        concurrent_requests=cfg.concurrent_requests,
    ) as client:
        cursor = "*"
        console.print("[dim]Scanning all matching papers for true random sampling...[/dim]")

        while cursor:
            data = await client.fetch_page(
                api_filter, cursor,
                extra_params={"select": SAMPLE_FIELDS},
            )
            if not data:
                break
            batch = data.get("results", [])
            if not batch:
                break

            # Reservoir sampling algorithm
            for paper in batch:
                n += 1
                if len(reservoir) < k:
                    # Fill reservoir first
                    reservoir.append(paper)
                else:
                    # Replace with probability k/n
                    j = random.randint(0, n - 1)
                    if j < k:
                        reservoir[j] = paper

            cursor = data["meta"].get("next_cursor")

            # Progress update every 10k papers
            if n % 10000 == 0:
                console.print(f"[dim]  Scanned {n:,} papers...[/dim]")

    console.print(f"[dim]  Total papers scanned: {n:,}[/dim]")
    console.print(f"[dim]  Sample size: {len(reservoir)}[/dim]")

    # Shuffle before returning
    random.shuffle(reservoir)
    return reservoir


async def _fetch_sample_fast(cfg: Any, api_filter: str, k: int, seed: Optional[int] = None) -> List[Dict]:
    """Fast sampling using OpenAlex's sample parameter (BIASED - use with caution).

    ⚠️ WARNING: OpenAlex's sample parameter samples from their search index,
    NOT from your actual filtered results. This can produce biased samples
    that don't represent the population distribution.

    Use only for quick exploration, NOT for statistical validation.

    For statistically valid sampling, use _reservoir_sample() instead.
    """
    results: List[Dict] = []

    async with AsyncOpenAlexClient(
        api_key=cfg.api_key,
        email=cfg.email,
        per_page=min(k, 200),
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        concurrent_requests=cfg.concurrent_requests,
    ) as client:
        extra_params = {
            "select": SAMPLE_FIELDS,
            "sample": k,
        }
        if seed is not None:
            extra_params["seed"] = seed

        data = await client.fetch_page(api_filter, cursor="*", extra_params=extra_params)
        if data:
            results = data.get("results", [])

    return results
