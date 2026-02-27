"""Command: download — full async JSONL download of matching papers."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openalex.api_client import AsyncOpenAlexClient, BufferedWriter
from openalex.config import load_config
from openalex.utils import build_filter
from openalex.validator import check_and_print_keyword_errors, validate_topic_format

console = Console()

BYTES_PER_PAPER = 8_700  # ~8.5 KB average per paper
MIN_FREE_BYTES_MULTIPLIER = 2


@click.command("download")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--no-topics", is_flag=True, help="Download using keywords only (no topic filter)")
@click.option("--output", "-o", default=None, help="Output JSONL filename (prompts if not set)")
def download_command(config_path: str, no_topics: bool, output: str) -> None:
    """Download all matching papers to a JSONL file."""
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
    if not no_topics:
        raw = cfg.get_topics()
        topics = [t for t in raw if validate_topic_format(t)]

    api_filter = build_filter(
        keywords=keywords,
        topics=topics if topics else None,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        doc_types=cfg.doc_types,
    )

    # Pre-flight: get total count
    console.print("[dim]Estimating total papers...[/dim]")
    total = asyncio.run(_get_count(cfg, api_filter))
    estimated_mb = (total * BYTES_PER_PAPER) / (1024 * 1024)

    console.print(
        Panel(
            f"  [dim]Total papers:[/dim]  [bold green]{total:,}[/bold green]\n"
            f"  [dim]Estimated size:[/dim] [cyan]~{estimated_mb:.0f} MB[/cyan]\n"
            f"  [dim]Topics filter:[/dim] [cyan]{len(topics)} IDs[/cyan]" if topics else
            f"  [dim]Total papers:[/dim]  [bold green]{total:,}[/bold green]\n"
            f"  [dim]Estimated size:[/dim] [cyan]~{estimated_mb:.0f} MB[/cyan]\n"
            f"  [dim]Topics filter:[/dim] [cyan]None (keywords only)[/cyan]",
            title="[bold]Pre-flight Check[/bold]",
            expand=False,
        )
    )

    # Check disk space
    import shutil
    out_dir = Path(cfg.jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(out_dir).free
    needed = total * BYTES_PER_PAPER * MIN_FREE_BYTES_MULTIPLIER
    if free_bytes < needed:
        console.print(
            f"[yellow]⚠ Low disk space: {free_bytes / (1024**3):.1f} GB free, "
            f"need ~{needed / (1024**3):.1f} GB[/yellow]"
        )

    import questionary
    if not questionary.confirm("Proceed with download?", default=True).ask():
        console.print("[dim]Cancelled.[/dim]")
        return

    if not output:
        default_name = f"openalex_{datetime.now().strftime('%Y%m%d')}.jsonl"
        output = questionary.text("Output filename:", default=default_name).ask() or default_name

    output_path = str(out_dir / output) if not Path(output).is_absolute() and "/" not in output else output

    console.print(f"\n[bold]Downloading to:[/bold] [cyan]{output_path}[/cyan]\n")

    asyncio.run(_run_download(cfg, api_filter, output_path, total, topics))


async def _get_count(cfg: Any, api_filter: str) -> int:
    async with AsyncOpenAlexClient(
        api_key=cfg.api_key, email=cfg.email, max_retries=cfg.max_retries, retry_delay=cfg.retry_delay
    ) as client:
        return await client.get_total_count(api_filter)


async def _run_download(cfg: Any, api_filter: str, output_path: str, total: int, topics: List[str]) -> None:
    progress_counter: Counter = Counter()
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[green]{task.completed:,}[/green]/[dim]{task.total:,}[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Downloading papers...", total=total)

        async with BufferedWriter(output_path) as writer:
            # Split topics into batches or use single stream
            if topics:
                batches = [
                    topics[i:i + cfg.batch_size_topics]
                    for i in range(0, len(topics), cfg.batch_size_topics)
                ]
            else:
                batches = [None]  # single stream, no topic batching

            semaphore = asyncio.Semaphore(cfg.concurrent_requests)

            async with AsyncOpenAlexClient(
                api_key=cfg.api_key,
                email=cfg.email,
                per_page=cfg.per_page,
                max_retries=cfg.max_retries,
                retry_delay=cfg.retry_delay,
                concurrent_requests=cfg.concurrent_requests,
            ) as client:
                tasks = [
                    _process_batch(client, batch, api_filter, writer, progress_counter, semaphore, topics)
                    for batch in batches
                ]
                # Update progress periodically
                update_task = asyncio.create_task(
                    _update_progress(progress, task, progress_counter, total)
                )
                await asyncio.gather(*tasks)
                update_task.cancel()

    elapsed = time.time() - start_time
    collected = progress_counter["collected"]
    console.print(
        f"\n[bold green]✓ Download complete![/bold green]\n"
        f"  Papers collected: [green]{collected:,}[/green]\n"
        f"  Output file:      [cyan]{output_path}[/cyan]\n"
        f"  Time elapsed:     [dim]{elapsed / 60:.1f} min[/dim]"
    )


async def _process_batch(
    client: AsyncOpenAlexClient,
    batch: list | None,
    base_filter: str,
    writer: BufferedWriter,
    counter: Counter,
    semaphore: asyncio.Semaphore,
    all_topics: list,
) -> None:
    """Process one topic batch (or full download if no topics)."""
    if batch is not None:
        topic_str = "|".join(batch)
        # Replace topics portion of filter
        parts = base_filter.split(",")
        # Rebuild with this batch's topics
        from openalex.utils import build_filter as _bf
        # Extract other parts (keywords, dates, types) from existing filter
        filter_str = base_filter.replace(
            f"primary_topic.id:{'|'.join(all_topics)}" if all_topics else "",
            f"primary_topic.id:{topic_str}",
        )
    else:
        filter_str = base_filter

    cursor = "*"
    async with semaphore:
        while cursor:
            data = await client.fetch_page(filter_str, cursor)
            if not data:
                break
            results = data.get("results", [])
            if not results:
                break
            cursor = data["meta"].get("next_cursor")
            for paper in results:
                await writer.write(json.dumps(paper))
                counter["collected"] += 1


async def _update_progress(progress: Progress, task: Any, counter: Counter, total: int) -> None:
    while True:
        await asyncio.sleep(1)
        progress.update(task, completed=min(counter["collected"], total))
