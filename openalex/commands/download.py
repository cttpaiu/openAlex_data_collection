"""Command: download — full async JSONL download of matching papers."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openalex.api_client import AsyncOpenAlexClient, BufferedWriter
from openalex.config import load_config
from openalex.utils import build_filter
from openalex.validator import check_and_print_keyword_errors, validate_topic_format

console = Console()

BYTES_PER_PAPER = 8_700  # ~8.5 KB average per paper
MIN_FREE_BYTES_MULTIPLIER = 2


# ---------------------------------------------------------------------------
# Cursor-progress helpers
# ---------------------------------------------------------------------------

def _progress_file(output_path: str) -> Path:
    """Return the path of the cursor-progress sidecar file."""
    return Path(output_path).with_suffix(".download_progress.json")


def _load_cursor_state(output_path: str) -> Dict[str, Any]:
    """Load saved cursor positions; return empty dict if none exist."""
    pf = _progress_file(output_path)
    if pf.exists():
        try:
            with open(pf, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"batches": {}}


def _save_cursor_state(output_path: str, state: Dict[str, Any]) -> None:
    """Atomically write cursor state to the sidecar file."""
    pf = _progress_file(output_path)
    tmp = pf.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(pf)


def _mark_batch_done(output_path: str, state: Dict[str, Any], batch_key: str) -> None:
    """Mark a batch as fully completed (cursor = null)."""
    state["batches"].setdefault(batch_key, {})["last_cursor"] = None
    state["batches"][batch_key]["done"] = True
    _save_cursor_state(output_path, state)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _get_count(cfg: Any, api_filter: str) -> int:
    try:
        async with AsyncOpenAlexClient(
            api_key=cfg.api_key,
            email=cfg.email,
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
        ) as client:
            count = await client.get_total_count(api_filter)
            if count == 0:
                console.print("[yellow]⚠ API returned 0 papers - check API key and filters[/yellow]")
            return count
    except Exception as e:
        console.print(f"[red]✗ Error getting paper count: {e}[/red]")
        console.print("[dim]Check your API key in config/collection.yml[/dim]")
        return 0


async def _run_download(cfg: Any, api_filter: str, output_path: str, total: int, topics: List[str]) -> None:
    progress_counter: Counter = Counter()
    start_time = time.time()

    # Load cursor state saved from a previous run
    cursor_state = _load_cursor_state(output_path)

    # Count already-written papers for the progress bar starting point
    resume_count = 0
    output_file = Path(output_path)
    if output_file.exists():
        console.print("[yellow]⚠ Output file exists - resuming from saved cursor positions[/yellow]")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    resume_count += 1
        console.print(f"[dim]  Found {resume_count:,} existing papers[/dim]\n")

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
        progress_counter["collected"] = resume_count
        progress.update(task, completed=resume_count)

        async with BufferedWriter(output_path, mode="a") as writer:
            # Build batches
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
                    _process_batch(
                        client=client,
                        batch=batch,
                        batch_index=i,
                        base_filter=api_filter,
                        writer=writer,
                        counter=progress_counter,
                        semaphore=semaphore,
                        all_topics=topics,
                        output_path=output_path,
                        cursor_state=cursor_state,
                    )
                    for i, batch in enumerate(batches)
                ]
                update_task = asyncio.create_task(
                    _update_progress(progress, task, progress_counter, total)
                )
                await asyncio.gather(*tasks)
                update_task.cancel()

    elapsed = time.time() - start_time
    collected = progress_counter["collected"]
    new_papers = collected - resume_count
    console.print(
        f"\n[bold green]✓ Download complete![/bold green]\n"
        f"  Papers collected: [green]{collected:,}[/green] ([dim]+{new_papers:,} new, {resume_count:,} existing[/dim])\n"
        f"  Output file:      [cyan]{output_path}[/cyan]\n"
        f"  Time elapsed:     [dim]{elapsed / 60:.1f} min[/dim]"
    )

    # Clean up sidecar file on successful completion if all batches are done
    pf = _progress_file(output_path)
    all_done = all(
        v.get("done") for v in cursor_state.get("batches", {}).values()
    )
    if all_done and pf.exists():
        pf.unlink()
        console.print("[dim]  Progress file removed (all batches complete).[/dim]")


async def _process_batch(
    client: AsyncOpenAlexClient,
    batch: Optional[list],
    batch_index: int,
    base_filter: str,
    writer: BufferedWriter,
    counter: Counter,
    semaphore: asyncio.Semaphore,
    all_topics: list,
    output_path: str,
    cursor_state: Dict[str, Any],
) -> None:
    """Process one topic batch (or full download if no topics), resuming from saved cursor."""

    batch_key = str(batch_index)

    # Skip this batch if it was already fully completed in a previous run
    batch_info = cursor_state["batches"].get(batch_key, {})
    if batch_info.get("done"):
        console.print(f"[dim]  Batch {batch_index}: already complete, skipping.[/dim]")
        return

    # Resume from the last saved cursor, or start from the beginning
    saved_cursor: str = batch_info.get("last_cursor") or "*"

    if batch is not None:
        topic_str = "|".join(batch)
        filter_str = base_filter.replace(
            f"primary_topic.id:{'|'.join(all_topics)}" if all_topics else "",
            f"primary_topic.id:{topic_str}",
        )
    else:
        filter_str = base_filter

    if saved_cursor != "*":
        console.print(f"[dim]  Batch {batch_index}: resuming from saved cursor (skipping already-fetched pages).[/dim]")

    cursor: Optional[str] = saved_cursor
    collected_in_batch = 0

    async with semaphore:
        while cursor:
            data = await client.fetch_page(filter_str, cursor)
            if not data:
                console.print(f"[red]✗ API returned empty response at batch {batch_index}[/red]")
                break

            results = data.get("results", [])
            if not results:
                break

            # Advance cursor BEFORE writing so a crash mid-page is safe to retry
            next_cursor: Optional[str] = data["meta"].get("next_cursor")

            for paper in results:
                await writer.write(json.dumps(paper))
                counter["collected"] += 1
                collected_in_batch += 1

            # Persist the cursor we are about to move to
            cursor_state["batches"].setdefault(batch_key, {})
            cursor_state["batches"][batch_key]["last_cursor"] = next_cursor
            cursor_state["batches"][batch_key]["topics"] = batch  # informational
            _save_cursor_state(output_path, cursor_state)

            cursor = next_cursor

    # Mark batch fully done
    _mark_batch_done(output_path, cursor_state, batch_key)
    console.print(f"[dim]  Batch {batch_index}: collected {collected_in_batch:,} papers.[/dim]")


async def _update_progress(progress: Progress, task: Any, counter: Counter, total: int) -> None:
    while True:
        await asyncio.sleep(1)
        progress.update(task, completed=min(counter["collected"], total))
