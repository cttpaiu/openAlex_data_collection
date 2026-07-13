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
@click.option("--reservoir", is_flag=True, help="Use reservoir sampling (slow but statistically valid - scans all papers)")
def sample_command(size: int, config_path: str, no_topics: bool, yes: bool, save_csv: bool, output: str, seed: int, reservoir: bool) -> None:
    """Take a random validation sample from matching papers.

    Default: Uses OpenAlex's built-in sample parameter (fast).
    With --reservoir: Scans all papers for statistically valid random sampling.
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

    # Get total count for progress display
    console.print("[dim]Getting total paper count...[/dim]")
    total_count = asyncio.run(_get_count(cfg, api_filter))
    console.print(f"[dim]Total matching papers: {total_count:,}[/dim]\n")

    if not no_topics and topics and len(topics) > 40:
        console.print(f"[dim]Topic list contains {len(topics)} topics. Using proportional sampling across chunks...[/dim]")
        sampled = asyncio.run(_fetch_sample_proportional(cfg, keywords, topics, size, seed))
    elif reservoir:
        console.print(f"\n[dim]Using reservoir sampling (scans all {total_count:,} papers...)[/dim]")
        sampled = asyncio.run(_reservoir_sample(cfg, api_filter, size, seed))
    else:
        # Default: Use OpenAlex sample API (fast)
        sampled = asyncio.run(_fetch_sample_fast(cfg, api_filter, size, seed))

    if not sampled:
        console.print("[yellow]⚠ No papers returned.[/yellow]")
        return

    # Classify the papers into Relevant vs Noise
    console.print("[dim]Analyzing sample relevance (Relevant vs Noise)...[/dim]")
    relevant_count = 0
    noise_count = 0
    
    for idx, p in enumerate(sampled, 1):
        title = p.get("title") or ""
        abstract = reconstruct_abstract(p.get("abstract_inverted_index"))
        topic_name = (p.get("primary_topic") or {}).get("display_name") or ""
        
        relevance, rationale = classify_paper(title, abstract, topic_name, cfg)
        p["relevance"] = relevance
        p["rationale"] = rationale
        
        if relevance == "Relevant":
            relevant_count += 1
        else:
            noise_count += 1
            
        if idx % 10 == 0 or idx == len(sampled):
            console.print(f"[dim]  Classified {idx}/{len(sampled)} papers...[/dim]")

    _print_sample_table(sampled)

    pct_relevant = (relevant_count / len(sampled)) * 100 if sampled else 0
    pct_noise = (noise_count / len(sampled)) * 100 if sampled else 0

    console.print(
        Panel(
            f"  [dim]Total checked:[/dim]  [bold]{len(sampled)}[/bold]\n"
            f"  [dim]Relevant:[/dim]       [bold green]{relevant_count} ({pct_relevant:.1f}%)[/bold green]\n"
            f"  [dim]Noise:[/dim]          [bold red]{noise_count} ({pct_noise:.1f}%)[/bold red]",
            title="[bold green]Sample Analysis Summary[/bold green]",
            expand=False,
        )
    )

    if save_csv:
        filename = output or f"sample_{datetime.now().strftime('%Y%m%d')}.csv"
        _save_sample_csv(sampled, filename)
    elif not yes:
        import questionary
        if questionary.confirm("\nSave sample to CSV?", default=True).ask():
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


async def _get_count(cfg: Any, api_filter: str) -> int:
    """Get total count of matching papers."""
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        return await client.get_total_count(api_filter)


def _print_sample_table(papers: List[Dict]) -> None:
    table = Table(title=f"Random Sample ({len(papers)} papers)", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Year", width=6)
    table.add_column("Title", style="white", max_width=45)
    table.add_column("Topic", style="cyan", max_width=25)
    table.add_column("Cited", justify="right", style="green")
    table.add_column("Relevance", style="bold")

    for i, p in enumerate(papers, 1):
        topic = (p.get("primary_topic") or {}).get("display_name", "—")
        title = (p.get("title") or "Untitled")[:54]
        rel = p.get("relevance", "—")
        rel_style = "[green]Relevant[/green]" if rel == "Relevant" else "[red]Noise[/red]" if rel == "Noise" else "—"
        table.add_row(
            str(i),
            str(p.get("publication_year") or "—"),
            title,
            topic[:24],
            str(p.get("cited_by_count") or 0),
            rel_style,
        )

    console.print(table)



def _save_sample_csv(papers: List[Dict], filename: str) -> None:
    path = Path(filename)
    fieldnames = [
        "id", "doi", "title", "publication_year", "type",
        "topic_id", "topic_name", "abstract_text",
        "cited_by_count", "fwci", "institutions_count", "countries_count",
        "relevance", "rationale",
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
                "relevance": p.get("relevance", "Unclassified"),
                "rationale": p.get("rationale", ""),
            })
    console.print(f"[green]✓ Saved to [cyan]{filename}[/cyan][/green]")


def classify_paper(title: str, abstract: str, topic_name: str, cfg: Any) -> tuple[str, str]:
    title = title or ""
    abstract = abstract or ""
    topic_name = topic_name or ""
    
    # Heuristic fallback (fast, local, and reliable)
    text = (title + " " + abstract + " " + topic_name).lower()
    
    battery_keywords = [
        "battery", "batteries", "li-ion", "lithium-ion", "lithium-sulfur", "li-s",
        "state of charge", "state of health", "soc", "soh", "bms", "bess",
        "remaining useful life", "rul", "equivalent-circuit model", "cell balancing",
        "state of energy", "soe", "battery modeling", "battery model",
        "charge estimation", "health estimation"
    ]
    
    noise_keywords = [
        "solar cell", "solar cells", "photovoltaic cell", "photovoltaic cells",
        "fuel cell", "fuel cells", "microbial fuel",
        "biological cell", "biological cells", "cell division", "cancer cell", "cancer cells",
        "cellular network", "cellular networks", "5g", "lte", "cellular neural network",
        "cellular manufacturing", "cell biology"
    ]
    
    has_battery = any(kw in text for kw in battery_keywords)
    has_noise = any(kw in text for kw in noise_keywords)
    
    if has_battery and not has_noise:
        return "Relevant", "Matches battery keywords and contains no obvious noise keywords."
    elif has_noise and not has_battery:
        return "Noise", "Contains explicit noise keywords (e.g. solar/fuel cells, biological cells, or telecommunications) and lacks battery terms."
    elif has_noise and has_battery:
        if "battery management" in text or "bms" in text or "bess" in text:
            return "Relevant", "Contains noise keywords but includes strong battery system terms like BMS/BESS."
        return "Noise", "Contains both battery and noise keywords, likely focusing on fuel cell/solar cell hybrid systems or general power system applications."
    else:
        if "energy storage" in text and ("grid" in text or "microgrid" in text or "renewable" in text):
            return "Relevant", "Mentions energy storage in grid/microgrid context."
        return "Noise", "Does not contain strong battery or energy storage indicators."


async def _fetch_sample_proportional(cfg: Any, keywords: str, topics: list[str], k: int, seed: Optional[int] = None) -> List[Dict]:
    """Proportionally sample from large topic list to prevent URL size overflow."""
    chunk_size = 18
    chunks = [topics[i:i + chunk_size] for i in range(0, len(topics), chunk_size)]
    
    chunk_counts = []
    total_count = 0
    
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        per_page=1,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        for idx, chunk in enumerate(chunks):
            chunk_filter = build_filter(
                keywords=keywords,
                topics=chunk,
                date_from=cfg.date_from,
                date_to=cfg.date_to,
                doc_types=cfg.doc_types,
            )
            count = await client.get_total_count(chunk_filter)
            chunk_counts.append(count)
            total_count += count
            
    sample_sizes = []
    allocated = 0
    for count in chunk_counts:
        if total_count > 0:
            sz = int(round(k * count / total_count))
        else:
            sz = 0
        sample_sizes.append(sz)
        allocated += sz
        
    diff = k - allocated
    if diff != 0 and total_count > 0:
        max_idx = chunk_counts.index(max(chunk_counts))
        sample_sizes[max_idx] += diff
        
    sampled_papers = []
    
    if seed is None:
        base_seed = random.randint(1, 1000000)
    else:
        base_seed = seed
        
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        per_page=200,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
    ) as client:
        for idx, (chunk, sz) in enumerate(zip(chunks, sample_sizes)):
            if sz <= 0:
                continue
                
            chunk_filter = build_filter(
                keywords=keywords,
                topics=chunk,
                date_from=cfg.date_from,
                date_to=cfg.date_to,
                doc_types=cfg.doc_types,
            )
            
            extra_params = {
                "select": SAMPLE_FIELDS,
                "sample": sz,
                "seed": base_seed + idx,
            }
            
            data = await client.fetch_page(chunk_filter, cursor="*", extra_params=extra_params)
            if data and "results" in data:
                sampled_papers.extend(data["results"])
                
    random.shuffle(sampled_papers)
    return sampled_papers[:k]



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
        api_keys=cfg.api_keys,
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
    """Use OpenAlex API's built-in sample parameter (default method).

    OpenAlex docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/sample-entity-lists

    Fast sampling using the native sample parameter.
    For statistically rigorous sampling, use --reservoir flag instead.
    
    Note: OpenAlex returns max 200 papers per page, so we make multiple requests
    with different random seeds to get k > 200 papers.
    
    By default, uses random seeds for different samples each run.
    Use --seed flag for reproducible samples.
    """
    results: List[Dict] = []
    
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email,
        per_page=200,  # Always use max page size
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        concurrent_requests=cfg.concurrent_requests,
    ) as client:
        # For k > 200, we need to make multiple requests with different seeds
        # because OpenAlex sample parameter doesn't support cursor pagination
        pages_needed = (k // 200) + 1
        
        # Generate random base seed if not provided (ensures different sample each run)
        if seed is None:
            # Use microseconds for high entropy - guarantees different seed each run
            base_seed = int(datetime.now().timestamp() * 1_000_000) % (2**31)
            console.print(f"[dim]Using random seed: {base_seed} (use --seed {base_seed} to reproduce this sample)[/dim]\n")
        else:
            base_seed = seed
            console.print(f"[dim]Using fixed seed: {seed} (reproducible sample)[/dim]\n")
        
        for page in range(pages_needed):
            extra_params = {
                "select": SAMPLE_FIELDS,
                "sample": min(200, k - len(results)),
                "seed": base_seed + page,  # Different seed for each page
            }

            data = await client.fetch_page(api_filter, cursor="*", extra_params=extra_params)
            if not data:
                break
            batch = data.get("results", [])
            if not batch:
                break
            
            # Filter out duplicates by ID
            existing_ids = {p["id"] for p in results}
            new_papers = [p for p in batch if p["id"] not in existing_ids]
            results.extend(new_papers)
            
            console.print(f"[dim]  Fetched {len(results)}/{k} papers...[/dim]")
            
            if len(results) >= k:
                break

    return results[:k]
