"""Shared utilities: filter builder, abstract reconstruction, rich helpers."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def build_filter(
    keywords: str,
    topics: Optional[list[str]] = None,
    date_from: str = "2003-01-01",
    date_to: str = "2025-12-31",
    doc_types: Optional[list[str]] = None,
) -> str:
    """Build an OpenAlex API filter string."""
    parts = [f"title_and_abstract.search:{keywords}"]

    if topics:
        parts.append(f"primary_topic.id:{'|'.join(topics)}")

    parts.append(f"from_publication_date:{date_from}")
    parts.append(f"to_publication_date:{date_to}")

    if doc_types:
        parts.append(f"type:{'|'.join(doc_types)}")

    return ",".join(parts)


def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Rebuild plain text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    try:
        word_positions = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]
        return " ".join(w for _, w in sorted(word_positions))
    except Exception:
        return ""


def print_search_result_panel(
    title: str,
    count: int,
    keywords_file: str,
    date_from: str,
    date_to: str,
    doc_types: list[str],
    topics_count: Optional[int] = None,
) -> None:
    """Print a styled result panel for search commands."""
    doc_str = ", ".join(doc_types)
    lines = [
        f"  [dim]Query file:[/dim]  [cyan]{keywords_file}[/cyan]",
    ]
    if topics_count is not None:
        lines.append(f"  [dim]Topics:[/dim]      [cyan]{topics_count} IDs from topics file[/cyan]")
    lines += [
        f"  [dim]Date range:[/dim]  [cyan]{date_from}[/cyan] → [cyan]{date_to}[/cyan]",
        f"  [dim]Doc types:[/dim]   [cyan]{doc_str}[/cyan]",
        "",
        f"  [bold green]✓ Total records: {count:,}[/bold green]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold green]{title}[/bold green]",
            expand=False,
        )
    )


def format_number(n: int) -> str:
    return f"{n:,}"
