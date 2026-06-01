"""Command: build-categorized-query — assemble bounded boolean query (Phase 2).

Bounding logic:
    (Core OR Methods) OR (Mechanisms AND Applications)

Core technology and methods are high-precision and stand alone. Mechanisms and
applications are broad and produce noise on their own, so they are AND'd
together to bound the result set.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from openalex.validator import check_and_print_keyword_errors

console = Console()

CATEGORY_KEYS = (
    "1_core_technology",
    "2_methods_tools",
    "3_mechanisms",
    "4_applications",
)


def _format_group(terms: list[str]) -> str:
    """Wrap multi-token / hyphenated terms in quotes, OR-join, parenthesise."""
    if not terms:
        return ""
    formatted: list[str] = []
    for term in terms:
        t = term.strip()
        if not t:
            continue
        if " " in t or "-" in t:
            formatted.append(f'"{t}"')
        else:
            formatted.append(t)
    if not formatted:
        return ""
    return "(" + " OR ".join(formatted) + ")"


def compile_bibliometric_query(config: dict) -> str:
    """Build the bounded boolean query from a four-bucket config dict."""
    core = _format_group(config.get("1_core_technology", []))
    methods = _format_group(config.get("2_methods_tools", []))
    mechanisms = _format_group(config.get("3_mechanisms", []))
    apps = _format_group(config.get("4_applications", []))

    parts: list[str] = []

    standalone = [g for g in (core, methods) if g]
    if standalone:
        parts.append("(" + " OR ".join(standalone) + ")")

    if mechanisms and apps:
        parts.append(f"({mechanisms} AND {apps})")
    elif mechanisms:
        # No applications to bound against — drop mechanisms (would flood results).
        console.print(
            "[yellow]⚠ Mechanisms present but no applications — skipping mechanisms "
            "to preserve precision.[/yellow]"
        )
    elif apps:
        console.print(
            "[yellow]⚠ Applications present but no mechanisms — skipping applications "
            "to preserve precision.[/yellow]"
        )

    if not parts:
        raise click.ClickException(
            "All category buckets are empty. Fill at least Core or Methods, or both "
            "Mechanisms and Applications."
        )

    final_query = " OR ".join(parts)
    return f"({final_query})"


def _load_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise click.ClickException(f"{path} must contain a JSON object at the top level.")
    for key in CATEGORY_KEYS:
        if key in raw and not isinstance(raw[key], list):
            raise click.ClickException(f"Bucket '{key}' must be a list of strings.")
    return raw


def _confirm_overwrite(path: Path, force: bool) -> bool:
    if not path.exists() or force:
        return True
    import questionary

    return bool(questionary.confirm(f"{path} exists. Overwrite?", default=False).ask())


@click.command("build-categorized-query")
@click.argument("config_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("config/keywords.txt"),
    show_default=True,
    help="Where to write the compiled boolean query",
)
@click.option("--no-write", is_flag=True, help="Print only, skip file write")
@click.option("--force", is_flag=True, help="Overwrite output file without prompting")
def build_categorized_query_command(
    config_json: Path,
    output_path: Path,
    no_write: bool,
    force: bool,
) -> None:
    """Phase 2 — compile a bounded boolean query from a bucketed JSON config.

    Reads CONFIG_JSON (produced by `openalex extract-keywords` and then hand-edited
    to bucket each term) and emits the boolean query using the bounding logic:

        (Core OR Methods) OR (Mechanisms AND Applications)
    """
    config = _load_config(config_json)

    counts = {key: len(config.get(key, [])) for key in CATEGORY_KEYS}
    console.print(
        "[dim]Bucket counts: "
        + "  ".join(f"{key.split('_', 1)[1]}={counts[key]}" for key in CATEGORY_KEYS)
        + "[/dim]"
    )
    leftover = config.get("uncategorized_tfidf_candidates", [])
    if leftover:
        console.print(
            f"[yellow]⚠ {len(leftover)} terms still in 'uncategorized_tfidf_candidates' "
            "— they will be ignored.[/yellow]"
        )

    query = compile_bibliometric_query(config)
    console.print("\n[bold]Compiled query:[/bold]")
    console.print(query)

    valid = check_and_print_keyword_errors(query)
    if not valid:
        console.print("[yellow]⚠ Query failed validation — writing anyway for inspection.[/yellow]")

    if no_write:
        return

    if not _confirm_overwrite(output_path, force):
        console.print("[yellow]Skipped writing output file.[/yellow]")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Generated by openalex build-categorized-query from {config_json}\n"
        f"# Bounding logic: (Core OR Methods) OR (Mechanisms AND Applications)\n"
    )
    output_path.write_text(header + query + "\n", encoding="utf-8")
    console.print(f"[green]✓ Wrote query to [cyan]{output_path}[/cyan][/green]")
