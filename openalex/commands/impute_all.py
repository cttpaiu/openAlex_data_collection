"""Command: impute-all — orchestrate full enrichment + imputation pipeline."""

from __future__ import annotations

import click
from rich.console import Console

from openalex.commands.enrich_crossref import enrich_crossref_command
from openalex.commands.enrich_openalex import enrich_openalex_command
from openalex.commands.enrich_semanticscholar import enrich_semanticscholar_command
from openalex.commands.impute_affiliation import impute_affiliation_command
from openalex.commands.impute_pdf import impute_pdf_command

console = Console()


@click.command("all")
@click.option("--config", "config_path", default="config/collection.yml", show_default=True)
@click.option("--db", "db_path", default=None, help="Path to DuckDB file")
@click.option("--dry-run", is_flag=True, help="Preview without DB writes")
@click.option("--limit", type=int, default=None, help="Cap eligible papers per step")
@click.option("--llm-provider", default=None, help="LLM provider (groq|ollama)")
@click.option("--llm-model", default=None, help="LLM model name")
@click.option("--groq-api-key", default=None, help="Groq API key")
@click.option("--s2-api-key", default=None, help="Semantic Scholar API key")
@click.option("--stages", default="1,2,3", show_default=True, help="Impute stages (1=inst, 2=backfill, 3=country)")
@click.pass_context
def impute_all_command(
    ctx: click.Context,
    config_path: str,
    db_path: str | None,
    dry_run: bool,
    limit: int | None,
    llm_provider: str | None,
    llm_model: str | None,
    groq_api_key: str | None,
    s2_api_key: str | None,
    stages: str,
) -> None:
    """Run full pipeline: OpenAlex sync → CrossRef → SemScholar → LLM Impute → PDF Fallback."""

    console.print("\n[bold cyan]Step 1: enrich-openalex (re-querying stubs + titles)[/bold cyan]")
    ctx.invoke(
        enrich_openalex_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=dry_run,
        limit=limit,
        concurrency=None,
    )

    console.print("\n[bold cyan]Step 2: enrich-crossref (raw affiliation strings)[/bold cyan]")
    ctx.invoke(
        enrich_crossref_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=dry_run,
        limit=limit,
        concurrent_requests=10,
        max_retries=3,
        retry_delay=2,
    )

    console.print("\n[bold cyan]Step 3: enrich-semanticscholar (author affiliations)[/bold cyan]")
    ctx.invoke(
        enrich_semanticscholar_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=dry_run,
        limit=limit,
        api_key=s2_api_key,
        concurrent_requests=1,
    )

    console.print("\n[bold cyan]Step 4: impute llm (LLM extraction pipeline)[/bold cyan]")
    ctx.invoke(
        impute_affiliation_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=dry_run,
        limit=limit,
        stages=stages,
        llm_fallback=True,
        groq_api_key=groq_api_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=None,
        llm_batch_size=20,
        llm_concurrency=2,
        llm_min_confidence=0.8,
        llm_max_tokens_stage1=1200,
        llm_max_tokens_stage2=400,
        llm_max_tokens_stage3=300,
        match_threshold=0.78,
    )

    console.print("\n[bold cyan]Step 5: impute pdf (Final Fallback for missing affiliations)[/bold cyan]")
    ctx.invoke(
        impute_pdf_command,
        config_path=config_path,
        db_path=db_path,
        dry_run=dry_run,
        limit=limit,
        source_csv="unpaywall,openalex,arxiv",
        concurrent_downloads=4,
        max_pdf_mb=30,
        cache_dir="data/cache/pdfs",
        skip_cache=False,
        request_timeout=60,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=None,
        groq_api_key=groq_api_key,
        llm_max_tokens=1500,
        llm_min_confidence=0.0,
        match_threshold=0.78,
    )

    console.print("\n[bold green]✓ Impute all complete.[/bold green]")
