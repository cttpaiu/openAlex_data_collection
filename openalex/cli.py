"""OpenAlex CLI — entry point. All commands registered here."""

import click

from openalex.config import init_command
from openalex.commands.search import search_command, search_filtered_command
from openalex.commands.check_anchor import check_anchor_command
from openalex.commands.topics import get_topics_command
from openalex.commands.sample import sample_command
from openalex.commands.download import download_command
from openalex.commands.database import convert_to_db_command
from openalex.commands.check_db import check_db_command
from openalex.commands.enrich_crossref import enrich_crossref_command
from openalex.commands.impute_affiliation import impute_affiliation_command
from openalex.commands.impute_pdf import impute_pdf_command
from openalex.commands.export import export_format_command
from openalex.commands.validate import validate_command


@click.group()
@click.version_option(version="0.1.0", prog_name="openalex")
def cli() -> None:
    """OpenAlex quantum computing data collection pipeline.

    \b
    Workflow:
      1. openalex init                  Create config template files
      2. openalex validate              Check keywords.txt and topics.txt
      3. openalex search                Keyword count + anchor check
      4. openalex get-topics            What topics appear in results?
      5. openalex get-topics --details --csv  Topic counts → CSV
      6. openalex search-filtered       Keyword + topic count + anchor check
      7. openalex check-anchor          Explicit anchor coverage check
      8. openalex sample --size 385     Random validation sample
      9. openalex download              Download all papers → JSONL
     10. openalex convert-to-db         JSONL → DuckDB
     11. openalex check-db              Paper-centric coverage health report
     12. openalex impute <source>       Imputation pipeline — see `openalex impute --help`

    \b
    Imputation sources (pick one):
      openalex impute crossref          Restore raw_affiliation_string from CrossRef
      openalex impute llm               LLM-extract institution + country from raw affiliations
      openalex impute pdf               Resolve PDF URL via Unpaywall/arXiv, download, extract page-1 text (LLM step pending)
    """


@click.group("impute")
def impute_group() -> None:
    """Imputation pipeline — pick a source for the missing institution/country data.

    \b
    Sources:
      crossref   Pull raw_affiliation_string from CrossRef by DOI (fast, low yield).
      llm        Three-stage LLM pipeline over raw_affiliation_string.
      pdf        Fetch the paper's PDF (Unpaywall/arXiv) and extract page-1 text into a cache. LLM extraction lands in a follow-up.
    """


cli.add_command(init_command)
cli.add_command(validate_command)
cli.add_command(search_command)
cli.add_command(search_filtered_command)
cli.add_command(check_anchor_command)
cli.add_command(get_topics_command)
cli.add_command(sample_command)
cli.add_command(download_command)
cli.add_command(convert_to_db_command)
cli.add_command(check_db_command)
cli.add_command(export_format_command)

# Unified imputation subgroup. Names come from each command's own decorator
# (@click.command("crossref") / @click.command("llm")) so help/usage output
# consistently shows "openalex impute crossref" / "openalex impute llm".
impute_group.add_command(enrich_crossref_command)
impute_group.add_command(impute_affiliation_command)
impute_group.add_command(impute_pdf_command)
cli.add_command(impute_group)

# Deprecation aliases — keep the old top-level names working for one release.
# Explicit name= override since the underlying commands now identify as
# "crossref" / "llm".
cli.add_command(enrich_crossref_command, name="enrich-crossref")
cli.add_command(impute_affiliation_command, name="impute-affiliation")
