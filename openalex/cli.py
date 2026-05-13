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
     12. openalex enrich-crossref       Restore raw_affiliation_string from CrossRef (DOI lookup)
     13. openalex impute-affiliation    Impute missing institution + country from raw affiliations
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
cli.add_command(enrich_crossref_command)
cli.add_command(impute_affiliation_command)
cli.add_command(export_format_command)
