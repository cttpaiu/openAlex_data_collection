import asyncio
import click
from openalex.config import load_config
from openalex.api_client import AsyncOpenAlexClient
from rich.console import Console
from rich.table import Table

console = Console()

async def get_details(dois):
    cfg = load_config()
    results = []
    async with AsyncOpenAlexClient(cfg.api_keys, cfg.email) as client:
        for doi in dois:
            # Clean DOI
            clean_doi = doi.strip()
            if "doi.org/" in clean_doi:
                clean_doi = clean_doi.split("doi.org/")[-1]
            
            data = await client.fetch_page(f"doi:{clean_doi}")
            if data and data.get("results"):
                work = data["results"][0]
                results.append({
                    "doi": clean_doi,
                    "title": work.get("title"),
                    "concepts": [c.get("display_name") for c in work.get("concepts", [])[:5]],
                    "keywords": [k.get("display_name") for k in work.get("keywords", [])[:5]]
                })
            else:
                results.append({"doi": clean_doi, "title": "NOT FOUND", "concepts": [], "keywords": []})
    return results

@click.command()
@click.option("--limit", default=10, help="Limit number of DOIs to check")
def main(limit):
    cfg = load_config()
    anchors = cfg.get_anchors()
    # For this exercise, we'll just check the first 'limit' anchors that we know are missing
    # In a real scenario, we'd cross-reference with search results
    to_check = anchors[:limit]
    
    details = asyncio.run(get_details(to_check))
    
    table = Table(title="Anchor Paper Details")
    table.add_column("DOI")
    table.add_column("Title")
    table.add_column("Top Concepts")
    table.add_column("Top Keywords")
    
    for d in details:
        table.add_row(
            d["doi"],
            d["title"][:50] + "..." if d["title"] else "N/A",
            ", ".join(d["concepts"]),
            ", ".join(d["keywords"])
        )
    
    console.print(table)

if __name__ == "__main__":
    main()
