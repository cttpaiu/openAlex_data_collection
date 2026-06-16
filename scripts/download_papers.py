import asyncio
import polars as pl
from openalex.api_client import AsyncOpenAlexClient
from openalex.config import load_config
from openalex.utils import reconstruct_abstract

DOIS = [
    "10.1016/j.renene.2019.04.130", "10.1016/j.apenergy.2011.04.025",
    "10.1016/j.rser.2020.110296", "10.1038/s41893-019-0364-5",
    "10.1016/j.rser.2026.115201", "10.1016/j.jece.2025.112045",
    "10.3390/en18174776", "10.1016/j.enpol.2014.03.008",
    "10.1016/j.solener.2021.05.012", "10.1016/j.rser.2024.114387",
    "10.1016/j.agrformet.2013.04.011", "10.1002/ppp3.70179",
    "10.1016/j.fcr.2017.06.014", "10.1016/j.solener.2019.03.042",
    "10.1016/j.scienta.2018.09.022", "10.1016/j.eja.2021.126310",
    "10.1111/gcb.14493", "10.1016/j.apenergy.2016.08.135",
    "10.1016/j.scienta.2022.111245", "10.1016/j.fcr.2015.02.009",
    "10.1016/j.jaridenv.2018.01.002", "10.1016/j.agrformet.2018.04.015",
    "10.1016/j.ecoleng.2020.105958", "10.1016/j.solener.2019.09.077",
    "10.1016/j.geoderma.2022.115890", "10.1016/j.jweia.2021.104689",
    "10.1016/j.catena.2021.105612", "10.1016/j.jaridenv.2020.104211",
    "10.1016/j.renene.2018.06.079", "10.1016/j.apenergy.2020.114953",
    "10.1038/s41467-023-41871-3", "10.1016/j.solener.2020.08.031",
    "10.1016/j.solener.2017.03.045", "10.1016/j.joule.2020.04.012",
    "10.1016/j.enbuild.2016.03.024", "10.1016/j.enpol.2019.111005",
    "10.1016/j.compag.2022.106981", "10.1016/j.mattod.2021.02.008",
    "10.1016/j.renene.2021.09.043", "10.1016/j.esd.2025.101630",
    "10.1016/j.landusepol.2025.107512", "10.1016/j.enpol.2021.112440",
    "10.1016/j.biocon.2020.108432", "10.1016/j.landusepol.2018.04.053",
    "10.1016/j.jenvman.2021.113200", "10.3390/en18246417",
    "10.1016/j.solener.2020.11.015", "10.1007/s40003-026-00912-x",
    "10.1016/j.smallrumres.2021.106403"
]

async def download_to_csv():
    cfg = load_config("config/collection.yml")
    output_file = "data/agrivoltaics_papers.csv"
    
    print(f"Fetching metadata for {len(DOIS)} DOIs from OpenAlex...")
    
    async with AsyncOpenAlexClient(
        api_keys=cfg.api_keys,
        email=cfg.email
    ) as client:
        # OpenAlex filter by multiple DOIs
        doi_filter = f"doi:{'|'.join(DOIS)}"
        records = await client.fetch_all_pages(doi_filter)
        
    print(f"Found {len(records)} records.")
    
    # Flatten records for CSV
    flattened = []
    for r in records:
        authors = ", ".join([a.get("author", {}).get("display_name", "") for a in r.get("authorships", [])])
        institutions = ", ".join(set([inst.get("display_name", "") for auth in r.get("authorships", []) for inst in auth.get("institutions", [])]))
        
        # Backward citations (referenced_works) are OpenAlex IDs
        references = "|".join(r.get("referenced_works", []))
        
        flattened.append({
            "id": r.get("id"),
            "doi": r.get("doi"),
            "title": r.get("title"),
            "publication_year": r.get("publication_year"),
            "journal": r.get("primary_location", {}).get("source", {}).get("display_name"),
            "authors": authors,
            "institutions": institutions,
            "cited_by_count": r.get("cited_by_count"),
            "reference_count": len(r.get("referenced_works", [])),
            "references": references,
            "type": r.get("type"),
            "abstract": reconstruct_abstract(r.get("abstract_inverted_index"))
        })
        
    if flattened:
        df = pl.DataFrame(flattened)
        df.write_csv(output_file)
        print(f"Successfully saved to {output_file}")
    else:
        print("No records found to save.")

if __name__ == "__main__":
    asyncio.run(download_to_csv())
