import polars as pl
import json
from pathlib import Path

def convert_jsonl_to_csv(input_path: str, output_path: str):
    print(f"Reading {input_path}...")
    
    # Read the JSONL file as a list of dictionaries first to handle complex nesting easily
    # for 1000 lines, this is perfectly fine.
    data = []
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    processed_data = []
    for entry in data:
        lens_id = entry.get("lens_id")
        biblio = entry.get("biblio", {})
        
        # Extract English Title
        titles = biblio.get("invention_title", [])
        title = next((t.get("text") for t in titles if t.get("lang") == "en"), None)
        if not title and titles:
            title = titles[0].get("text")
            
        # Extract English Abstract
        abstracts = entry.get("abstract", [])
        abstract = next((a.get("text") for a in abstracts if a.get("lang") == "en"), None)
        if not abstract and abstracts:
            abstract = abstracts[0].get("text")
            
        # Extract CPC Symbols
        cpc_data = biblio.get("classifications_cpc", {}).get("classifications", [])
        cpc = ";".join([c.get("symbol") for c in cpc_data if c.get("symbol")])
        
        # Extract IPC Symbols
        ipc_data = biblio.get("classifications_ipcr", {}).get("classifications", [])
        ipc = ";".join([c.get("symbol") for c in ipc_data if c.get("symbol")])
        
        # Other basic fields
        jurisdiction = entry.get("jurisdiction")
        doc_number = entry.get("doc_number")
        date_published = entry.get("date_published")
        
        processed_data.append({
            "lens_id": lens_id,
            "title": title,
            "abstract": abstract,
            "cpc": cpc,
            "ipc": ipc,
            "jurisdiction": jurisdiction,
            "doc_number": doc_number,
            "date_published": date_published
        })
    
    df = pl.DataFrame(processed_data)
    
    print(f"Writing to {output_path}...")
    df.write_csv(output_path)
    print("Done!")

if __name__ == "__main__":
    input_file = "data/green-hyd-v1.jsonl"
    output_file = "data/green-hyd-v1.csv"
    convert_jsonl_to_csv(input_file, output_file)
