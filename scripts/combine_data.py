import polars as pl
import duckdb
import os
from pathlib import Path
import re
import pycountry

# Database and file paths
DB_PATH = "data/db/Quantum_Sensing_and_Metrology.duckdb"
QSM_DIR = "data/QSM/"
OUTPUT_PATH = "data/combined_country_metrics.csv"

# Country mapping for QSM file names to 2-letter codes
COUNTRY_CODE_MAPPING = {
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Brazil": "BR",
    "Canada": "CA",
    "China": "CN",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "England": "GB",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Hong Kong": "CN",
    "India": "IN",
    "Iran": "IR",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Mexico": "MX",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Russia": "RU",
    "Saudi Arabia": "SA",
    "Scotland": "GB",
    "Wales": "GB",
    "Northern Ireland": "GB",
    "Singapore": "SG",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Turkey": "TR",
    "UAE": "AE",
    "UK": "GB",
    "US": "US",
}

def extract_year(date_str):
    """Extract 4-digit year from string."""
    if date_str is None:
        return None
    match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    if match:
        return int(match.group(0))
    return None

def normalize_bool(val):
    """Normalize various representations of boolean flags."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ('1', 'true', 'yes', 'y', 't', 'x')

def get_country_name(code):
    """Get full country name from ISO code with overrides."""
    overrides = {
        "CN": "China",
        "GB": "United Kingdom",
        "US": "United States",
        "KR": "South Korea",
        "RU": "Russia",
        "TW": "Taiwan",
        "IR": "Iran",
        "CZ": "Czech Republic",
        "AE": "United Arab Emirates",
        "TR": "Turkey",
        "HK": "China",
    }
    if code in overrides:
        return overrides[code]
    
    try:
        country = pycountry.countries.get(alpha_2=code)
        if country:
            # Some names have extra info in commas, e.g., "Iran, Islamic Republic of"
            name = country.name
            if "," in name:
                name = name.split(",")[0]
            return name
    except Exception:
        pass
    return code

def get_openalex_data():
    """Extract DOI, Year, Country Code, Top1, Top10 from DuckDB."""
    print("Extracting data from OpenAlex DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    
    query = """
    SELECT 
        LOWER(TRIM(p.doi)) as doi,
        p.publication_year as year,
        c.country_code,
        p.is_top_1_percent as top1,
        p.is_top_10_percent as top10
    FROM papers p
    JOIN contributions c ON p.id = c.paper_id
    WHERE c.country_code IS NOT NULL AND p.doi IS NOT NULL
    """
    df = con.execute(query).pl()
    con.close()
    
    # Normalize HK to CN and deduplicate
    df = (
        df.with_columns([
            pl.col("country_code").replace("HK", "CN")
        ])
        .unique(subset=["doi", "country_code"])
        .with_columns([
            pl.col("top1").fill_null(False).cast(pl.Boolean),
            pl.col("top10").fill_null(False).cast(pl.Boolean)
        ])
    )
    return df

def get_qsm_data():
    """Extract DOI, Year, Country Code, Top1, Top10 from QSM Excel files."""
    print("Extracting data from QSM Excel files...")
    all_dfs = []
    
    for filename in os.listdir(QSM_DIR):
        if not filename.endswith('.xlsx') or filename.startswith('1 QSM'):
            continue
            
        country_name_file = filename.replace('.xlsx', '')
        country_code = COUNTRY_CODE_MAPPING.get(country_name_file)
        
        if not country_code:
            continue
            
        file_path = os.path.join(QSM_DIR, filename)
        print(f"  Processing {filename} ({country_code})...")
        
        try:
            df = pl.read_excel(file_path)
            
            col_map = {
                'DOI': 'doi',
                'Publication Date': 'pub_date',
                'TOP 1%': 'top1',
                'TOP 10%': 'top10'
            }
            
            rename_dict = {orig: target for orig, target in col_map.items() if orig in df.columns}
            df = df.rename(rename_dict)
            
            for target in ['doi', 'pub_date', 'top1', 'top10']:
                if target not in df.columns:
                    if target == 'doi':
                        df = df.with_columns(pl.lit(None).alias('doi'))
                    elif target == 'pub_date':
                        df = df.with_columns(pl.lit(None).alias('pub_date'))
                    else:
                        df = df.with_columns(pl.lit(False).alias(target))

            df = (
                df.filter(pl.col("doi").is_not_null())
                .with_columns([
                    pl.col("doi").str.strip_chars().str.to_lowercase(),
                    pl.lit(country_code).alias("country_code"),
                    pl.col("pub_date").map_elements(extract_year, return_dtype=pl.Int64).alias("year")
                ])
                .with_columns([
                    pl.col("top1").map_elements(normalize_bool, return_dtype=pl.Boolean).alias("top1"),
                    pl.col("top10").map_elements(normalize_bool, return_dtype=pl.Boolean).alias("top10")
                ])
                .select(['doi', 'country_code', 'year', 'top1', 'top10'])
                .unique(subset=["doi", "country_code"])
            )
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if not all_dfs:
        return pl.DataFrame(schema={'doi': pl.String, 'country_code': pl.String, 'year': pl.Int64, 'top1': pl.Boolean, 'top10': pl.Boolean})
        
    return pl.concat(all_dfs)

def main():
    oa_df = get_openalex_data()
    qsm_df = get_qsm_data()
    
    print("Merging datasets...")
    # Join on DOI and country_code
    combined = oa_df.join(
        qsm_df, 
        on=["doi", "country_code"], 
        how="full", 
        suffix="_qsm",
        coalesce=True,
    )
    
    # Resolve Year and Boolean flags
    combined = combined.with_columns([
        pl.col("country_code").alias("code"),
        pl.coalesce(["year_qsm", "year"]).alias("final_year"),
        (pl.col("top1").fill_null(False) | pl.col("top1_qsm").fill_null(False)).alias("final_top1"),
        (pl.col("top10").fill_null(False) | pl.col("top10_qsm").fill_null(False)).alias("final_top10")
    ])
    
    # Filter out rows with no year and aggregate
    final_agg = (
        combined.filter(
            pl.col("final_year").is_not_null(),
            pl.col("code").is_not_null(),
            pl.col("code") != ""
        )
        .group_by(["code", "final_year"])
        .agg([
            pl.col("final_top1").sum().alias("top1_count"),
            pl.col("final_top10").sum().alias("top10_count"),
            pl.len().alias("total_count")
        ])
        .sort(["code", "final_year"])
        .rename({"final_year": "year"})
    )
    
    # Map country names at the very end using pycountry for all codes
    print("Mapping country names...")
    final_agg = final_agg.with_columns(
        pl.col("code").map_elements(get_country_name, return_dtype=pl.String).alias("country")
    ).select(["country", "code", "year", "top1_count", "top10_count", "total_count"])
    
    print(f"Saving results to {OUTPUT_PATH}...")
    final_agg.write_csv(OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
