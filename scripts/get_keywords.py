import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def extract_keywords(input_path: str, top_n: int = 50):
    # 1. Load data using polars
    print(f"Loading {input_path}...")
    df = pl.read_excel(input_path)
    
    # 2. Prepare documents (combine Title and Abstract)
    # Note: Based on inspection, columns are 'Title' and 'Abstract'
    docs = []
    for row in df.select(["Title", "Abstract"]).iter_rows(named=True):
        title = (row.get("Title") or "").strip()
        abstract = (row.get("Abstract") or "").strip()
        combined = f"{title}. {abstract}".strip(". ").strip()
        if len(combined) > 20:
            docs.append(combined)
            
    if not docs:
        print("No usable documents found.")
        return

    print(f"Processing {len(docs)} documents...")

    # 3. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3), # Bi-grams and Tri-grams are usually more meaningful
        stop_words="english",
        min_df=2,
        max_df=0.85,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"
    )
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # 4. Get mean scores across all documents
    mean_scores = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    
    # 5. Rank and display
    ranked_keywords = sorted(zip(terms, mean_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Keywords:")
    print("-" * 40)
    for i, (term, score) in enumerate(ranked_keywords[:top_n], 1):
        print(f"{i:2d}. {term:<30} (score: {score:.4f})")

if __name__ == "__main__":
    file_path = "data/Green_Hydrogen_v1_03Jun2026104212.xlsx"
    if Path(file_path).exists():
        extract_keywords(file_path)
    else:
        print(f"Error: File {file_path} not found.")
