"""
Sprawdza co dokładnie jest zaindeksowane w Chroma.
Uruchom: python inspect_index.py
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from collections import Counter

CHROMA_DIR  = r"C:\data\discord_automate_sales\v4\chroma_db"
COLLECTION  = "lovable_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# Pobierz wszystkie dokumenty
all_docs = db._collection.get(include=["metadatas"])
metadatas = all_docs["metadatas"]

print(f"\nŁączna liczba chunków: {len(metadatas)}")

# Unikalne URL-e (strony)
urls = [m.get("source", "BRAK") for m in metadatas]
unique_urls = sorted(set(urls))
print(f"Unikalne strony (source URL): {len(unique_urls)}")

print("\n--- Lista wszystkich zaindeksowanych stron ---")
for url in unique_urls:
    count = urls.count(url)
    print(f"  [{count:3d} chunków]  {url}")

# Unikalne h1
h1s = [m.get("h1", "") for m in metadatas if m.get("h1")]
h1_counter = Counter(h1s)
print(f"\n--- Top 20 sekcji h1 (najczęstsze) ---")
for h1, cnt in h1_counter.most_common(20):
    print(f"  [{cnt:3d}]  {h1}")