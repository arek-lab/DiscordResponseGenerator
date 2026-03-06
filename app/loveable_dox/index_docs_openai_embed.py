"""
Load cleaned .pkl files → chunk → embed (OpenAI) → cache → upsert to Pinecone.

Embeddingi generowane są dokładnie raz:
  - zapisywane do ./openai_embeddings/<index>_<date>.pkl (cache)
  - wgrywane do Pinecone przez natywne SDK (index.upsert)

Usage:
    python index_docs.py
    python index_docs.py --cleaned-dir ./cleaned --index-name lovable-docs
    python index_docs.py --dry-run --sample 5

Required env vars:
    PINECONE_API_KEY
    OPENAI_API_KEY

Index must exist in Pinecone console (https://app.pinecone.io):
    Dimensions : 1536  (text-embedding-3-small)
    Metric     : cosine
"""

import os
import re
import pickle
import hashlib
import logging
import argparse
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME    = "text-embedding-3-small"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 160
BATCH_SIZE    = 100
CACHE_DIR     = "./openai_embeddings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _url_to_title(url: str) -> str:
    path = url.rstrip("/").split("://")[-1]
    parts = [p for p in path.split("/") if p and p not in ("docs", "www", "lovable.dev")]
    return " / ".join(p.replace("-", " ").title() for p in parts) if parts else url


def _extract_sections(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("", text)]

    sections = []
    prev_end = 0
    prev_heading = ""

    for m in matches:
        body = text[prev_end:m.start()].strip()
        if body or prev_heading:
            sections.append((prev_heading, body))
        prev_heading = m.group(2).strip()
        prev_end = m.end()

    body = text[prev_end:].strip()
    sections.append((prev_heading, body))

    return [(h, b) for h, b in sections if h or b]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_path(index_name: str) -> Path:
    today = date.today().isoformat()
    return Path(CACHE_DIR) / f"{index_name}_{today}.pkl"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_docs(cleaned_dir: str) -> list[Document]:
    docs = []
    for path in sorted(Path(cleaned_dir).glob("*.pkl")):
        with open(path, "rb") as f:
            batch = pickle.load(f)
        docs.extend(batch)
    logger.info("Docs loaded: %d (from %s)", len(docs), cleaned_dir)
    return docs


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def chunk_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    seen_hashes: set[str] = set()
    duplicates = 0

    for doc in docs:
        source = doc.metadata.get("source", "")
        title  = _url_to_title(source)

        for section_heading, section_body in _extract_sections(doc.page_content):
            if not section_body.strip():
                continue

            for text in splitter.split_text(section_body):
                text = text.strip()
                if not text:
                    continue

                h = _sha1(text)
                if h in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(h)

                context_parts = [f"[doc: {title}]"]
                if section_heading:
                    context_parts.append(f"[section: {section_heading}]")
                page_content = "\n".join(context_parts) + "\n" + text

                chunks.append(Document(
                    page_content=page_content,
                    metadata={
                        "source":  source,
                        "title":   title,
                        "section": section_heading,
                        "hash":    h,
                    },
                ))

    logger.info(
        "Chunking complete: %d docs → %d chunks (%d duplicates skipped)",
        len(docs), len(chunks), duplicates,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embed & cache
# ---------------------------------------------------------------------------

def embed_and_cache(chunks: list[Document], index_name: str) -> list[dict]:
    """
    Generuje embeddingi dokładnie raz, zapisuje cache do .pkl.
    Zwraca listę rekordów: page_content + metadata + embedding.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY env var not set.")

    embeddings_model = OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key=api_key,
    )

    import tiktoken
    enc        = tiktoken.encoding_for_model("text-embedding-3-small")
    MAX_TOKENS = 8192

    def _truncate(text: str) -> str:
        tokens = enc.encode(text)
        if len(tokens) > MAX_TOKENS:
            return enc.decode(tokens[:MAX_TOKENS])
        return text

    logger.info("Generating embeddings for %d chunks via OpenAI API...", len(chunks))
    texts     = [_truncate(c.page_content) for c in chunks]
    truncated = sum(1 for c in chunks if len(enc.encode(c.page_content)) > MAX_TOKENS)
    if truncated:
        logger.warning("Truncated %d chunks exceeding %d tokens", truncated, MAX_TOKENS)
    vectors = embeddings_model.embed_documents(texts)

    records = [
        {
            "page_content": chunk.page_content,
            "metadata":     chunk.metadata,
            "embedding":    vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(index_name)
    with open(cache_file, "wb") as f:
        pickle.dump(records, f)
    logger.info("Embeddings cached: %s (%d records)", cache_file, len(records))

    return records


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_to_pinecone(records: list[dict], index_name: str) -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY env var not set.")

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        raise ValueError(
            f"Index '{index_name}' not found. Available: {existing}\n"
            "Create it at https://app.pinecone.io (dim=1536, metric=cosine)."
        )

    index = pc.Index(index_name)

    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    total = 0
    for i, batch in enumerate(batches):
        try:
            vectors = [
                {
                    "id":       r["metadata"]["hash"],
                    "values":   r["embedding"],
                    "metadata": {**r["metadata"], "text": r["page_content"]},
                }
                for r in batch
            ]
            index.upsert(vectors=vectors)
            total += len(batch)
        except Exception as e:
            logger.error("Batch %d upsert failed: %s", i, e)

    logger.info("Upsert complete: %d chunks → Pinecone index '%s'", total, index_name)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(chunks: list[Document], sample: int) -> None:
    print(f"\n{'='*70}")
    print(f"DRY RUN — total chunks: {len(chunks)}")
    print(f"{'='*70}\n")

    for i, chunk in enumerate(chunks[:sample], 1):
        print(f"--- Chunk {i}/{sample} ---")
        print(f"Title  : {chunk.metadata.get('title')}")
        print(f"Section: {chunk.metadata.get('section') or '(brak)'}")
        print(f"Source : {chunk.metadata.get('source')}")
        print(f"Hash   : {chunk.metadata.get('hash')}")
        print(f"Length : {len(chunk.page_content)} znaków")
        print("Content:")
        print(chunk.page_content)
        print()

    lengths = [len(c.page_content) for c in chunks]
    avg = sum(lengths) / len(lengths) if lengths else 0
    titles = {c.metadata.get("title") for c in chunks}
    sections_with_heading = sum(1 for c in chunks if c.metadata.get("section"))

    print(f"{'='*70}")
    print(f"Statystyki:")
    print(f"  Chunki ogółem       : {len(chunks)}")
    print(f"  Unikalne strony     : {len(titles)}")
    print(f"  Chunki z sekcją     : {sections_with_heading} ({100*sections_with_heading//len(chunks)}%)")
    print(f"  Średnia długość     : {avg:.0f} znaków")
    print(f"  Min / Max           : {min(lengths)} / {max(lengths)} znaków")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> list[dict]:
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    logger.info("Loaded %d records from cache: %s", len(records), cache_path)
    return records


def run(cleaned_dir: str, index_name: str, is_dry_run: bool, sample: int) -> None:
    docs   = load_docs(cleaned_dir)
    chunks = chunk_docs(docs)

    if is_dry_run:
        dry_run(chunks, sample)
        return

    cache_file = _cache_path(index_name)
    if cache_file.exists():
        answer = input(
            f"\nCache już istnieje: {cache_file}\n"
            "Użyć istniejącego cache? [T/n]: "
        ).strip().lower()
        if answer in ("", "t", "y", "tak", "yes"):
            records = load_cache(cache_file)
            upsert_to_pinecone(records, index_name)
            return

    records = embed_and_cache(chunks, index_name)
    upsert_to_pinecone(records, index_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk, embed (OpenAI) and upsert docs to Pinecone."
    )
    parser.add_argument("--cleaned-dir", default="./cleaned_docs",
                        help="Katalog z plikami .pkl (domyślnie: ./cleaned_docs)")
    parser.add_argument("--index-name",  default="lovable-docs",
                        help="Nazwa indeksu Pinecone (domyślnie: lovable-docs)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Pokaż przykładowe chunki bez embeddingu i upsert")
    parser.add_argument("--sample",      type=int, default=5,
                        help="Liczba chunków w dry-run (domyślnie: 5)")
    args = parser.parse_args()

    run(
        cleaned_dir=args.cleaned_dir,
        index_name=args.index_name,
        is_dry_run=args.dry_run,
        sample=args.sample,
    )