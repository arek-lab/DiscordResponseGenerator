"""
Load cleaned .pkl files → chunk → embed → upsert to Pinecone (concurrently).

Usage:
    python index_docs.py
    python index_docs.py --cleaned-dir ./cleaned --index-name lovable-docs

Required env var:
    PINECONE_API_KEY

Index must exist in Pinecone console (https://app.pinecone.io):
    Dimensions : 384   (bge-small-en-v1.5)
    Metric     : cosine
"""

import os
import pickle
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()  # czyta .env z bieżącego katalogu

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME  = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE  = 512   # znaki; ~100-130 tokenów dla angielskiego tekstu technicznego
CHUNK_OVERLAP = 64
BATCH_SIZE  = 256
MAX_WORKERS = 4


def load_docs(cleaned_dir: str) -> list[Document]:
    docs = []
    for path in sorted(Path(cleaned_dir).glob("*.pkl")):
        with open(path, "rb") as f:
            batch = pickle.load(f)
        docs.extend(batch)
        logger.info("Loaded %d docs from %s", len(batch), path.name)
    logger.info("Total docs loaded: %d", len(docs))
    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        page_hint = "/".join(source.rstrip("/").split("/")[-2:]) if source else ""
        for text in splitter.split_text(doc.page_content):
            text = text.strip()
            if not text:
                continue
            content_with_context = f"[page: {page_hint}]\n{text}" if page_hint else text
            chunks.append(Document(
                page_content=content_with_context,
                metadata={"source": source},
            ))
    logger.info("Chunking complete: %d docs → %d chunks", len(docs), len(chunks))
    return chunks


def upsert_batch(vectorstore: PineconeVectorStore, batch: list[Document], batch_id: int) -> int:
    vectorstore.add_documents(batch)
    logger.info("Batch %d — upserted %d chunks", batch_id, len(batch))
    return len(batch)


def run(cleaned_dir: str, index_name: str) -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY env var not set.")

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        raise ValueError(
            f"Index '{index_name}' not found. Available: {existing}\n"
            "Create it at https://app.pinecone.io (dim=384, metric=cosine)."
        )

    docs   = load_docs(cleaned_dir)
    chunks = chunk_docs(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    logger.info("Upserting %d batches concurrently (workers=%d) ...", len(batches), MAX_WORKERS)

    total = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upsert_batch, vectorstore, batch, i): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            total += future.result()

    logger.info("Done. Total upserted: %d chunks", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-dir", default="./cleaned")
    parser.add_argument("--index-name", default="lovable-docs")
    args = parser.parse_args()
    run(args.cleaned_dir, args.index_name)