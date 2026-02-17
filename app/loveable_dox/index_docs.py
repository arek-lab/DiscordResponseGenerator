"""
Chunking → Embedding → Chroma indexing pipeline for Lovable docs.

Usage:
    python index_docs.py C:/data/.../cleaned
    python index_docs.py C:/data/.../cleaned --chroma-dir ./chroma_db
    python index_docs.py C:/data/.../cleaned --chunk-size 400 --chunk-overlap 40
"""

import pickle
import logging
import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load cleaned docs
# ---------------------------------------------------------------------------

def load_cleaned_docs(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    docs = []
    for filepath in sorted(folder.glob("*.pkl")):
        with open(filepath, "rb") as f:
            batch = pickle.load(f)
        docs.extend(batch)
        logger.info("Loaded %d docs from %s", len(batch), filepath.name)
    logger.info("Total docs loaded: %d", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Chunking
#
# Strategy:
#   1. Split by Markdown headers first (preserves section context)
#   2. Then split oversized sections by characters with overlap
#
# Each chunk inherits:
#   - source URL from original doc metadata
#   - h1/h2/h3 header context so retrieval knows which section it came from
# ---------------------------------------------------------------------------

def chunk_docs(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split documents into chunks suitable for bge-small-en-v1.5 (512 token window).

    chunk_size=512 chars ≈ 100-130 tokens for English technical text.
    If you switch to nomic-embed-text-v1.5, use chunk_size=1024.
    """

    # Stage A: split by Markdown headers to preserve section context
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,  # keep headers in chunk text — critical for context
    )

    # Stage B: split oversized sections by character count
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "")

        # Stage A
        try:
            header_chunks = header_splitter.split_text(doc.page_content)
        except Exception:
            header_chunks = [Document(page_content=doc.page_content, metadata={})]

        # Stage B — further split if chunk is too large
        for hchunk in header_chunks:
            text = hchunk.page_content.strip()
            if not text:
                continue

            if len(text) <= chunk_size:
                # Small enough — keep as-is with enriched metadata
                all_chunks.append(Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "h1": hchunk.metadata.get("h1", ""),
                        "h2": hchunk.metadata.get("h2", ""),
                        "h3": hchunk.metadata.get("h3", ""),
                    }
                ))
            else:
                # Too large — split by characters
                sub_chunks = char_splitter.split_text(text)
                for sub in sub_chunks:
                    if sub.strip():
                        all_chunks.append(Document(
                            page_content=sub.strip(),
                            metadata={
                                "source": source,
                                "h1": hchunk.metadata.get("h1", ""),
                                "h2": hchunk.metadata.get("h2", ""),
                                "h3": hchunk.metadata.get("h3", ""),
                            }
                        ))

    logger.info(
        "Chunking complete: %d docs → %d chunks (avg %.0f chars/chunk)",
        len(docs),
        len(all_chunks),
        sum(len(c.page_content) for c in all_chunks) / max(len(all_chunks), 1),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Step 3 — Embed + save to Chroma
#
# Model: BAAI/bge-small-en-v1.5
#   - 384 dimensions, ~33M params
#   - Fast on CPU, good quality for technical EN docs
#   - IMPORTANT: query prefix required at retrieval time (not at indexing time)
#     Use: "Represent this sentence for searching relevant passages: <query>"
# ---------------------------------------------------------------------------

def build_chroma_index(
    chunks: list[Document],
    chroma_dir: str | Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> Chroma:
    """
    Embed chunks and save to a persistent Chroma vector store.

    Parameters
    ----------
    chunks      : list of Document chunks from chunk_docs()
    chroma_dir  : directory where Chroma will persist the index
    model_name  : HuggingFace embedding model
    batch_size  : number of chunks to embed at once
                  256 works well for bge-small on CPU
                  reduce to 64-128 for nomic-embed-text-v1.5
    """
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={
            "normalize_embeddings": True,  # required for cosine similarity
            "batch_size": batch_size,
        },
    )

    logger.info(
        "Embedding %d chunks into Chroma at: %s",
        len(chunks), chroma_dir
    )

    # Chroma handles batching internally when given a list of Documents
    # Collection name identifies this index — useful if you have multiple
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="lovable_docs",
        persist_directory=str(chroma_dir),
    )

    logger.info("Index saved. Collection: lovable_docs | Vectors: %d", len(chunks))
    return vectorstore


# ---------------------------------------------------------------------------
# Step 4 — Quick sanity check
# ---------------------------------------------------------------------------

def sanity_check(vectorstore: Chroma, queries: list[str] | None = None) -> None:
    """Run a few test queries to verify the index works."""
    if queries is None:
        queries = [
            "how to connect Supabase to Lovable",
            "what is plan mode",
            "how to deploy to custom domain",
        ]

    logger.info("--- Sanity check ---")
    for query in queries:
        # bge-small requires this prefix for queries (NOT for indexed documents)
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        results = vectorstore.similarity_search(prefixed, k=2)
        logger.info("Query: %r", query)
        for i, doc in enumerate(results):
            logger.info(
                "  [%d] %s | h1=%r | %d chars | %.80s...",
                i + 1,
                doc.metadata.get("source", "?"),
                doc.metadata.get("h1", ""),
                len(doc.page_content),
                doc.page_content.replace("\n", " "),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    cleaned_folder: str | Path,
    chroma_dir: str | Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
    skip_sanity: bool = False,
) -> Chroma:

    cleaned_folder = Path(cleaned_folder)

    if chroma_dir is None:
        chroma_dir = cleaned_folder.parent / "chroma_db"

    # 1. Load
    docs = load_cleaned_docs(cleaned_folder)

    # 2. Chunk
    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 3. Embed + index
    vectorstore = build_chroma_index(
        chunks,
        chroma_dir=chroma_dir,
        model_name=model_name,
        batch_size=batch_size,
    )

    # 4. Verify
    if not skip_sanity:
        sanity_check(vectorstore)

    return vectorstore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk, embed and index cleaned Lovable docs into Chroma."
    )
    parser.add_argument(
        "cleaned_folder",
        help="Folder with cleaned .pkl files (output of clean_docs.py)"
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Chroma persist directory (default: ../chroma_db)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Max chars per chunk (default: 512). Use 1024 for nomic-embed-text-v1.5"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks in chars (default: 64)"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256 for bge-small, use 64 for nomic)"
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity check queries after indexing"
    )
    args = parser.parse_args()

    run(
        cleaned_folder=args.cleaned_folder,
        chroma_dir=args.chroma_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        batch_size=args.batch_size,
        skip_sanity=args.skip_sanity,
    )
