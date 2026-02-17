"""
Retriever for Lovable docs Chroma index.

Usage:
    from retriever import Retriever

    r = Retriever()
    context = r.search("how to connect Supabase")
    # przekaż context do LLM jako kontekst
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — nadpisz przez konstruktor lub zmienne środowiskowe
# ---------------------------------------------------------------------------

DEFAULT_CHROMA_DIR   = r"C:\data\discord_automate_sales\v4\chroma_db"
DEFAULT_COLLECTION   = "lovable_docs"
DEFAULT_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

# Prefix wymagany przez bge-small przy wyszukiwaniu (NIE przy indeksowaniu)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Wyszukiwarka semantyczna po Chroma index z opcjonalnym rerankerem.

    Parameters
    ----------
    chroma_dir    : ścieżka do folderu z zapisanym Chroma index
    collection    : nazwa kolekcji Chroma (musi zgadzać się z index_docs.py)
    embed_model   : model embeddingów (musi być ten sam co przy indeksowaniu!)
    rerank_model  : CrossEncoder do rerankingu — None wyłącza reranking
    candidates_k  : ile kandydatów pobrać z Chroma przed rerankiem
    final_k       : ile wyników zwrócić po reranku
    score_threshold : minimalny rerank score (0.0–1.0); None = brak filtrowania
    """

    def __init__(
        self,
        chroma_dir:      str | Path = DEFAULT_CHROMA_DIR,
        collection:      str        = DEFAULT_COLLECTION,
        embed_model:     str        = DEFAULT_EMBED_MODEL,
        rerank_model:    str | None = DEFAULT_RERANK_MODEL,
        candidates_k:    int        = 10,
        final_k:         int        = 5,
        score_threshold: float | None = 0.3,
    ):
        self.candidates_k    = candidates_k
        self.final_k         = final_k
        self.score_threshold = score_threshold

        logger.info("Loading embedding model: %s", embed_model)
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Connecting to Chroma: %s / %s", chroma_dir, collection)
        self.vectorstore = Chroma(
            collection_name=collection,
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
        )

        self.reranker = None
        if rerank_model:
            logger.info("Loading reranker: %s", rerank_model)
            self.reranker = CrossEncoder(rerank_model)

        logger.info("Retriever ready.")

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """
        Wyszukaj najbardziej trafne fragmenty dokumentacji dla podanego zapytania.

        Zwraca gotowy string do wklejenia jako kontekst dla LLM.
        Format:
            [1] URL: https://...
            section: h1 > h2
            ---
            <treść chunka>

            [2] ...

        Parameters
        ----------
        query : zapytanie w dowolnym języku
        """
        chunks = self._retrieve(query)
        if not chunks:
            return "Nie znaleziono pasujących fragmentów dokumentacji."
        return self._format_for_llm(chunks)

    def search_raw(self, query: str) -> list[dict]:
        """
        Jak search(), ale zwraca listę słowników zamiast stringa.
        Przydatne gdy chcesz samodzielnie formatować kontekst.

        Każdy element:
            {
                "url":     str,   # pełny URL źródła — gotowy do użycia w poście
                "section": str,   # np. "Getting Started > Connect Supabase"
                "h1":      str,
                "h2":      str,
                "h3":      str,
                "content": str,
            }
        """
        chunks = self._retrieve(query)
        return [
            {
                "url":     c.metadata.get("source", ""),
                "section": self._build_section(c.metadata),
                "h1":      c.metadata.get("h1", ""),
                "h2":      c.metadata.get("h2", ""),
                "h3":      c.metadata.get("h3", ""),
                "content": c.page_content,
            }
            for c in chunks
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve(self, query: str):
        """Pobierz kandydatów z Chroma, opcjonalnie zreankuj."""

        # bge-small wymaga prefixu przy zapytaniach
        prefixed_query = BGE_QUERY_PREFIX + query

        candidates = self.vectorstore.similarity_search(
            prefixed_query,
            k=self.candidates_k,
        )

        if not candidates:
            return []

        if self.reranker is None:
            return candidates[:self.final_k]

        return self._rerank(query, candidates)

    def _rerank(self, query: str, candidates):
        """Użyj CrossEncoder do rerankingu kandydatów."""

        # CrossEncoder nie używa prefixu — dostaje surowe zapytanie
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        # Opcjonalne filtrowanie po minimalnym score
        if self.score_threshold is not None:
            ranked = [(s, d) for s, d in ranked if s >= self.score_threshold]

        return [doc for _, doc in ranked[:self.final_k]]

    @staticmethod
    def _build_section(metadata: dict) -> str:
        """Zbuduj czytelną ścieżkę sekcji z nagłówków h1/h2/h3."""
        parts = [metadata.get(h, "") for h in ("h1", "h2", "h3")]
        parts = [p for p in parts if p]
        return " > ".join(parts) if parts else "—"

    @staticmethod
    def _format_for_llm(chunks) -> str:
        """Formatuj chunki jako czytelny kontekst dla LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            url     = chunk.metadata.get("source", "")
            section = Retriever._build_section(chunk.metadata)

            parts.append(
                f"[{i}] URL: {url}\n"
                f"    section: {section}\n"
                f"---\n"
                f"{chunk.page_content}"
            )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Szybki test z CLI: python retriever.py "twoje zapytanie"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "how to connect Supabase to Lovable"

    r = Retriever()
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60 + "\n")
    print(r.search(query))