"""
Retriever for Lovable docs Pinecone index.

Usage:
    from retriever import Retriever

    r = Retriever(api_key="twój-klucz")
    context = r.search("how to connect Supabase")
    # przekaż context do LLM jako kontekst
"""

import logging

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INDEX_NAME   = "lovable-docs"
DEFAULT_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

# Prefix wymagany przez bge-small przy wyszukiwaniu (NIE przy indeksowaniu)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Wyszukiwarka semantyczna po Pinecone index z opcjonalnym rerankerem.

    Parameters
    ----------
    api_key         : Pinecone API key (pobierz z https://app.pinecone.io)
    index_name      : nazwa indeksu Pinecone (musi zgadzać się z index_docs.py)
    embed_model     : model embeddingów (musi być ten sam co przy indeksowaniu!)
    rerank_model    : CrossEncoder do rerankingu — None wyłącza reranking
    candidates_k    : ile kandydatów pobrać z Pinecone przed rerankiem
    final_k         : ile wyników zwrócić po reranku
    score_threshold : minimalny rerank score (0.0–1.0); None = brak filtrowania
    """

    def __init__(
        self,
        api_key:         str,
        index_name:      str        = DEFAULT_INDEX_NAME,
        embed_model:     str        = DEFAULT_EMBED_MODEL,
        rerank_model:    str | None = DEFAULT_RERANK_MODEL,
        candidates_k:    int        = 10,
        final_k:         int        = 5,
        score_threshold: float | None = 0.3,
    ):
        self.candidates_k    = candidates_k
        self.final_k         = final_k
        self.score_threshold = score_threshold

        # Sprawdź czy indeks istnieje
        pc = Pinecone(api_key=api_key, score_threshold=-10.0)
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            raise ValueError(
                f"Pinecone index '{index_name}' not found. Available: {existing}"
            )

        logger.info("Loading embedding model: %s", embed_model)
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Connecting to Pinecone index: %s", index_name)
        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=api_key,
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
            ---
            <treść chunka>

            [2] ...
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
                "url":     str,
                "content": str,
            }
        """
        chunks = self._retrieve(query)
        return [
            {
                "url":     c.metadata.get("source", ""),
                "content": c.page_content,
            }
            for c in chunks
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve(self, query: str):
        """Pobierz kandydatów z Pinecone, opcjonalnie zreankuj."""
        prefixed_query = BGE_QUERY_PREFIX + query
        candidates = self.vectorstore.similarity_search(prefixed_query, k=self.candidates_k)

        if not candidates:
            return []

        if self.reranker is None:
            return candidates[:self.final_k]

        return self._rerank(query, candidates)

    def _rerank(self, query: str, candidates):
        """Użyj CrossEncoder do rerankingu kandydatów."""
        pairs  = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        if self.score_threshold is not None:
            ranked = [(s, d) for s, d in ranked if s >= self.score_threshold]

        return [doc for _, doc in ranked[:self.final_k]]

    @staticmethod
    def _format_for_llm(chunks) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            url = chunk.metadata.get("source", "")
            parts.append(f"[{i}] URL: {url}\n---\n{chunk.page_content}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Szybki test z CLI: python retriever.py "twoje zapytanie" --api-key "klucz"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", default=["how to connect Supabase to Lovable"])
    parser.add_argument("--api-key", required=True, help="Pinecone API key")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    args = parser.parse_args()

    r = Retriever(api_key=args.api_key, index_name=args.index_name)
    query = " ".join(args.query)
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60 + "\n")
    print(r.search(query))