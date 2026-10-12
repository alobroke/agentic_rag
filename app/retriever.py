
import logging

import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from app.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    WEB_SEARCH_API_KEY,
)

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._closed = False

    def retrieve_local(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks from the local Qdrant collection."""
        if not query.strip() or top_k <= 0:
            return []

        if not self.client.collection_exists(COLLECTION_NAME):
            return []

        try:
            query_vector = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
            ).tolist()

            result = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k * 3,
                with_payload=True,
            )

            evidence = []
            seen_text = set()

            for point in result.points:
                payload = point.payload or {}
                text = payload.get("text", "").strip()
                source = payload.get("source", "knowledge_base")

                if not text:
                    continue

                normalized = " ".join(text.lower().split())
                if normalized in seen_text:
                    continue
                seen_text.add(normalized)

                evidence.append({
                    "text": text,
                    "source": source,
                    "title": source,
                    "url": None,
                    "score": float(point.score),
                    "source_type": "local",
                })

                if len(evidence) >= top_k:
                    break

            return evidence

        except Exception:
            logger.exception("Local retrieval failed")
            return []

    def retrieve_web(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.25,
    ) -> list[dict]:
        """Retrieve and filter web evidence using Tavily relevance scores."""
        if not query.strip() or max_results <= 0:
            return []

        if not WEB_SEARCH_API_KEY:
            logger.warning("TAVILY_API_KEY is not configured")
            return []

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": WEB_SEARCH_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            evidence = []
            seen_urls = set()

            for item in data.get("results", []):
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                title = item.get("title", "").strip()
                score = float(item.get("score", 0.0))

                if not content or not url or url in seen_urls:
                    continue

                # Ignore low-scoring results rather than treating them
                # as evidence. Tune this threshold after testing.
                if score < min_score:
                    logger.debug(
                        "Skipping low-relevance result (%.3f): %s",
                        score,
                        url,
                    )
                    continue

                seen_urls.add(url)
                evidence.append({
                    "text": content,
                    "source": url,
                    "title": title or url,
                    "url": url,
                    "score": score,
                    "source_type": "web",
                })

            return evidence

        except (requests.RequestException, ValueError):
            logger.exception("Web retrieval failed")
            return []

    def retrieve(
        self,
        query: str,
        source: str = "local",
        top_k: int = 5,
    ) -> list[dict]:
        """Route a query to local retrieval, web search, or both."""
        if source == "none":
            return []

        if source == "local":
            return self.retrieve_local(query, top_k)

        if source == "web":
            return self.retrieve_web(query, top_k)

        if source == "both":
            local = self.retrieve_local(query, top_k)
            web = self.retrieve_web(query, top_k)
            return self._deduplicate(local + web, top_k * 2)

        raise ValueError(f"Unsupported retrieval source: {source}")

    @staticmethod
    def _deduplicate(evidence: list[dict], limit: int) -> list[dict]:
        """Remove duplicate source/text pairs."""
        unique = []
        seen = set()

        for item in evidence:
            text = " ".join(item.get("text", "").lower().split())
            source = item.get("source", "")
            key = (source, text)

            if not text or key in seen:
                continue

            seen.add(key)
            unique.append(item)

            if len(unique) >= limit:
                break

        return unique

    def close(self) -> None:
        """Release the local Qdrant database handle."""
        if not self._closed:
            self._closed = True
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with Retriever() as retriever:
        question = input("Enter a test query: ").strip()
        results = retriever.retrieve(question, source="web", top_k=5)

        if not results:
            print("No sufficiently relevant web evidence found.")

        for index, item in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Title: {item['title']}")
            print(f"URL: {item['url']}")
            print(f"Score: {item['score']:.3f}")
            print(item["text"])
