
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCUMENTS_DIR,
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_text(file_path: Path) -> str:
    """Read a supported knowledge-base file."""
    return file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def create_client() -> QdrantClient:
    """Initialize persistent local Qdrant storage."""
    client = QdrantClient(path=str(QDRANT_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        dimension = embedding_model.get_sentence_embedding_dimension()

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    return client


def ingest_documents() -> None:
    """Index knowledge-base files into Qdrant."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No knowledge-base files found in {DOCUMENTS_DIR}")
        print("Add .txt or .md files if you want a custom knowledge base.")
        return

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = create_client()

    total_chunks = 0

    for file_path in files:
        try:
            text = load_text(file_path)
            chunks = chunk_text(text)

            if not chunks:
                print(f"Skipping empty file: {file_path.name}")
                continue

            embeddings = embedding_model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = []

            relative_source = str(
                file_path.relative_to(DOCUMENTS_DIR)
            )

            for index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings)
            ):
                point_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{relative_source}:{index}:{chunk}",
                    )
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            "source": relative_source,
                            "chunk_index": index,
                        },
                    )
                )

            client.upload_points(
                collection_name=COLLECTION_NAME,
                points=points,
                batch_size=64,
            )

            total_chunks += len(points)

            print(
                f"Indexed {file_path.name}: "
                f"{len(points)} chunks"
            )

        except Exception as error:
            print(f"Failed to index {file_path.name}: {error}")

    print(f"\nIngestion completed: {total_chunks} chunks indexed.")


if __name__ == "__main__":
    ingest_documents()
