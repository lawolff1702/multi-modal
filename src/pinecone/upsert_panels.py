"""Upsert comic panel documents into Pinecone."""

import json
import os
import time
from pathlib import Path

from pinecone import Pinecone


INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "comic-panels")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "comics-v1")
DOCS_PATH = Path("data/comics/processed/pinecone_documents.jsonl")
BATCH_SIZE = 100


def batched_jsonl(path: Path, batch_size: int = BATCH_SIZE):
    batch = []
    with open(path) as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def upsert_with_retry(index, batch: list, retries: int = 5):
    for attempt in range(retries):
        try:
            return index.documents.upsert(namespace=NAMESPACE, documents=batch)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Upsert attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
            time.sleep(wait)


def validate_search(index) -> None:
    """Quick sanity check: run a dense search after upsert."""
    import random
    dummy_vector = [random.gauss(0, 1) for _ in range(int(os.environ.get("IMAGE_EMBED_DIM", "512")))]
    norm = sum(x ** 2 for x in dummy_vector) ** 0.5
    dummy_vector = [x / norm for x in dummy_vector]

    results = index.documents.search(
        namespace=NAMESPACE,
        top_k=3,
        score_by=[{"type": "dense_vector", "field": "image_dense", "values": dummy_vector}],
        include_fields=["panel_id", "comic_id", "page_num"],
    )
    hits = results.matches
    print(f"Validation search returned {len(hits)} hits")
    for h in hits:
        print(f"  {h.id} score={h.score:.4f}")


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.preview.index(name=INDEX_NAME)

    total = 0
    failed_batches = 0

    for batch_num, batch in enumerate(batched_jsonl(DOCS_PATH), start=1):
        try:
            upsert_with_retry(index, batch)
            total += len(batch)
            if batch_num % 10 == 0:
                print(f"  Upserted {total} documents...")
        except Exception as exc:
            failed_batches += 1
            print(f"  Batch {batch_num} failed permanently: {exc}")

    print(f"\nUpsert complete: {total} documents, {failed_batches} failed batches")

    if total > 0:
        validate_search(index)


if __name__ == "__main__":
    main()
