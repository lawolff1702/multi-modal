"""Upsert FSD50K sound vectors into the dense Pinecone index.

Classic dense API: index.upsert(vectors=[{id, values, metadata}], namespace=...).
Idempotent (deterministic ids), batched, with exponential-backoff retry.

Run:
    python -m src.sounds.pinecone.upsert_sounds
"""
from __future__ import annotations

import json
import os
import time

from pinecone import Pinecone

from src.sounds import config

BATCH_SIZE = 100


def batched_jsonl(path, batch_size: int = BATCH_SIZE):
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
            return index.upsert(vectors=batch, namespace=config.PINECONE_SOUND_NAMESPACE)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Upsert attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(config.PINECONE_SOUND_INDEX_NAME)

    total = 0
    failed_batches = 0
    for batch_num, batch in enumerate(batched_jsonl(config.SOUND_VECTORS_JSONL), start=1):
        try:
            upsert_with_retry(index, batch)
            total += len(batch)
            if batch_num % 5 == 0:
                print(f"  Upserted {total} vectors...")
        except Exception as exc:
            failed_batches += 1
            print(f"  Batch {batch_num} failed permanently: {exc}")

    print(f"\nUpsert complete: {total} vectors, {failed_batches} failed batches "
          f"(namespace={config.PINECONE_SOUND_NAMESPACE})")
    print("\n=== index stats ===")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
