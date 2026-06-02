"""Create the dense Pinecone index for FSD50K sound effects.

This is a STANDARD dense index (classic API: pc.create_index / index.upsert /
index.query), NOT a document-schema index like comic-panels. Dense-only for v1:
no sparse, no FTS, no SchemaBuilder.

Run:
    python -m src.sounds.pinecone.create_sound_index
"""
from __future__ import annotations

import os
import time

from pinecone import Pinecone, ServerlessSpec

from src.sounds import config


def create_index(pc: Pinecone) -> None:
    name = config.PINECONE_SOUND_INDEX_NAME
    existing = [idx.name for idx in pc.list_indexes()]
    if name in existing:
        print(f"Index '{name}' already exists — skipping creation")
        return

    print(f"Creating dense index '{name}' "
          f"(dim={config.CLAP_EMBED_DIM}, metric=cosine, {config.PINECONE_CLOUD}/{config.PINECONE_REGION})")
    pc.create_index(
        name=name,
        dimension=config.CLAP_EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )

    for _ in range(60):
        if pc.describe_index(name).status.get("ready"):
            print(f"Index '{name}' is ready")
            return
        print("  waiting for index to be ready...")
        time.sleep(5)
    raise TimeoutError(f"Index '{name}' did not become ready within 5 minutes")


def main() -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    create_index(pc)


if __name__ == "__main__":
    main()
