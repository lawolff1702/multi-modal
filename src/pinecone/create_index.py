"""Create the Pinecone document-schema index for comic panel search."""

import os
import time
from pinecone import Pinecone
from pinecone.preview import SchemaBuilder


INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "comic-panels")
IMAGE_EMBED_DIM = int(os.environ.get("IMAGE_EMBED_DIM", "512"))
IMAGE_EMBED_METRIC = os.environ.get("IMAGE_EMBED_METRIC", "cosine")


def create_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Index '{INDEX_NAME}' already exists — skipping creation")
        return

    print(f"Creating index '{INDEX_NAME}' (dim={IMAGE_EMBED_DIM}, metric={IMAGE_EMBED_METRIC})")

    schema = (
        SchemaBuilder()
        .add_dense_vector_field(
            "image_dense",
            dimension=IMAGE_EMBED_DIM,
            metric=IMAGE_EMBED_METRIC,
        )
        .add_sparse_vector_field("text_sparse")
        .add_string_field("ocr_text", full_text_search={"language": "en"})
        .add_string_field("search_text", full_text_search={"language": "en"})
        .build()
    )

    pc.preview.indexes.create(name=INDEX_NAME, schema=schema)

    # Wait for the index to be ready
    for _ in range(60):
        status = pc.describe_index(INDEX_NAME).status
        if status.get("ready"):
            print(f"Index '{INDEX_NAME}' is ready")
            return
        print("  waiting for index to be ready...")
        time.sleep(5)

    raise TimeoutError(f"Index '{INDEX_NAME}' did not become ready within 5 minutes")


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    create_index(pc)


if __name__ == "__main__":
    main()
