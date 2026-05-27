"""Generate sparse text embeddings for each panel using Pinecone inference."""

from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/text_sparse_vectors.parquet")

SPARSE_MODEL = os.environ.get("SPARSE_EMBED_MODEL", "pinecone-sparse-english-v0")
BATCH_SIZE = 96  # Stay within Pinecone inference batch limits

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def embed_sparse_batch(texts: list[str], input_type: str = "passage") -> list[dict | None]:
    """Embed a batch of texts using pinecone-sparse-english-v0."""
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    results: list[dict | None] = [None] * len(texts)

    if not non_empty:
        return results

    indices, batch_texts = zip(*non_empty)
    response = _pc.inference.embed(
        model=SPARSE_MODEL,
        inputs=list(batch_texts),
        parameters={"input_type": input_type},
    )

    for idx, embedding in zip(indices, response):
        if embedding.sparse_indices and embedding.sparse_values:
            results[idx] = {
                "indices": embedding.sparse_indices,
                "values": embedding.sparse_values,
            }

    return results


def embed_sparse_query(text: str) -> dict | None:
    """Embed a single query string for search time."""
    if not text.strip():
        return None
    results = embed_sparse_batch([text], input_type="query")
    return results[0]


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    print(f"Sparse-embedding {len(manifest)} panels with {SPARSE_MODEL}")

    texts = manifest["search_text"].fillna("").tolist()
    panel_ids = manifest["panel_id"].tolist()

    all_sparse: list[dict | None] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start : start + BATCH_SIZE]
        all_sparse.extend(embed_sparse_batch(batch))

    rows = [{"panel_id": pid, "text_sparse": sparse} for pid, sparse in zip(panel_ids, all_sparse)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)

    filled = sum(1 for r in rows if r["text_sparse"] is not None)
    print(f"Saved {len(rows)} rows ({filled} with sparse vectors) → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
