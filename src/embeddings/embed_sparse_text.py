"""Generate sparse text embeddings for each panel using Pinecone inference.

Processes panels in chunks of CHUNK_SIZE, writing one parquet file per chunk.
Resumable: skips chunks whose output file already exists.
"""

from pathlib import Path
import os
import time
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/text_sparse_vectors.parquet")
CHUNKS_DIR = Path("data/comics/processed/sparse_chunks")
CHUNK_SIZE = 10_000
BATCH_SIZE = 96

SPARSE_MODEL = os.environ.get("SPARSE_EMBED_MODEL", "pinecone-sparse-english-v0")
_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def embed_sparse_batch(texts: list[str], input_type: str = "passage", retries: int = 5) -> list[dict | None]:
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    results: list[dict | None] = [None] * len(texts)
    if not non_empty:
        return results
    indices, batch_texts = zip(*non_empty)
    for attempt in range(retries):
        try:
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
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt + 1}: {exc}; waiting {wait}s")
            time.sleep(wait)
    return results


def embed_sparse_query(text: str) -> dict | None:
    if not text.strip():
        return None
    return embed_sparse_batch([text], input_type="query")[0]


def merge_chunks() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.parquet"))
    if not chunk_files:
        print("No chunk files to merge")
        return
    print(f"Merging {len(chunk_files)} chunk files...")
    df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Merged {len(df):,} rows → {OUTPUT_PATH}")


def done_chunks() -> set[int]:
    if not CHUNKS_DIR.exists():
        return set()
    return {int(f.stem.split("_")[1]) for f in CHUNKS_DIR.glob("chunk_*.parquet")}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_chunks()
        return

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(MANIFEST_PATH)
    already_done = done_chunks()
    total_chunks = (len(manifest) + CHUNK_SIZE - 1) // CHUNK_SIZE
    remaining = [i for i in range(total_chunks) if i not in already_done]

    print(f"Total panels: {len(manifest):,} | chunks done: {len(already_done)}/{total_chunks} | remaining: {len(remaining)}")
    print(f"Model: {SPARSE_MODEL}")

    for chunk_idx in remaining:
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(manifest))
        chunk_df = manifest.iloc[start:end]

        texts = chunk_df["search_text"].fillna("").tolist()
        panel_ids = chunk_df["panel_id"].tolist()
        rows = []

        pbar = tqdm(range(0, len(texts), BATCH_SIZE), desc=f"chunk {chunk_idx+1}/{total_chunks}", leave=False)
        for start_b in pbar:
            batch = texts[start_b: start_b + BATCH_SIZE]
            batch_ids = panel_ids[start_b: start_b + BATCH_SIZE]
            sparse_results = embed_sparse_batch(batch)
            for pid, sparse in zip(batch_ids, sparse_results):
                rows.append({"panel_id": pid, "text_sparse": sparse})

        chunk_path = CHUNKS_DIR / f"chunk_{chunk_idx:06d}.parquet"
        pd.DataFrame(rows).to_parquet(chunk_path, index=False)

        done_count = len(already_done) + (remaining.index(chunk_idx) + 1)
        filled = sum(1 for r in rows if r["text_sparse"] is not None)
        print(f"  chunk {chunk_idx:06d}: {len(rows)} rows ({filled} with vectors) [{done_count}/{total_chunks} done]")

    print("All chunks complete. Merging...")
    merge_chunks()


if __name__ == "__main__":
    main()
