"""Join manifest, dense vectors, and sparse vectors into Pinecone documents."""

from pathlib import Path
import json
import pandas as pd


PROCESSED_DIR = Path("data/comics/processed")
MANIFEST_PATH = PROCESSED_DIR / "panels_manifest.parquet"
DENSE_PATH = PROCESSED_DIR / "image_dense_vectors.parquet"
SPARSE_PATH = PROCESSED_DIR / "text_sparse_vectors.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pinecone_documents.jsonl"


def to_list(val) -> list:
    """Convert numpy array or other sequence to a plain Python list."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return val.tolist()
    return list(val)


def valid_sparse(sparse) -> bool:
    if not isinstance(sparse, dict):
        return False
    indices = sparse.get("indices")
    values = sparse.get("values")
    return indices is not None and len(indices) > 0 and values is not None and len(values) > 0


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    dense = pd.read_parquet(DENSE_PATH)
    sparse = pd.read_parquet(SPARSE_PATH)

    # Inner join on dense — only panels that were successfully embedded
    df = manifest.merge(dense, on="panel_id", how="inner")
    # Left join on sparse — panels without text keep their document slot
    df = df.merge(sparse, on="panel_id", how="left")

    dropped = len(manifest) - len(df)
    if dropped:
        print(f"Warning: {dropped} panels dropped (missing dense vector)")

    count = 0
    with open(OUTPUT_PATH, "w") as f:
        for row in df.itertuples(index=False):
            doc: dict = {
                "_id": row.panel_id,
                "image_dense": to_list(row.image_dense),
                "ocr_text": row.ocr_text_clean or "",
                "search_text": row.search_text or "",
                "comic_id": row.comic_id,
                "book_id": row.book_id,
                "page_id": row.page_id,
                "page_num": int(row.page_num) if row.page_num is not None and not pd.isna(row.page_num) else None,
                "panel_num": int(row.panel_num) if row.panel_num is not None and not pd.isna(row.panel_num) else None,
                "image_path": row.image_path,
                "source": "COMICS",
                "is_ad_page": bool(row.is_ad_page),
            }

            sparse_vec = getattr(row, "text_sparse", None)
            if valid_sparse(sparse_vec):
                doc["text_sparse"] = {
                    "indices": to_list(sparse_vec["indices"]),
                    "values": to_list(sparse_vec["values"]),
                }

            f.write(json.dumps(doc) + "\n")
            count += 1

    print(f"Wrote {count} documents → {OUTPUT_PATH}")
    with_sparse = sum(1 for row in df.itertuples(index=False) if valid_sparse(getattr(row, "text_sparse", None)))
    print(f"  With sparse vector: {with_sparse}")
    print(f"  Dense only:         {count - with_sparse}")


if __name__ == "__main__":
    main()
