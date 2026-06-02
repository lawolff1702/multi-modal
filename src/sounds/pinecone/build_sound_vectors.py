"""Join manifest + audio vectors into Pinecone upsert records (JSONL).

Output line shape:
    {"id": "fsd50k:123", "values": [...512...], "metadata": {...}}

Metadata is kept Pinecone-safe: only str / number / bool / list-of-str, no
nested dicts, no None values (keys with None are dropped — Pinecone rejects
nulls).

Run:
    python -m src.sounds.pinecone.build_sound_vectors
"""
from __future__ import annotations

import json

import pandas as pd

from src.sounds import config

# Manifest columns that go into Pinecone metadata (audio_dense is the vector).
_META_FIELDS = [
    "source", "source_id", "dataset_name", "split", "audio_path",
    "labels", "label_ids", "title", "description", "tags",
    "license", "license_ok_for_commercial", "requires_attribution",
    "attribution", "duration_sec",
]


def _clean_value(v):
    """Coerce numpy/list types and signal 'drop me' for null/NaN."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "tolist"):                 # numpy array -> list
        v = v.tolist()
    if isinstance(v, list):
        return [str(x) for x in v]           # Pinecone lists must be strings
    return v


def build_metadata(row) -> dict:
    meta = {}
    for field in _META_FIELDS:
        val = _clean_value(getattr(row, field))
        if val is None:
            continue                         # omit nulls; Pinecone rejects them
        meta[field] = val
    return meta


def main() -> None:
    manifest = pd.read_parquet(config.MANIFEST_PARQUET)
    vectors = pd.read_parquet(config.AUDIO_VECTORS_PARQUET)

    merged = manifest.merge(vectors, on="sound_id", how="inner")
    missing = len(manifest) - len(merged)
    if missing:
        print(f"Note: {missing} manifest rows have no vector yet (skipped).")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(config.SOUND_VECTORS_JSONL, "w") as f:
        for row in merged.itertuples(index=False):
            values = row.audio_dense
            values = values.tolist() if hasattr(values, "tolist") else list(values)
            record = {
                "id": row.sound_id,
                "values": [float(x) for x in values],
                "metadata": build_metadata(row),
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} records -> {config.SOUND_VECTORS_JSONL}")
    if written:
        sample = json.loads(open(config.SOUND_VECTORS_JSONL).readline())
        sample["values"] = f"[{len(sample['values'])} floats]"
        print("\n=== sample record ===")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
