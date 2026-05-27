# COMICS Panel Search: Pinecone Build Spec

Last updated: 2026-05-27

## Goal

Build a Pinecone-backed panel search index over the COMICS dataset.

Each individual comic panel should be one searchable document in Pinecone. The system should support:

1. Dense vector search over panel images.
2. Sparse vector search over panel OCR/search text.
3. Full-text search over OCR/search text.
4. Combined retrieval by running separate searches and merging results client-side.
5. A Jupyter notebook for manually running and inspecting queries.

Do not build the UI, Streamlit app, FastAPI app, story builder, sequencing logic, generation layer, or fine-tuning workflow yet.

## Source notes

Use these sources as grounding references while implementing:

- COMICS GitHub repo: https://github.com/miyyer/comics
- COMICS direct downloads: https://obj.umiacs.umd.edu/comics/index.html
- Pinecone indexing overview: https://docs.pinecone.io/guides/index-data/indexing-overview
- Pinecone data modeling: https://docs.pinecone.io/guides/index-data/data-modeling
- Pinecone search overview: https://docs.pinecone.io/guides/search/search-overview

Important implementation assumptions:

- The COMICS repo provides `setup.sh`, which downloads raw panel images, OCR transcriptions, advertisement page IDs, cached features, and original page images.
- For this project, use the raw dataset files directly instead of depending on the old Python 2/Theano training pipeline.
- Pinecone document-schema indexes can hold a `dense_vector`, a `sparse_vector`, one or more FTS-enabled `string` fields, and metadata on the same JSON document.
- Metadata fields do not need to be declared in the schema. Any undeclared fields upserted with a document are stored and indexed for filtering.
- A document-schema index supports at most one dense vector field and at most one sparse vector field.
- A single Pinecone document search request ranks by one scoring signal. To blend dense, sparse, and BM25/FTS rankings, run separate searches and merge results client-side.

## Core data model

One comic panel equals one Pinecone document.

Do not index full pages as the primary retrieval unit. Store page-level metadata on each panel so the exact panel can be displayed and traced back to its source page later.

Target document shape:

```json
{
  "_id": "comic_id:page_num:panel_num",

  "image_dense": [0.012, -0.034, 0.091],
  "text_sparse": {
    "indices": [123, 456, 789],
    "values": [0.41, 0.29, 0.13]
  },

  "ocr_text": "Wait... I know that symbol.",
  "search_text": "Wait... I know that symbol.",

  "comic_id": "black_terror_001",
  "book_id": "black_terror_001",
  "page_id": "page_012",
  "page_num": 12,
  "panel_num": 4,
  "image_path": "data/comics/raw/raw_panel_images/...",
  "source": "COMICS",
  "is_ad_page": false
}
```

## Target repository structure

```text
comic-search/
  data/
    comics/
      raw/
        raw_panel_images.tar.gz
        COMICS_ocr_file.csv
        predadpages.txt
        raw_panel_images/
      processed/
        panels_manifest.parquet
        panels_manifest.jsonl
        image_dense_vectors.parquet
        text_sparse_vectors.parquet
        pinecone_documents.jsonl
  notebooks/
    query_comics_index.ipynb
  src/
    ingest/
      build_manifest.py
      clean_text.py
    embeddings/
      embed_images.py
      embed_sparse_text.py
    pinecone/
      create_index.py
      build_documents.py
      upsert_panels.py
    search/
      search_dense.py
      search_sparse.py
      search_fts.py
      search_combined.py
      fusion.py
      router.py
  README.md
  requirements.txt
  config.yaml
```

## Environment variables

Required:

```bash
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="comic-panels"
export PINECONE_NAMESPACE="comics-v1"
export COMICS_DATA_DIR="data/comics"
```

Optional:

```bash
export IMAGE_EMBED_MODEL="open_clip_ViT-B-32"
export IMAGE_EMBED_DIM="512"
export IMAGE_EMBED_METRIC="cosine"
export SPARSE_EMBED_MODEL="pinecone-sparse-english-v0"
```

## Config file

Create:

```text
config.yaml
```

Suggested contents:

```yaml
pinecone:
  index_name: comic-panels
  namespace: comics-v1

dense_embedding:
  model_name: open_clip_ViT-B-32
  field_name: image_dense
  dimension: 512
  metric: cosine

sparse_embedding:
  model_name: pinecone-sparse-english-v0
  field_name: text_sparse
  source_text_field: search_text

fields:
  dense_vector: image_dense
  sparse_vector: text_sparse
  fts_text:
    - ocr_text
    - search_text

data:
  comics_data_dir: data/comics
  manifest_path: data/comics/processed/panels_manifest.parquet
  dense_vectors_path: data/comics/processed/image_dense_vectors.parquet
  sparse_vectors_path: data/comics/processed/text_sparse_vectors.parquet
  pinecone_documents_path: data/comics/processed/pinecone_documents.jsonl
```

If the selected dense image embedding model has a different dimension, update `dense_embedding.dimension` before creating the Pinecone index.

## Step 1: Download COMICS files

Create a download script or include this in the README:

```bash
mkdir -p data/comics/raw
cd data/comics/raw

wget -c https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz
wget -c https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv
wget -c https://obj.umiacs.umd.edu/comics/predadpages.txt

tar -xzf raw_panel_images.tar.gz
```

Optional, only if page-level reconstruction is needed later:

```bash
wget -c https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz
```

Do not rely on the old COMICS training pipeline unless necessary. Use the downloaded raw panel images and OCR CSV directly.

## Step 2: Clean OCR text

Create:

```text
src/ingest/clean_text.py
```

Clean text conservatively. Comics contain sound effects, stylized spelling, names, unusual punctuation, and OCR noise.

Requirements:

- Normalize whitespace.
- Remove obvious OCR junk.
- Preserve punctuation where useful.
- Preserve all-caps sound effects such as `BANG`, `WHAM`, `CRASH`.
- Preserve names, unusual spellings, and comic-style phrasing.
- Store both raw and cleaned text.
- Do not aggressively lowercase the stored text fields. Pinecone FTS applies server-side tokenization and lowercasing for FTS-enabled fields.

Example:

```python
import re


def clean_ocr(text: str | None) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Keep this conservative. Do not remove punctuation or all-caps words.
    return text


def build_search_text(ocr_text_clean: str, extra_terms: list[str] | None = None) -> str:
    parts = [ocr_text_clean]
    if extra_terms:
        parts.extend(extra_terms)

    return " ".join(p for p in parts if p).strip()
```

For v1:

```python
search_text = ocr_text_clean
```

## Step 3: Build the panel manifest

Create:

```text
src/ingest/build_manifest.py
```

This script should read:

```text
data/comics/raw/raw_panel_images/
data/comics/raw/COMICS_ocr_file.csv
data/comics/raw/predadpages.txt
```

Do not hardcode assumptions about the CSV columns or image filename format before inspecting them. Build a deterministic mapping between panel image paths and OCR rows.

Output one row per panel:

```python
{
    "panel_id": str,
    "image_path": str,
    "ocr_text_raw": str,
    "ocr_text_clean": str,
    "search_text": str,
    "comic_id": str | None,
    "book_id": str | None,
    "page_id": str | None,
    "page_num": int | None,
    "panel_num": int | None,
    "is_ad_page": bool,
    "source": "COMICS"
}
```

ID requirements:

```text
Preferred: {comic_id}:{page_num}:{panel_num}
Fallback:  comics:{sha1(image_path)}
```

Validation requirements:

- Every row must have a stable `panel_id`.
- Every row must point to an existing image file.
- Empty OCR is allowed.
- Empty OCR panels must still be indexed because they are visually searchable.
- `search_text` must be non-null, but can be an empty string.
- Ad pages should be flagged with `is_ad_page`, not automatically dropped.
- Log unmatched panel images and unmatched OCR rows.
- Do not silently drop rows unless the image is corrupt or unreadable.

Write:

```text
data/comics/processed/panels_manifest.parquet
data/comics/processed/panels_manifest.jsonl
```

Suggested skeleton:

```python
from pathlib import Path
import hashlib
import pandas as pd

from src.ingest.clean_text import clean_ocr, build_search_text


COMICS_DATA_DIR = Path("data/comics")
RAW_DIR = COMICS_DATA_DIR / "raw"
PROCESSED_DIR = COMICS_DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def infer_panel_id(image_path: Path, row: dict | None = None) -> str:
    # Prefer structured IDs when comic/page/panel metadata is available.
    # Fallback to deterministic path hash.
    return f"comics:{stable_hash(str(image_path))}"


def main():
    panel_dir = RAW_DIR / "raw_panel_images"
    ocr_path = RAW_DIR / "COMICS_ocr_file.csv"
    ad_pages_path = RAW_DIR / "predadpages.txt"

    ocr_df = pd.read_csv(ocr_path)
    panel_paths = sorted(panel_dir.rglob("*"))

    panel_paths = [
        p for p in panel_paths
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    # TODO: inspect ocr_df columns and build robust mapping to panel image paths.
    print("OCR columns:", list(ocr_df.columns))
    print("Panel count:", len(panel_paths))

    rows = []
    for image_path in panel_paths:
        ocr_text_raw = ""
        ocr_text_clean = clean_ocr(ocr_text_raw)
        search_text = build_search_text(ocr_text_clean)

        rows.append({
            "panel_id": infer_panel_id(image_path),
            "image_path": str(image_path),
            "ocr_text_raw": ocr_text_raw,
            "ocr_text_clean": ocr_text_clean,
            "search_text": search_text,
            "comic_id": None,
            "book_id": None,
            "page_id": None,
            "page_num": None,
            "panel_num": None,
            "is_ad_page": False,
            "source": "COMICS",
        })

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(PROCESSED_DIR / "panels_manifest.parquet", index=False)
    manifest.to_json(PROCESSED_DIR / "panels_manifest.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
```

The agent should replace the TODO with the real filename-to-OCR mapping after inspecting `COMICS_ocr_file.csv`.

## Step 4: Create dense image embeddings

Create:

```text
src/embeddings/embed_images.py
```

Use a multimodal image embedding model suitable for text-to-image and image-to-image search. CLIP or SigLIP-style models are acceptable for v1.

Requirements:

- Embed the panel image, not the page.
- Use one dense vector per panel.
- Use the same model family at query time.
- Store the model name, dimension, and metric in `config.yaml`.
- Save vectors before upsert so the process is restartable.
- Log failed image reads.
- Do not drop rows with empty OCR.

Output:

```text
data/comics/processed/image_dense_vectors.parquet
```

Each row:

```python
{
    "panel_id": str,
    "image_dense": list[float]
}
```

Suggested skeleton:

```python
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/image_dense_vectors.parquet")


def embed_image(image: Image.Image) -> list[float]:
    """
    Use the selected CLIP/SigLIP-style image encoder.
    Must return a vector matching config.yaml dense_embedding.dimension.
    """
    raise NotImplementedError("Wire this to the selected image embedding model.")


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)

    rows = []
    failures = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        try:
            image = Image.open(row.image_path).convert("RGB")
            vector = embed_image(image)
            rows.append({
                "panel_id": row.panel_id,
                "image_dense": vector,
            })
        except Exception as exc:
            failures.append({
                "panel_id": row.panel_id,
                "image_path": row.image_path,
                "error": str(exc),
            })

    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            "data/comics/processed/image_embedding_failures.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
```

## Step 5: Create sparse text embeddings

Create:

```text
src/embeddings/embed_sparse_text.py
```

Use cleaned `search_text`.

Preferred model:

```text
pinecone-sparse-english-v0
```

Requirements:

- Use passage/document mode for indexing.
- Use query mode for user queries.
- Skip sparse embedding only when `search_text` is empty.
- Store empty text rows with no sparse vector, but keep the panel document.
- Save vectors before upsert.
- Sparse vectors must use parallel `indices` and `values` arrays.
- Respect sparse vector limits, including max non-zero values per vector.

Output:

```text
data/comics/processed/text_sparse_vectors.parquet
```

Each row:

```python
{
    "panel_id": str,
    "text_sparse": {
        "indices": [123, 456],
        "values": [0.25, 0.19]
    }
}
```

Suggested skeleton:

```python
from pathlib import Path
import pandas as pd
from tqdm import tqdm


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/text_sparse_vectors.parquet")


def embed_sparse_passage(text: str) -> dict:
    """
    Use pinecone-sparse-english-v0 or another sparse encoder in passage/document mode.
    Must return:
    {
        "indices": [...],
        "values": [...]
    }
    """
    raise NotImplementedError("Wire this to the selected sparse embedding model.")


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)

    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        text = row.search_text or ""

        if not text.strip():
            sparse = None
        else:
            sparse = embed_sparse_passage(text)

        rows.append({
            "panel_id": row.panel_id,
            "text_sparse": sparse,
        })

    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
```

## Step 6: Create the Pinecone index

Create:

```text
src/pinecone/create_index.py
```

Create one document-schema index with:

```text
image_dense   dense_vector
text_sparse   sparse_vector
ocr_text      string with full_text_search enabled
search_text   string with full_text_search enabled
```

Metadata fields should not be declared in the schema. Include them at upsert time.

Logical schema:

```python
from pinecone import Pinecone
from pinecone.preview import SchemaBuilder
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "comic-panels")
IMAGE_EMBED_DIM = int(os.environ.get("IMAGE_EMBED_DIM", "512"))
IMAGE_EMBED_METRIC = os.environ.get("IMAGE_EMBED_METRIC", "cosine")

schema = (
    SchemaBuilder()
    .add_dense_vector_field(
        "image_dense",
        dimension=IMAGE_EMBED_DIM,
        metric=IMAGE_EMBED_METRIC,
    )
    .add_sparse_vector_field("text_sparse")
    .add_string_field(
        "ocr_text",
        full_text_search={"language": "en"},
    )
    .add_string_field(
        "search_text",
        full_text_search={"language": "en"},
    )
    .build()
)

pc.preview.indexes.create(
    name=INDEX_NAME,
    schema=schema,
)
```

Important:

- Do not create separate Pinecone indexes for image, sparse, and FTS search.
- Use one document-schema index.
- Do not declare metadata-only fields in the schema.
- Plan fields carefully because schema migration may require recreating the index.
- Use the dense vector dimension from the selected image embedding model.
- Use cosine or dotproduct depending on the embedding model. If unsure, use cosine for normalized CLIP-style vectors.

## Step 7: Build Pinecone documents

Create:

```text
src/pinecone/build_documents.py
```

Join:

```text
panels_manifest.parquet
image_dense_vectors.parquet
text_sparse_vectors.parquet
```

Output:

```text
data/comics/processed/pinecone_documents.jsonl
```

Each document:

```python
{
    "_id": panel_id,

    "image_dense": image_dense,

    # Only include this when available and non-empty.
    "text_sparse": {
        "indices": [...],
        "values": [...]
    },

    "ocr_text": ocr_text_clean,
    "search_text": search_text,

    # Metadata fields:
    "comic_id": comic_id,
    "book_id": book_id,
    "page_id": page_id,
    "page_num": page_num,
    "panel_num": panel_num,
    "image_path": image_path,
    "source": "COMICS",
    "is_ad_page": is_ad_page
}
```

Requirements:

- Do not include raw image bytes.
- Do not include huge blobs.
- Store local image paths for now.
- Later, image paths can become S3/GCS/R2 URLs.
- Preserve panels with empty OCR.
- Preserve panels with missing sparse vectors.
- Every document must have `_id`, `image_dense`, `ocr_text`, `search_text`, and metadata.
- Only include `text_sparse` when it has non-empty `indices` and `values`.

Suggested skeleton:

```python
from pathlib import Path
import json
import pandas as pd


PROCESSED_DIR = Path("data/comics/processed")
MANIFEST_PATH = PROCESSED_DIR / "panels_manifest.parquet"
DENSE_PATH = PROCESSED_DIR / "image_dense_vectors.parquet"
SPARSE_PATH = PROCESSED_DIR / "text_sparse_vectors.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pinecone_documents.jsonl"


def valid_sparse(sparse) -> bool:
    if not isinstance(sparse, dict):
        return False
    return bool(sparse.get("indices")) and bool(sparse.get("values"))


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    dense = pd.read_parquet(DENSE_PATH)
    sparse = pd.read_parquet(SPARSE_PATH)

    df = manifest.merge(dense, on="panel_id", how="inner")
    df = df.merge(sparse, on="panel_id", how="left")

    with open(OUTPUT_PATH, "w") as f:
        for row in df.itertuples(index=False):
            doc = {
                "_id": row.panel_id,
                "image_dense": row.image_dense,
                "ocr_text": row.ocr_text_clean or "",
                "search_text": row.search_text or "",
                "comic_id": row.comic_id,
                "book_id": row.book_id,
                "page_id": row.page_id,
                "page_num": row.page_num,
                "panel_num": row.panel_num,
                "image_path": row.image_path,
                "source": "COMICS",
                "is_ad_page": bool(row.is_ad_page),
            }

            sparse_vec = getattr(row, "text_sparse", None)
            if valid_sparse(sparse_vec):
                doc["text_sparse"] = sparse_vec

            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
```

## Step 8: Upsert documents to Pinecone

Create:

```text
src/pinecone/upsert_panels.py
```

Pseudo-code:

```python
import json
import os
import time
from pathlib import Path

from pinecone import Pinecone


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "comic-panels")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "comics-v1")
DOCS_PATH = Path("data/comics/processed/pinecone_documents.jsonl")

index = pc.preview.index(name=INDEX_NAME)


def batched_jsonl(path: Path, batch_size: int = 100):
    batch = []
    with open(path) as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def upsert_with_retry(batch, retries=5):
    for attempt in range(retries):
        try:
            return index.documents.upsert(
                namespace=NAMESPACE,
                documents=batch,
            )
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    for batch in batched_jsonl(DOCS_PATH, batch_size=100):
        upsert_with_retry(batch)


if __name__ == "__main__":
    main()
```

Requirements:

- Upserts must be idempotent.
- Re-running the job should overwrite/update the same `panel_id`.
- Use deterministic IDs.
- Batch writes.
- Log batch failures.
- Retry transient failures with exponential backoff.
- After upsert, run a quick validation search.

## Step 9: Dense search primitive

Create:

```text
src/search/search_dense.py
```

Use for visual/conceptual queries:

```text
a detective in a dark alley
a superhero punching a robot
a close-up of a frightened face
```

Process:

1. Embed the user query using the text side of the same multimodal model used for panel image embeddings.
2. Search the `image_dense` field.
3. Return panel metadata and OCR fields.

```python
def search_dense(index, namespace, query_vector, top_k=20, filters=None):
    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            {
                "type": "dense_vector",
                "field": "image_dense",
                "values": query_vector,
            }
        ],
        filter=filters or {},
        include_fields=[
            "ocr_text",
            "search_text",
            "comic_id",
            "book_id",
            "page_id",
            "page_num",
            "panel_num",
            "image_path",
            "source",
            "is_ad_page",
        ],
    )
```

## Step 10: Sparse search primitive

Create:

```text
src/search/search_sparse.py
```

Use for text-heavy lexical/conceptual queries:

```text
masked villain laboratory
secret treasure island
police detective murder clue
```

Process:

1. Sparse-embed the query using query mode.
2. Search the `text_sparse` field.
3. Return panel metadata and OCR fields.

```python
def search_sparse(index, namespace, query_sparse, top_k=20, filters=None):
    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            {
                "type": "sparse_vector",
                "field": "text_sparse",
                "sparse_values": query_sparse,
            }
        ],
        filter=filters or {},
        include_fields=[
            "ocr_text",
            "search_text",
            "comic_id",
            "book_id",
            "page_id",
            "page_num",
            "panel_num",
            "image_path",
            "source",
            "is_ad_page",
        ],
    )
```

## Step 11: Full-text search primitive

Create:

```text
src/search/search_fts.py
```

Use for:

```text
exact dialogue
quoted phrases
character names
sound effects
Boolean-style keyword search
```

Examples:

```text
"secret formula"
BANG
laboratory AND formula
treasure NOT island
```

Basic FTS search:

```python
def search_fts(index, namespace, query, top_k=20, filters=None):
    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            {
                "type": "text",
                "field": "search_text",
                "query": query,
            }
        ],
        filter=filters or {},
        include_fields=[
            "ocr_text",
            "search_text",
            "comic_id",
            "book_id",
            "page_id",
            "page_num",
            "panel_num",
            "image_path",
            "source",
            "is_ad_page",
        ],
    )
```

For exact phrase filters, use FTS-enabled string fields as filters:

```python
def search_dense_with_phrase_filter(
    index,
    namespace,
    query_vector,
    phrase,
    top_k=20,
    filters=None,
):
    merged_filter = filters.copy() if filters else {}
    merged_filter["search_text"] = {"$match_phrase": phrase}

    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[
            {
                "type": "dense_vector",
                "field": "image_dense",
                "values": query_vector,
            }
        ],
        filter=merged_filter,
        include_fields=[
            "ocr_text",
            "search_text",
            "comic_id",
            "book_id",
            "page_id",
            "page_num",
            "panel_num",
            "image_path",
            "source",
            "is_ad_page",
        ],
    )
```

## Step 12: Combined search primitive

Create:

```text
src/search/search_combined.py
src/search/fusion.py
```

A single Pinecone document search request ranks by one signal. For a true blend of dense, sparse, and FTS, run separate searches and merge client-side.

Use Reciprocal Rank Fusion for v1.

```python
def rrf_merge(result_groups, k=60):
    scores = {}
    payloads = {}

    for source_name, hits in result_groups:
        for rank, hit in enumerate(hits, start=1):
            panel_id = hit["_id"]

            scores.setdefault(panel_id, 0.0)
            scores[panel_id] += 1.0 / (k + rank)

            if panel_id not in payloads:
                payloads[panel_id] = dict(hit)
                payloads[panel_id]["sources"] = []

            payloads[panel_id]["sources"].append(source_name)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    output = []
    for panel_id, score in ranked:
        item = payloads[panel_id]
        item["rrf_score"] = score
        output.append(item)

    return output
```

Combined search:

```python
from src.search.search_dense import search_dense
from src.search.search_sparse import search_sparse
from src.search.search_fts import search_fts
from src.search.fusion import rrf_merge


def search_combined(
    index,
    namespace,
    query_text,
    dense_query_vector,
    sparse_query_vector=None,
    top_k=20,
    filters=None,
    run_fts=True,
):
    result_groups = []

    dense_response = search_dense(
        index=index,
        namespace=namespace,
        query_vector=dense_query_vector,
        top_k=top_k,
        filters=filters,
    )
    result_groups.append(("dense", dense_response["result"]["hits"]))

    if sparse_query_vector:
        sparse_response = search_sparse(
            index=index,
            namespace=namespace,
            query_sparse=sparse_query_vector,
            top_k=top_k,
            filters=filters,
        )
        result_groups.append(("sparse", sparse_response["result"]["hits"]))

    if run_fts:
        fts_response = search_fts(
            index=index,
            namespace=namespace,
            query=query_text,
            top_k=top_k,
            filters=filters,
        )
        result_groups.append(("fts", fts_response["result"]["hits"]))

    return rrf_merge(result_groups)[:top_k]
```

Do not directly compare raw dense, sparse, and BM25 scores. Merge by rank unless there is a later calibration step.

## Step 13: Query router

Create:

```text
src/search/router.py
```

The router should decide which signals to run.

```python
def route_query(query: str) -> dict:
    q = query.strip()
    tokens = q.split()

    has_quotes = '"' in q or "'" in q
    has_boolean = any(t.upper() in {"AND", "OR", "NOT"} for t in tokens)
    has_all_caps = any(t.isupper() and len(t) >= 3 for t in tokens)
    has_exact_intent = any(
        phrase in q.lower()
        for phrase in [
            "says",
            "where they say",
            "exact",
            "phrase",
            "quote",
            "dialogue",
            "sound effect",
            "contains",
        ]
    )

    return {
        "dense": True,
        "sparse": bool(q),
        "fts": has_quotes or has_boolean or has_all_caps or has_exact_intent,
    }
```

Default behavior:

```text
Natural-language visual query: dense + sparse
Quoted/exact/dialogue query: dense + sparse + FTS
All-caps sound effect query: dense + sparse + FTS
Boolean query: FTS + optional dense/sparse
```

## Step 14: Jupyter query notebook

Create:

```text
notebooks/query_comics_index.ipynb
```

This notebook is not a UI. It is a developer-facing query playground for validating search quality, inspecting returned panels, and comparing dense, sparse, FTS, and combined retrieval.

The notebook should make it easy to:

1. Connect to the existing Pinecone index.
2. Run dense image search from a text query.
3. Run sparse text search from a text query.
4. Run full-text search against OCR/search text.
5. Run combined search with RRF.
6. Display returned panel images inline.
7. Compare results across search modes.
8. Inspect metadata, OCR text, and source signals.

The notebook should call the repo’s search primitives rather than duplicating search logic.

### Notebook setup cell

```python
import os
import sys
from pathlib import Path

from pinecone import Pinecone
from PIL import Image
import pandas as pd
from IPython.display import display, Markdown

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "comic-panels")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "comics-v1")
COMICS_DATA_DIR = Path(os.environ.get("COMICS_DATA_DIR", "data/comics"))

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.preview.index(name=PINECONE_INDEX_NAME)

print("Connected to:", PINECONE_INDEX_NAME)
print("Namespace:", PINECONE_NAMESPACE)
```

### Notebook imports cell

```python
from src.search.search_dense import search_dense
from src.search.search_sparse import search_sparse
from src.search.search_fts import search_fts
from src.search.search_combined import search_combined
from src.search.router import route_query
```

### Notebook embedding helper cell

The notebook needs helpers to embed queries. These should call the same embedding modules/config used during ingestion.

```python
def embed_query_dense(query: str):
    """
    Use the text side of the same multimodal model used for panel image embeddings.
    Must return a dense vector with the same dimension as image_dense.
    """
    raise NotImplementedError("Wire this to the selected CLIP/SigLIP query embedding function.")


def embed_query_sparse(query: str):
    """
    Use the sparse model in query mode.
    Must return:
    {
        "indices": [...],
        "values": [...]
    }
    """
    raise NotImplementedError("Wire this to pinecone-sparse-english-v0 query embedding.")
```

Agents should replace these stubs with calls into the project’s embedding modules once those modules exist.

### Notebook display helper cell

```python
def display_results(results, max_items=10, image_width=300):
    for i, hit in enumerate(results[:max_items], start=1):
        fields = hit.get("fields", hit)

        panel_id = hit.get("_id") or hit.get("panel_id")
        score = hit.get("_score") or hit.get("score") or hit.get("rrf_score")
        sources = hit.get("sources", fields.get("sources", []))

        image_path = fields.get("image_path") or hit.get("image_path")
        ocr_text = fields.get("ocr_text") or hit.get("ocr_text", "")
        comic_id = fields.get("comic_id") or hit.get("comic_id")
        page_num = fields.get("page_num") or hit.get("page_num")
        panel_num = fields.get("panel_num") or hit.get("panel_num")

        display(Markdown(f"### {i}. `{panel_id}`"))
        display(Markdown(
            f"**Score:** `{score}`  \n"
            f"**Sources:** `{sources}`  \n"
            f"**Comic:** `{comic_id}`  \n"
            f"**Page:** `{page_num}`  \n"
            f"**Panel:** `{panel_num}`"
        ))

        if image_path and Path(image_path).exists():
            img = Image.open(image_path)
            display(img.resize((image_width, int(image_width * img.height / img.width))))
        else:
            display(Markdown(f"`Image not found: {image_path}`"))

        if ocr_text:
            display(Markdown(f"**OCR:** {ocr_text}"))

        display(Markdown("---"))
```

### Dense search section

```python
query = "a detective in a dark alley"
top_k = 10

dense_vector = embed_query_dense(query)

dense_response = search_dense(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query_vector=dense_vector,
    top_k=top_k,
    filters={"is_ad_page": False},
)

dense_hits = dense_response["result"]["hits"]
display_results(dense_hits, max_items=top_k)
```

Use for searches like:

```text
a superhero flying through the sky
a monster attacking a city
a close-up of a frightened face
a robot in a laboratory
```

### Sparse search section

```python
query = "masked villain laboratory"
top_k = 10

sparse_vector = embed_query_sparse(query)

sparse_response = search_sparse(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query_sparse=sparse_vector,
    top_k=top_k,
    filters={"is_ad_page": False},
)

sparse_hits = sparse_response["result"]["hits"]
display_results(sparse_hits, max_items=top_k)
```

Use for searches like:

```text
secret formula
police detective clue
treasure island
masked villain laboratory
```

### Full-text search section

```python
query = '"secret formula"'
top_k = 10

fts_response = search_fts(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query=query,
    top_k=top_k,
    filters={"is_ad_page": False},
)

fts_hits = fts_response["result"]["hits"]
display_results(fts_hits, max_items=top_k)
```

Use for searches like:

```text
"secret formula"
BANG
WHAM
laboratory AND formula
treasure NOT island
```

### Combined search section

```python
query = "a laboratory panel containing formula"
top_k = 10

route = route_query(query)

dense_vector = embed_query_dense(query) if route["dense"] else None
sparse_vector = embed_query_sparse(query) if route["sparse"] else None

combined_hits = search_combined(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query_text=query,
    dense_query_vector=dense_vector,
    sparse_query_vector=sparse_vector,
    top_k=top_k,
    filters={"is_ad_page": False},
    run_fts=route["fts"],
)

display_results(combined_hits, max_items=top_k)
```

### Compare modes section

```python
query = "secret formula laboratory"
top_k = 10

dense_vector = embed_query_dense(query)
sparse_vector = embed_query_sparse(query)

dense_hits = search_dense(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query_vector=dense_vector,
    top_k=top_k,
    filters={"is_ad_page": False},
)["result"]["hits"]

sparse_hits = search_sparse(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query_sparse=sparse_vector,
    top_k=top_k,
    filters={"is_ad_page": False},
)["result"]["hits"]

fts_hits = search_fts(
    index=index,
    namespace=PINECONE_NAMESPACE,
    query=query,
    top_k=top_k,
    filters={"is_ad_page": False},
)["result"]["hits"]


def hits_to_rows(mode, hits):
    rows = []
    for rank, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {})
        rows.append({
            "mode": mode,
            "rank": rank,
            "panel_id": hit.get("_id"),
            "score": hit.get("_score"),
            "ocr_text": fields.get("ocr_text"),
            "image_path": fields.get("image_path"),
            "comic_id": fields.get("comic_id"),
            "page_num": fields.get("page_num"),
            "panel_num": fields.get("panel_num"),
        })
    return rows


df = pd.DataFrame(
    hits_to_rows("dense", dense_hits)
    + hits_to_rows("sparse", sparse_hits)
    + hits_to_rows("fts", fts_hits)
)

df
```

### Metadata filtering examples

```python
filters = {
    "is_ad_page": False
}
```

```python
filters = {
    "comic_id": "some_comic_id",
    "is_ad_page": False
}
```

```python
filters = {
    "page_num": {"$gte": 5, "$lte": 20},
    "is_ad_page": False
}
```

### Manual evaluation section

```python
test_queries = [
    '"secret formula"',
    "BANG",
    "a detective in a dark alley",
    "a superhero flying through the sky",
    "masked villain laboratory",
    "a laboratory panel containing formula",
]

for q in test_queries:
    print("=" * 80)
    print("Query:", q)

    route = route_query(q)
    dense_vector = embed_query_dense(q) if route["dense"] else None
    sparse_vector = embed_query_sparse(q) if route["sparse"] else None

    hits = search_combined(
        index=index,
        namespace=PINECONE_NAMESPACE,
        query_text=q,
        dense_query_vector=dense_vector,
        sparse_query_vector=sparse_vector,
        top_k=5,
        filters={"is_ad_page": False},
        run_fts=route["fts"],
    )

    display_results(hits, max_items=5)
```

## README command flow

The README should let a new developer run:

```bash
python -m src.ingest.build_manifest
python -m src.embeddings.embed_images
python -m src.embeddings.embed_sparse_text
python -m src.pinecone.create_index
python -m src.pinecone.build_documents
python -m src.pinecone.upsert_panels
```

Then launch the query notebook:

```bash
jupyter lab notebooks/query_comics_index.ipynb
```

or:

```bash
jupyter notebook notebooks/query_comics_index.ipynb
```

Optional search smoke tests:

```bash
python -m src.search.search_fts --query '"secret formula"'
python -m src.search.search_dense --query "a detective in a dark alley"
python -m src.search.search_combined --query "a laboratory panel containing formula"
```

## Non-goals

Do not build:

- UI
- Streamlit app
- FastAPI app
- Story builder
- Panel sequencing system
- Generative story construction
- Fine-tuning pipeline
- ComicsPAP task evaluation
- Page-level search as the primary retrieval unit

## Final deliverables

The agent should produce:

1. COMICS download instructions.
2. Manifest builder.
3. OCR cleaning module.
4. Dense image embedding script.
5. Sparse text embedding script.
6. Pinecone document-schema index creation script.
7. Pinecone document builder.
8. Pinecone upsert script.
9. Dense search primitive.
10. Sparse search primitive.
11. FTS search primitive.
12. Combined search primitive with RRF.
13. Query router.
14. Jupyter notebook for querying the Pinecone index.
15. README command flow.
16. `config.yaml`.
17. `requirements.txt`.

## Acceptance criteria

The build is complete when:

- The COMICS panel manifest exists.
- Each panel has a deterministic `panel_id`.
- Raw and cleaned OCR fields exist.
- Panel image embeddings have been generated and saved.
- Sparse text embeddings have been generated and saved where text exists.
- A Pinecone document-schema index exists with:
  - `image_dense`
  - `text_sparse`
  - `ocr_text`
  - `search_text`
- Documents have been upserted into the configured namespace.
- Dense search returns relevant visual/conceptual panels.
- Sparse search returns relevant text-heavy panels.
- FTS search returns panels with matching dialogue, names, phrases, or sound effects.
- Combined search merges dense, sparse, and FTS results with RRF.
- The Jupyter notebook can run each search mode and display panel images inline.
