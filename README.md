# COMICS Panel Search

Panel-level vector search over the [COMICS dataset](https://github.com/miyyer/comics) using Pinecone. Supports dense image search (OpenCLIP), sparse keyword search, full-text search over OCR text, and combined retrieval with Reciprocal Rank Fusion.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` and fill in your keys:

```bash
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="comic-panels"
PINECONE_NAMESPACE="comics-v1"
COMICS_DATA_DIR="data/comics"
```

## Step 1: Download COMICS data

```bash
mkdir -p data/comics/raw
cd data/comics/raw

wget -c https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz
wget -c https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv
wget -c https://obj.umiacs.umd.edu/comics/predadpages.txt

tar -xzf raw_panel_images.tar.gz
cd ../../..
```

> Optional — only needed if you want page-level reconstruction later:
> ```bash
> wget -c https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz
> ```

## Step 2: Run the ingestion pipeline

```bash
python -m src.ingest.build_manifest
python -m src.embeddings.embed_images
python -m src.embeddings.embed_sparse_text
python -m src.pinecone.create_index
python -m src.pinecone.build_documents
python -m src.pinecone.upsert_panels
```

Each script is idempotent and saves intermediate results to `data/comics/processed/` before the next step depends on them.

> **First run note:** `build_manifest` will print the OCR CSV column names. If the auto-detected column mapping looks wrong, update `build_ocr_lookup()` in `src/ingest/build_manifest.py` with the correct column names and re-run.

## Step 3: Query the index

```bash
jupyter lab notebooks/query_comics_index.ipynb
```

Or open the notebook directly:

```bash
jupyter notebook notebooks/query_comics_index.ipynb
```

## Search smoke tests

```bash
python -m src.search.search_fts --query '"secret formula"'
python -m src.search.search_dense --query "a detective in a dark alley"
python -m src.search.search_sparse --query "masked villain laboratory"
python -m src.search.search_combined --query "a laboratory panel containing formula"
```

## Project structure

```
src/
  ingest/
    clean_text.py         # Conservative OCR cleaning
    build_manifest.py     # Panel manifest from images + OCR CSV
  embeddings/
    embed_images.py       # OpenCLIP image → dense vectors
    embed_sparse_text.py  # pinecone-sparse-english-v0 → sparse vectors
  pinecone/
    create_index.py       # Create document-schema Pinecone index
    build_documents.py    # Join manifest + vectors → JSONL
    upsert_panels.py      # Batch upsert to Pinecone with retry
  search/
    search_dense.py       # Dense image vector search
    search_sparse.py      # Sparse text vector search
    search_fts.py         # Full-text search (BM25)
    search_combined.py    # All three signals + RRF merge
    fusion.py             # Reciprocal Rank Fusion
    router.py             # Route query to signal(s)
notebooks/
  query_comics_index.ipynb  # Interactive search playground
data/
  comics/
    raw/                  # Downloaded COMICS files
    processed/            # Manifest, vectors, and Pinecone documents
config.yaml               # Model and field configuration
```

## Configuration

`config.yaml` controls the embedding model, index name, namespace, and field names. If you change `dense_embedding.dimension`, update it before running `create_index.py`.

Default dense model: `open_clip_ViT-B-32` (512-dim, cosine).

## Data model

Each Pinecone document = one comic panel:

| Field | Type | Description |
|---|---|---|
| `_id` | string | `{comic_id}:{page_num}:{panel_num}` |
| `image_dense` | dense vector | 512-dim OpenCLIP embedding |
| `text_sparse` | sparse vector | pinecone-sparse-english-v0 on OCR text |
| `ocr_text` | string (FTS) | Raw cleaned OCR text |
| `search_text` | string (FTS) | Search-optimized text (= ocr_text for v1) |
| `comic_id` | metadata | Book identifier |
| `page_num` | metadata | Page number within book |
| `panel_num` | metadata | Panel number on the page |
| `is_ad_page` | metadata | True if page is an advertisement |
