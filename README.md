---
title: Comic Panel Search
emoji: 💥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

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
```

The Streamlit app combines these signals interactively:

```bash
streamlit run app.py
```

## FSD50K Sound Index

A **separate, dense-only** index lets you find sound effects that match a comic panel, using [`Fhrozen/FSD50k`](https://huggingface.co/datasets/Fhrozen/FSD50k) audio embedded with [CLAP](https://huggingface.co/laion/clap-htsat-unfused).

The comic index is unchanged:

- `comic-panels` · namespace `comics-v1` · document-schema (dense + sparse + FTS)

The sound index is independent:

- `comic-sounds` · namespace `fsd50k-commercial-v1` · standard dense index, CLAP audio vectors only (512-dim, cosine)

Only **commercial-safe** clips are ingested — CC0 and CC-BY (CC-BY-NC and CC Sampling+ are excluded). CC-BY attribution is stored in metadata. FSD50K licenses are Creative Commons *URLs*, normalized to short names during ingest.

### Environment

```bash
PINECONE_SOUND_INDEX_NAME="comic-sounds"
PINECONE_SOUND_NAMESPACE="fsd50k-commercial-v1"
SOUND_DATASET_NAME="Fhrozen/FSD50k"
CLAP_MODEL_NAME="laion/clap-htsat-unfused"
CLAP_EMBED_DIM="512"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
# optional: HF_TOKEN to speed up clip downloads
```

### Run the sound pipeline

```bash
# v1 ingests the commercial-safe EVAL split. SOUND_LIMIT caps clips for a dry run.
SOUND_LIMIT=500 ./scripts/run_sound_pipeline.sh   # 500-clip sample
./scripts/run_sound_pipeline.sh                   # full eval split (~8.4k clips)
```

Or step by step:

```bash
python -m src.sounds.ingest.build_fsd50k_manifest --split eval   # license-filtered manifest + per-clip audio
python -m src.sounds.embeddings.embed_fsd50k_audio               # CLAP audio vectors (resumable)
python -m src.sounds.pinecone.create_sound_index                 # dense index
python -m src.sounds.pinecone.build_sound_vectors                # join manifest + vectors → JSONL
python -m src.sounds.pinecone.upsert_sounds                      # batch upsert
```

Audio is pulled per-file from the HF Hub (individual wavs, resampled 44.1→48 kHz for CLAP) — the dataset's remote loader script is **not** used (`datasets`>=3.0 won't run it on Python 3.13).

### How a panel maps to sounds (no LLM)

`src/sounds/search/panel_to_sound_query.py` builds a CLAP **text** query from a panel using two signals:

1. **OCR onomatopoeia** — drawn sound words (`BANG`, `SMASH`, `ZAP`) → curated audio descriptors. Wins when present.
2. **CLIP image tags** — when there's no drawn SFX, the panel's OpenCLIP vector is scored against the FSD50K class names *within CLIP's own space* and the confident labels become query words. (CLIP and CLAP embeddings are **not** interchangeable — a CLIP vector queried against CLAP returns noise; the only bridge is text.)

If neither signal fires, the query is gated so the UI stays silent rather than returning noise. In the Streamlit app, each result carries a **Find sounds** button that runs this and plays matching clips inline with license/attribution.

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
    fusion.py             # Reciprocal Rank Fusion
  sounds/                 # FSD50K sound-effect index (dense-only, CLAP)
    config.py             # Sound index config (env vars + defaults)
    ingest/
      build_fsd50k_manifest.py  # License-filtered manifest + per-clip audio
    embeddings/
      clap_model.py             # CLAP audio/text encoder (48 kHz)
      embed_fsd50k_audio.py     # CLAP audio → vectors (resumable)
      embed_sound_query.py      # CLAP text query embedding
    pinecone/
      create_sound_index.py     # Standard dense index
      build_sound_vectors.py    # Join → JSONL records
      upsert_sounds.py          # Batch upsert with retry
    search/
      search_sounds_dense.py    # Dense sound search (commercial-safe filter)
      panel_to_sound_query.py   # Panel → CLAP query (OCR + CLIP tags)
      clip_tagger.py            # CLIP image → FSD50K class words
app.py                    # Streamlit search UI (+ "Find sounds" button)
scripts/
  run_sound_pipeline.sh   # End-to-end sound index pipeline
notebooks/
  query_comics_index.ipynb  # Interactive search playground
data/
  comics/
    raw/                  # Downloaded COMICS files
    processed/            # Manifest, vectors, and Pinecone documents
  sounds/fsd50k/
    raw/                  # (HF cache holds the wavs)
    processed/            # Manifest, CLAP vectors, Pinecone records
config.yaml               # Model and field configuration
```

## Configuration

`config.yaml` controls the embedding model, index name, namespace, and field names. If you change `dense_embedding.dimension`, update it before running `create_index.py`.

Default dense model: `open_clip_ViT-B-16` (512-dim, cosine).

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
