# One Index, Four Signals: Multi-Modal Search Over a Million Comic Panels with Pinecone

Comic panels are an unusually honest multi-modal problem. Every panel is three things at once: a **picture** (the art), a **block of text** (the dialogue and captions, recovered via OCR), and a **position** (which book, which page, which panel, ad or story). If you want to search a comics archive the way a reader actually thinks, you need to support all of it:

- *"Find panels that look like this one"* — image-to-image.
- *"A detective standing in a dark alley"* — text describing an image.
- *"secret formula"* — the exact words spoken in a panel.
- *"villain AND laboratory"* — keyword filtering with boolean logic.

I built exactly that over the [COMICS dataset](https://github.com/miyyer/comics) — **1,229,664 panels** — and the whole thing runs on a *single* Pinecone document-schema index. No separate vector store bolted onto a separate search cluster bolted onto a metadata DB. One record per panel, four retrieval signals living side by side.

This post walks through how each signal works, why full-text search earns its place next to the vectors, and how Reciprocal Rank Fusion stitches them into one ranked list.

## The anti-pattern we're avoiding

The default way to build this is federation: embeddings go in a vector database, OCR text goes in Elasticsearch, structured fields go in Postgres, and your application layer fans out queries and tries to merge the results. You end up reconciling IDs across three systems, keeping three things in sync on ingest, and reconciling three incompatible score scales at query time.

Pinecone's document schema collapses that. A single record can hold multiple vector fields, multiple full-text fields, and metadata — each independently queryable, all filterable in the same call. Here's the actual index definition for the panel index:

```python
from pinecone import Pinecone
from pinecone.preview import SchemaBuilder

schema = (
    SchemaBuilder()
    .add_dense_vector_field("image_dense", dimension=512, metric="cosine")
    .add_sparse_vector_field("text_sparse")
    .add_string_field("ocr_text",    full_text_search={"language": "en"})
    .add_string_field("search_text", full_text_search={"language": "en"})
    .build()
)

pc.preview.indexes.create(name="comic-panels", schema=schema)
```

That's the whole thesis in five lines. One panel becomes one document carrying:

| Field | Type | What it's for |
|---|---|---|
| `image_dense` | dense vector (512-d) | OpenCLIP visual/semantic similarity |
| `text_sparse` | sparse vector | learned keyword weighting over dialogue |
| `ocr_text` / `search_text` | full-text fields | exact phrase + boolean queries |
| `comic_id`, `page_num`, `panel_num`, `is_ad_page` | metadata | filtering |

Now let's look at each signal and how it actually behaves.

## Signal 1: dense vectors and the CLIP bridge

The interesting trick in multi-modal search is letting a *text* query retrieve an *image*. That works because of how CLIP is trained. CLIP has two encoders — one for images, one for text — trained contrastively on hundreds of millions of (image, caption) pairs so that a matching image and caption land near each other in the **same** embedding space. The consequence: once both an image and a string of text are encoded, they're directly comparable with cosine similarity. There's no separate "text-to-image model" — it's the same space.

So image ingestion and text querying are two halves of the same coin. On ingest, we encode the panel art:

```python
def embed_images_batch(images):
    tensors = torch.stack([_preprocess(img) for img in images]).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)  # unit-normalize for cosine
    return features.cpu().tolist()
```

At query time, we encode the user's text into the *same* 512-d space:

```python
def embed_text_query(text):
    tokens = _tokenizer([text]).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()
```

Same model (OpenCLIP ViT-B/16, OpenAI weights), same normalization, two different `encode_*` calls. "A detective in a dark alley" comes out as a vector that sits near panels depicting exactly that — even though no panel was ever labeled "detective." And image-to-image search is free: just reuse a stored panel's vector as the query.

A couple of production notes baked into the embedder:

- **Device handling** — it picks `mps` on Apple silicon, `cuda` on an NVIDIA box, `cpu` otherwise. Embedding a million images, you want the accelerator.
- **Resumable chunking** — panels are processed in chunks of 10k, each written to its own parquet file. A crash at panel 900k doesn't re-embed the first 900k.

What dense is *good* at: visual and semantic similarity, fuzzy intent, cross-modal queries. What it's *bad* at: exact words, rare proper nouns, sound effects. Cosine similarity smears those into the nearest semantic neighborhood. Which is why we don't stop at dense.

## Signal 2: learned sparse vectors

A sparse vector is the opposite shape from a dense one: very high-dimensional, almost entirely zeros, with nonzero weights only on the dimensions that correspond to meaningful terms. Classic BM25 is sparse with hand-tuned term weights; `pinecone-sparse-english-v0` is a *learned* sparse model — think of it as a neural BM25 that has learned which terms actually carry signal. It keeps the precision of keyword matching while being smarter about term importance.

We run it over the OCR dialogue, through Pinecone's inference API, in the same resumable-chunk style:

```python
response = pc.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=list(batch_texts),
    parameters={"input_type": "passage"},   # "query" at search time
)
# each result -> {"indices": [...], "values": [...]}
```

Note `input_type`: documents are embedded as `passage`, the user's search string as `query`. The model treats them slightly differently, the same way BM25 distinguishes a document from a query.

Why bother, when we already have dense? Because dialogue keywords — *laboratory*, *villain*, *formula* — matter as keywords, and dense embeddings tend to average them into a single semantic blob. Sparse keeps the individual terms weighted and addressable.

One thing worth saying out loud: **OCR is noisy**, and garbage tokens in means garbage weights out. A conservative cleaning pass runs before any text becomes a sparse vector or a full-text field. Cleaning quality upstream is doing as much for retrieval quality as the model choice.

## Signal 3: Pinecone full-text search (the underrated leg)

Here's the case neither dense nor sparse handles well: the user knows the *exact words*. They want the panel where someone says `"secret formula"` — the literal phrase, not the semantic neighborhood of formulas. Or they want `formula AND detective` with boolean logic. Or they're hunting a sound effect like `BANG`. Vector search quietly gets these wrong; it returns plausibly-similar panels instead of the right one.

Full-text search is what makes the system *trustworthy* for those queries. Because we declared `ocr_text` and `search_text` as full-text fields in the schema, Pinecone indexes them for BM25-scored text retrieval, and we query them through the same `documents.search` endpoint:

```python
def search_fts(index, namespace, query, top_k=20, filters=None, fts_type="query_string"):
    if fts_type == "query_string":
        score_by = [{"type": "query_string", "query": query}]
    else:
        score_by = [{"type": "text", "field": "search_text", "query": query}]

    response = index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=score_by,
        filter=filters or {},
        include_fields=["ocr_text", "comic_id", "page_num", "panel_num", "image_path"],
    )
    return {"result": {"hits": [h.to_dict() for h in response.matches]}}
```

There are two scoring modes, and the distinction matters:

- **`query_string`** — Lucene-style syntax. Phrases (`"secret formula"`), boolean operators (`formula AND detective`), multi-field matching. This is what you reach for when the user types operators.
- **`text`** — straightforward BM25 over a single field. Free-text relevance ranking without the operator grammar.

And notice the `filter` argument sitting right alongside `score_by`. The query *"panels saying 'secret formula', excluding advertisements"* is one call: BM25 scoring plus `{"is_ad_page": False}` metadata filtering, no second round-trip. That co-location is the payoff of putting everything on one document.

FTS is the leg people forget when they get excited about embeddings. It's the one that handles "I know exactly what I'm looking for."

## Fusing the signals with Reciprocal Rank Fusion

Now the hard part: a query can fire more than one signal, and the three score scales are not comparable. Cosine similarity lives in roughly [−1, 1]. BM25 is an unbounded positive score. Sparse dot products are something else again. You cannot just add them up.

Reciprocal Rank Fusion sidesteps the problem entirely by throwing away the raw scores and keeping only the **ranks**. Each signal contributes `1 / (k + rank)` to every document it returns, and we sum those contributions across signals. A document ranked #1 by dense and #3 by FTS accrues credit from both; a document only one signal liked still shows up, just lower.

```python
def rrf_merge(result_groups, k=60):
    scores, payloads = {}, {}
    for source_name, hits in result_groups:
        for rank, hit in enumerate(hits, start=1):
            panel_id = hit.get("_id") or hit.get("panel_id")
            scores[panel_id] = scores.get(panel_id, 0.0) + 1.0 / (k + rank)
            if panel_id not in payloads:
                payloads[panel_id] = dict(hit, sources=[])
            payloads[panel_id]["sources"].append(source_name)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [dict(payloads[pid], rrf_score=s) for pid, s in ranked]
```

The constant `k` (60 here, straight from the original RRF paper) damps the influence of the very top ranks so a single signal can't completely dominate. Rank-based fusion is what makes "merge these three lists" a one-liner instead of a score-calibration project — and it degrades gracefully: with a single signal active, it just passes the list through.

The application layer turns query shape into which signals fire, then fuses whatever comes back:

```python
groups = []
if dense_on and dense_query:                       # text → CLIP space, or a reused panel vector
    vec = embed_text_query(dense_query)
    groups.append(("dense", search_dense(idx, ns, vec, top_k, filters)))
if sparse_on and sparse_query:                      # learned keyword weighting
    svec = embed_sparse_query(sparse_query)
    groups.append(("sparse", search_sparse(idx, ns, svec, top_k, filters)))
if fts_on and fts_query:                             # exact phrase / boolean
    groups.append(("fts", search_fts(idx, ns, fts_query, top_k, filters)))

merged = rrf_merge(groups)                           # one ranked list
```

Image-to-image rides the same `dense` path — instead of encoding text, it reuses the clicked panel's stored vector as the query. One code path, four behaviors.

## The pipeline, end to end

Ingestion is six idempotent, resumable stages:

```bash
python -m src.ingest.build_manifest        # panels + cleaned OCR → manifest
python -m src.embeddings.embed_images       # OpenCLIP → dense vectors (chunked parquet)
python -m src.embeddings.embed_sparse_text  # pinecone-sparse-english-v0 → sparse vectors
python -m src.pinecone.create_index         # document schema (above)
python -m src.pinecone.build_documents      # join everything → JSONL
python -m src.pinecone.upsert_panels        # batch upsert with retry
```

Each stage writes its output to disk before the next depends on it, so the whole thing is restartable at every boundary — non-negotiable at a million-plus records. The join step is deliberately asymmetric: it **inner-joins** dense vectors (a panel with no art embedding has nothing to search on, so it's dropped) and **left-joins** sparse (a wordless panel — plenty of those in comics — keeps its document and stays findable by image). The result is one JSONL document per panel, exactly matching the schema, ready to upsert.

On top sits a Streamlit app: one search box, toggles for the three text signals plus image upload, metadata filters, and an RRF-merged result grid that shows which signals surfaced each panel.

## Takeaways

- **Co-locate modalities on one record instead of federating systems.** A Pinecone document schema holds dense vectors, sparse vectors, full-text fields, and metadata together — independently queryable, jointly filterable, no cross-store ID reconciliation.
- **The three signals are complementary, not competing.** Dense (CLIP) for visual and semantic intent, sparse for weighted keyword matching, full-text for exactness. FTS is the one teams under-use, and it's exactly what rescues "I know the precise words."
- **CLIP's shared embedding space is the cross-modal unlock.** Text and images become directly comparable, so a description retrieves a picture with no extra machinery.
- **RRF is the cross-signal unlock.** Rank-based fusion makes incomparable score scales a non-issue and collapses multi-signal merging into a few lines.

One index. Four signals. A million comic panels you can search by sight, by meaning, by keyword, or by exact quote.
