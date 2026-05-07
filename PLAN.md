# Plan: Multimodal Comics Search with Pinecone

## Context

Build a blog post-quality project that demonstrates Pinecone's vector search capabilities using public-domain golden age comics from the Digital Comic Museum. The project showcases multimodal data handling — combining image embeddings, LLM-generated descriptions, sparse vectors, and full-text search across different Pinecone namespaces. The goal is thought leadership content showing real trade-offs between embedding approaches, culminating in a "story builder" demo that assembles comic panels from multiple comics into a narrative.

---

## Project Structure

```
comics-pinecone/
├── 00_data_acquisition.ipynb       # Download + extract comics
├── 01_transcription.ipynb          # comic-transcriber pipeline
├── 02_approach1_text_dense_fts.ipynb     # Approach 1: LLM caption → dense + FTS
├── 03_approach2_image_embeddings.ipynb   # Approach 2: Direct image embeddings (CLIP)
├── 04_approach3_hybrid.ipynb             # Approach 3: Dense + Sparse + FTS hybrid
├── 05_story_builder.ipynb                # Demo: cross-namespace comparison + story assembly
├── 06_panel_level_exploration.ipynb      # Bonus: panel detection + panel-level embeddings
├── data/
│   ├── raw/          # Downloaded CBZ/zip files
│   ├── pages/        # Extracted page images (PNG)
│   └── transcripts/  # comic-transcriber markdown output
├── utils/
│   ├── downloader.py
│   ├── extractor.py
│   └── embedders.py
└── .env.example
```

---

## Notebook 00 — Data Acquisition

**Goal:** Download ~5–10 comics from https://digitalcomicmuseum.com/index.php?cid=110

**Steps:**
1. Identify download links from the category page (web scraping or manual curation)
2. Download CBZ/ZIP archives
3. Extract to `data/pages/<comic_title>/page_001.jpg` etc.
4. Build a manifest CSV: `comic_title, page_num, file_path, image_url`

**Key decision:** Start with 5 comics (~50–150 pages total) to keep costs and runtime manageable for a blog post demo.

---

## Notebook 01 — Transcription Pipeline

**Goal:** Run comic-transcriber on each page and produce structured output.

**Steps:**
1. Install comic-transcriber via NPM: `npx comic-transcriber`
2. Process each page image → outputs structured markdown per page:
   ```
   ## PAGE X, PANEL Y
   [Scene description]
   **CHARACTER**: Dialogue
   ```
3. Parse the markdown output into a structured Python dict per page:
   - `page_id`: `{comic_title}_{page_num}`
   - `panels`: list of panel dicts (description, dialogue)
   - `all_dialogue`: concatenated dialogue text (for FTS)
   - `scene_summary`: concatenated descriptions (for dense embedding)
4. Save to `data/transcripts/{comic_title}.json`

**Output schema per page:**
```json
{
  "page_id": "action_comics_001_p003",
  "comic_title": "Action Comics 001",
  "page_num": 3,
  "file_path": "data/pages/action_comics_001/page_003.jpg",
  "all_dialogue": "Look out! The villains are here...",
  "scene_descriptions": "A hero leaps across rooftops under a full moon...",
  "raw_transcript": "## PAGE 3, PANEL 1..."
}
```

---

## Namespace Strategy (Single Pinecone Index)

One **hybrid index** (`comics-multimodal`) via `pc.preview.indexes.create` with `SchemaBuilder`. A single Pinecone index can hold dense + sparse + FTS fields simultaneously (confirmed in docs: https://docs.pinecone.io/guides/search/full-text-search#schema-definition). Three namespaces, each representing a different embedding strategy.

**Critical constraint:** Each query uses exactly one ranking signal (dense, sparse, or BM25/text). To combine signals, use text-match filters (`$match_phrase`, `$match_all`) to narrow candidates, then rank by vector.

| Namespace | Embedding source | Model | Dim | Also has |
|---|---|---|---|---|
| `ns-text-caption` | LLM scene description → dense text embedding | `llama-text-embed-v2` | 1024 | FTS on dialogue |
| `ns-image-clip` | Raw comic page image → multimodal embedding | `voyage-multimodal-3` | 1024 | — |
| `ns-hybrid` | LLM description (dense) + dialogue (sparse + FTS) | `llama-text-embed-v2` + `pinecone-sparse-english-v0` | 1024 | FTS + sparse |

**Index configuration (matches Pinecone web console):**

| Setting | Value |
|---|---|
| Search type | Search by both meaning and exact words |
| Embeddings | Dense + sparse |
| Setup by model | `llama-text-embed-v2` (integrated inference for text namespaces) |
| Dimension | 1,024 |
| Metric | dotproduct (unit vectors → equivalent to cosine) |
| Dense field name | `embedding` |
| Sparse field name | `sparse_embedding` |
| FTS field | `dialogue`, language: en, stemming: ON, stop words: ON |

**Index schema (via `SchemaBuilder`):**
```python
schema = (
    SchemaBuilder()
      .add_string_field(name="dialogue", full_text_search={"language": "en", "stemming": True, "stop_words": True})
      .add_string_field(name="comic_title", filterable=True)
      .add_string_field(name="page_id", filterable=True)
      .add_dense_vector_field(name="embedding", dimension=1024, metric="dotproduct")
      .add_sparse_vector_field(name="sparse_embedding")
      .build()
)
```

**Upsert pattern differs by namespace:**
- `ns-text-caption` / `ns-hybrid`: pass text field → Pinecone auto-embeds via `llama-text-embed-v2` (integrated inference)
- `ns-image-clip`: upsert raw 1024-dim Voyage AI vectors directly (bypass integrated inference)

---

## Notebook 02 — Approach 1: Text Caption + Dense + FTS

**Embedding pipeline:**
1. Use Claude (`claude-haiku-4-5-20251001`) to generate a rich scene description per page from the comic-transcriber output. Prompt focuses on: setting, action, mood, characters present, visual style.
2. Embed the description using Pinecone's integrated `llama-text-embed-v2` (via `create_index_for_model`)
3. Store dialogue text as a separate FTS field using `pc.preview.indexes` with `SchemaBuilder`

**Note:** Since we need both dense + FTS, this may require a hybrid index type or two separate calls. Check Pinecone's current support for combined dense + FTS in a single index. If not supported together, use two separate indexes and merge results at query time.

**Metadata per vector:**
```python
{
  "page_id": "action_comics_001_p003",
  "comic_title": "Action Comics 001",
  "page_num": 3,
  "image_url": "...",
  "scene_description": "...",  # the LLM caption
  "all_dialogue": "..."         # FTS field
}
```

**Demo queries in notebook:**
- `"hero discovers villain's secret lair"` → semantic search on descriptions
- `"Look out"` → FTS search on dialogue
- Combined: FTS filter + dense reranking

**Blog post insight:** LLM captions add semantic richness but lose visual detail. FTS on dialogue is great for exact quote matching.

---

## Notebook 03 — Approach 2: Direct Image Embeddings

**Model:** Voyage AI `voyage-multimodal-3` (1024-dim, cosine)
- API: `voyageai` Python package (`pip install voyageai`)
- Cost: ~$0.000006/image
- Handles both images and text queries in the same embedding space — enables cross-modal search

**Pipeline:**
```python
import voyageai
vo = voyageai.Client()  # reads VOYAGE_API_KEY from env

# Indexing: embed page image
result = vo.multimodal_embed([[page_image]], model="voyage-multimodal-3", input_type="document")
vector = result.embeddings[0]  # 1024-dim list

# Querying: embed text query into same space
result = vo.multimodal_embed([["dark alley chase scene"]], model="voyage-multimodal-3", input_type="query")
query_vector = result.embeddings[0]
```

**Demo queries:**
- Text-to-image: `"dark alley chase scene"` → embed text, search `ns-image-clip`
- Image-to-image: Use a comic page image as the query input → find visually similar pages
- Cross-comic visual style matching: find pages with similar composition across comics

**Blog post insight:** Voyage multimodal embeddings capture visual style and composition that text descriptions miss. But they're less precise for abstract semantic concepts like "betrayal."

---

## Notebook 04 — Approach 3: Full Hybrid (Dense + Sparse + FTS)

**Pipeline:**
1. Dense vector: Same as Approach 1 (LLM caption → `llama-text-embed-v2`)
2. Sparse vector: Embed the dialogue text using `pinecone-sparse-english-v0`
3. FTS field: Dialogue text with BM25 (via `SchemaBuilder`)

**Index type:** Single hybrid index via `SchemaBuilder` (dense + sparse + FTS fields in one index). All in `ns-hybrid` namespace.

**Query types to demonstrate (one ranking signal per query, per Pinecone constraint):**

| Query mode | Ranking signal | Notes |
|---|---|---|
| Dense semantic | `dense_vector` | "find scenes about justice" |
| Sparse keyword | `sparse_vector` | "KAPOW detective badge" |
| BM25 / FTS | `text` on `dialogue` field | exact phrase in speech bubbles |
| Filter + rank | Text `$match_phrase` filter → dense ranking | narrow by keyword, then rerank by semantics |

```python
# Filter by keyword, rank by vector (the "hybrid" pattern in Pinecone)
index.documents.search(
    namespace="ns-hybrid",
    top_k=5,
    score_by=[{"type": "dense_vector", "field": "embedding", "query": query_vector}],
    filter={"dialogue": {"$match_phrase": "look out"}},
)
```

**Blog post insight:** A single index holds all three signal types. The key lesson: Pinecone picks one ranker per query, but you can chain: FTS/sparse filters narrow the candidate set, then dense vector re-ranks for semantic relevance. Dense retrieves conceptually similar scenes; sparse/FTS catches exact names, sound effects (KAPOW!), specific dialogue.

---

## Notebook 05 — Story Builder Demo

**Core demo:** Two sections in one notebook.

**Section 1 — Cross-namespace comparison view:**
- User types one query: e.g. `"detective discovers a clue"`
- Same query sent to all 3 namespaces simultaneously (parallel calls)
- Results displayed side-by-side: 3 columns (ns-text-caption | ns-image-clip | ns-hybrid), top-5 each
- Each result shows: thumbnail, comic title, page num, match score
- Purpose: show readers concretely how the same query returns different results depending on embedding strategy

**Section 2 — Narrative assembler:**
```
Query 1 (setup):      "detective arrives at crime scene"
Query 2 (conflict):   "villain threatens the hero"
Query 3 (climax):     "dramatic fight or confrontation"
Query 4 (resolution): "hero wins, villain defeated"
```
- Each query fetches top-1 page from `ns-hybrid` (best all-round namespace)
- Deduplicate: if two queries return the same page, fetch next best
- Display the 4 pages in sequence as a 2×2 image grid with caption: comic title + page num
- Output: a brand new "story" assembled from panels across multiple comics

**Output format:** Inline IPython `display(Image(...))` grid — no external UI needed, renders directly in the notebook.

---

## Notebook 06 — Panel-Level Exploration (Bonus)

**Goal:** Explore whether panel-level granularity produces meaningfully better results.

**Panel detection approach:**
- Option A (simple): Use `comic-panels` Python library or OpenCV edge detection + contour finding
- Option B (ML): Use a pre-trained comic panel segmentation model (e.g. from HuggingFace)

**Pipeline:**
1. Detect panel bounding boxes per page
2. Crop individual panels (100–400 panels per comic)
3. Embed each panel (both image CLIP + LLM caption for the panel's transcribed text)
4. Upsert to a `ns-panels` namespace

**Demo:** "Find all panels of a character running" — much more precise than page-level.

**Blog post insight:** Panel-level is more powerful for the story-builder, but requires significant additional pipeline complexity. Page-level is a good starting point for most use cases.

---

## Key Technical Decisions

1. **Pinecone index type for dense + FTS:** ✅ Confirmed — a single hybrid index via `SchemaBuilder` supports dense + sparse + FTS fields. One ranking signal per query; use text-match filters + vector ranking as the "hybrid" pattern.

2. **Image model:** ✅ Confirmed — Voyage AI `voyage-multimodal-3`. Requires `VOYAGE_API_KEY`. Include Jina CLIP v2 as a free open-source footnote for readers who want no extra API key.

3. **LLM for captions:** Use `claude-haiku-4-5-20251001` (fast + cheap). Prompt should be consistent and reproducible — include it verbatim in the notebook for transparency.

4. **Rate limiting / cost controls:** Comic Transcriber + Claude captions + Voyage AI embeddings will have associated costs. Add a `MAX_PAGES` constant to each notebook so readers can test with fewer pages.

---

## Environment Variables

```
PINECONE_API_KEY=
ANTHROPIC_API_KEY=     # for LLM captions
VOYAGE_API_KEY=        # for image embeddings
```

---

## Verification Plan

1. **Notebook 00:** Verify `data/pages/` populated with images; manifest CSV has correct row count
2. **Notebook 01:** Spot-check 3 transcripts; confirm dialogue and description fields are non-empty
3. **Notebook 02:** Run 5 test queries; verify semantic matches are qualitatively relevant
4. **Notebook 03:** Run text-to-image query; verify returned images visually match query intent
5. **Notebook 04:** Compare hybrid vs. dense-only recall on 3 test queries
6. **Notebook 05:** Build one complete story arc end-to-end; display assembled panel grid
7. **Notebook 06 (bonus):** Verify panel crops look correct; compare panel-level vs. page-level for a test query

---

## Critical Files to Create

- `00_data_acquisition.ipynb`
- `01_transcription.ipynb`
- `02_approach1_text_dense_fts.ipynb`
- `03_approach2_image_embeddings.ipynb`
- `04_approach3_hybrid.ipynb`
- `05_story_builder.ipynb`
- `06_panel_level_exploration.ipynb`
- `utils/downloader.py` — comic download helpers
- `utils/extractor.py` — CBZ extraction + page listing
- `utils/embedders.py` — shared embedding functions (Voyage, llama-text-embed-v2, sparse)
- `.env.example`
- `requirements.txt`
