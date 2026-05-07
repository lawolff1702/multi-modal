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

One dense index (`comics-multimodal`) with 3 namespaces, each representing a different embedding strategy. This lets readers compare results side-by-side on the same query.

| Namespace | Embedding source | Model | Dim | Also has |
|---|---|---|---|---|
| `ns-text-caption` | LLM scene description → dense text embedding | `llama-text-embed-v2` | 1024 | FTS on dialogue |
| `ns-image-clip` | Raw comic page image → multimodal embedding | `voyage-multimodal-3` | 1024 | — |
| `ns-hybrid` | LLM description (dense) + dialogue (sparse + FTS) | `llama-text-embed-v2` + `pinecone-sparse-english-v0` | 1024 | FTS + sparse |

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
- API: `voyageai` Python package
- Cost: ~$0.000006/image
- Handles both images and text queries in the same embedding space — enables cross-modal search

**Pipeline:**
1. Load each page image (`PIL.Image`)
2. Call Voyage AI API: `voyageai.Client().embed([image], model="voyage-multimodal-3", input_type="document")`
3. Upsert 1024-dim vectors to `ns-image-clip` namespace

**Demo queries:**
- Text-to-image: `"dark alley chase scene"` — embed the query text with `input_type="query"`, search
- Image-to-image: Use another comic page as query input (find visually similar pages)
- Cross-comic visual style matching: find pages with similar visual composition across comics

**Blog post insight:** CLIP-style embeddings capture visual style and composition that text descriptions miss. But they're less precise for semantic concepts like "betrayal" or "revelation."

**Open question:** Image model could also be Jina CLIP v2 (open source, free) — decide based on whether we want to avoid an additional API key for readers. Jina CLIP v2 via HuggingFace is more accessible but requires GPU or is slower on CPU.

---

## Notebook 04 — Approach 3: Full Hybrid (Dense + Sparse + FTS)

**Pipeline:**
1. Dense vector: Same as Approach 1 (LLM caption → `llama-text-embed-v2`)
2. Sparse vector: Embed the dialogue text using `pinecone-sparse-english-v0`
3. FTS field: Dialogue text with BM25 (via `SchemaBuilder`)

**Index type:** Dense index (GA) with explicit sparse vector stored alongside dense, plus a separate FTS index for dialogue — or if Pinecone supports it, a single hybrid index.

**Query types to demonstrate:**
- Dense-only search (semantic)
- Sparse-only search (keyword)
- FTS search (BM25)
- Hybrid (dense + sparse RRF/weighted fusion)
- Triple fusion: dense + sparse + FTS combined

**Blog post insight:** Hybrid gives best recall and precision. Dense retrieves conceptually similar scenes; sparse catches exact character names, sound effects (KAPOW!), place names; FTS gives BM25 ranking on dialogue.

---

## Notebook 05 — Story Builder Demo

**Core demo:** Cross-namespace comparison + narrative assembly

**Step 1 — Comparison view:**
- User types: `"detective discovers a clue"`
- Query is sent to ALL 3 namespaces simultaneously
- Results shown side-by-side (3 columns, top-5 each)
- Visualize: display thumbnail, comic title, page num, match score

**Step 2 — Narrative assembler (multi-step):**
```
Query 1 (setup):    "detective arrives at crime scene"
Query 2 (conflict): "villain threatens the hero"  
Query 3 (climax):   "dramatic fight or confrontation"
Query 4 (resolution): "hero wins, villain defeated"
```
Each query retrieves the best matching page from the index. Display the 4 pages in sequence — a new "story" assembled from pages across multiple different comics.

**Output:** An inline IPython display of the assembled story as a 2×2 or 4×1 image grid with captions (comic title + page num).

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

1. **Pinecone index type for dense + FTS:** Need to verify if Pinecone supports both in a single index (preview API). If not, use separate indexes and merge results client-side.

2. **Image model:** Default to Voyage AI `voyage-multimodal-3`. If readers prefer open-source, include a note pointing to Jina CLIP v2 via HuggingFace Inference API as an alternative.

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
