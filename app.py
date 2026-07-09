"""Comic panel search UI."""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.storage.images import image_src

st.set_page_config(
    page_title="Comic Panel Search",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* corner radius for custom HTML; keep in sync with theme.baseRadius in .streamlit/config.toml */
  :root { --radius: 2px; }

  #MainMenu, footer, [data-testid="collapsedControl"] { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

  /* hero header — margin-top keeps the title clear of Streamlit's fixed toolbar */
  .hero { margin-top: 2.25rem; padding-bottom: 14px; border-bottom: 1px solid #e5e5e5; }
  .hero-title { font-size: 26px; font-weight: 700; color: #0a0a0a; line-height: 1.2; }
  .hero-tag { font-size: 14px; color: #404040; margin-top: 4px; }
  .hero-caps { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .hero-chip {
    font-size: 11px; font-weight: 600; color: #002BFF;
    background: rgba(0,43,255,0.06); border-radius: var(--radius);
    padding: 3px 8px; cursor: default;
  }
  .hero-meta { font-size: 11px; color: #737373; margin-left: auto; }

  /* example strip: equal-width chips; extra margin separates it from the hero rule */
  div.st-key-strip { margin-top: 12px; }
  /* "Try:" label — Streamlit vertically centers the column by its wrapper box
     (~10px) while the label's line box (~26px) overflows below it; nudge the
     text up half the overflow so it sits on the chips' true centerline */
  div.st-key-strip [data-testid="stColumn"]:first-of-type .section-label {
    transform: translateY(-8px);
  }
  div[class*="st-key-strip_"] button {
    min-height: 30px !important;
    padding: 2px 12px !important;
  }
  div[class*="st-key-strip_"] button p {
    font-size: 12px !important;
    white-space: nowrap !important;
  }

  /* result card */
  .rc {
    background: #fff;
    border: 1px solid #e5e5e5;
    border-radius: var(--radius);
    padding: 18px 20px;
    display: grid;
    grid-template-columns: 62px 1fr;
    gap: 18px;
    margin-bottom: 10px;
  }
  .rc:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
  .rc-left {
    display: flex; flex-direction: column;
    align-items: center; gap: 8px; padding-top: 2px;
  }
  .rc-rank {
    width: 40px; height: 40px;
    background: rgba(0,43,255,0.05); border-radius: var(--radius);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; color: #0a0a0a; flex-shrink: 0;
  }
  .rc-score-lbl {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #737373; text-align: center;
  }
  .rc-score-val { font-family: monospace; font-size: 12px; color: #0a0a0a; text-align: center; }
  .rc-tag {
    font-size: 10px; background: rgba(0,43,255,0.06);
    color: #002BFF; border-radius: var(--radius);
    padding: 2px 5px; font-weight: 600; display: inline-block; margin: 1px;
  }
  .rc-right { display: flex; flex-direction: column; gap: 8px; min-width: 0; }

  /* "See more like this" button — top of right column, styled as a link */
  div[data-testid="stHorizontalBlock"]:has(.rc) > div[data-testid="column"]:last-child button {
    background: none !important; border: none !important;
    box-shadow: none !important; padding: 0 !important;
    min-height: unset !important; height: auto !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.rc) > div[data-testid="column"]:last-child button p {
    color: #002BFF !important; text-decoration: underline !important;
    font-size: 12px !important; white-space: nowrap !important; margin: 0 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.rc) > div[data-testid="column"]:last-child button:hover p {
    color: #0020CC !important;
  }
  /* panel action buttons (See more / Find sounds): identical, centered */
  div[class*="st-key-sim_"] button,
  div[class*="st-key-snd_"] button {
    width: 100% !important;
  }
  div[class*="st-key-sim_"] button p,
  div[class*="st-key-snd_"] button p {
    font-size: 14px !important;
    white-space: normal !important;
    text-align: center !important;
    width: 100%;
    margin: 0 auto !important;
  }
  .rc-img {
    max-width: 100%; max-height: 65vh;
    width: auto; height: auto;
    border-radius: var(--radius); display: block;
  }
  .rc-id {
    font-family: monospace; font-size: 13px;
    font-weight: 600; color: #0a0a0a;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .rc-field {
    display: grid; grid-template-columns: 95px 1fr;
    gap: 6px; font-size: 12px; line-height: 1.5;
  }
  .rc-key {
    color: #737373; font-family: monospace;
    font-size: 11px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .rc-val {
    color: #0a0a0a; word-break: break-word;
  }
  .rh {
    font-size: 13px; color: #737373;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e5e5;
    margin-bottom: 14px;
  }
  .rh strong { color: #0a0a0a; }
  .ph {
    text-align: center; padding: 72px 0;
    color: #737373; font-size: 14px;
  }
  .section-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #737373; margin: 0; white-space: nowrap;
  }

  /* disable typing in all selectboxes — click still opens the dropdown */
  div[data-testid="stSelectbox"] input { pointer-events: none; }

  /* filter editor popover: enough room for full-width field/operator/value controls */
  div[data-testid="stPopoverBody"] { min-width: 320px; }
  /* filter summary chips: left-align the text; clamp to one line with ellipsis */
  div[class*="st-key-fp_"] button { justify-content: flex-start !important; }
  div[class*="st-key-fp_"] button > div { min-width: 0; overflow: hidden; }
  div[class*="st-key-fp_"] button > div > div:first-child { min-width: 0 !important; overflow: hidden; }
  div[class*="st-key-fp_"] button > div > div[aria-hidden="true"] { flex-shrink: 0 !important; }
  div[class*="st-key-fp_"] button p {
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  /* center the ✕ glyph in filter remove buttons */
  div[class*="st-key-fr_"] button {
    display: flex; align-items: center; justify-content: center;
    padding: 0 !important; min-width: 0 !important; width: 100% !important;
  }
  div[class*="st-key-fr_"] button > div {
    display: flex; align-items: center; justify-content: center;
    width: 100%; height: 100%;
  }
  div[class*="st-key-fr_"] button p {
    line-height: 1 !important; margin: 0 !important; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── constants & resources ─────────────────────────────────────────────────────

NAMESPACE   = os.environ.get("PINECONE_NAMESPACE", "comics-v1")
SHOW_FIELDS = ["comic_id", "book_id", "page_num", "panel_num", "source", "ocr_text"]

OPS = ["==", "!=", ">", ">=", "<", "<=", "In", "Not In", "Match phrase", "Match all", "Match any"]
_MATCH_OPS = {"Match phrase", "Match all", "Match any"}
_OP_MAP = {
    "==": "$eq",  "!=": "$ne",
    ">":  "$gt",  ">=": "$gte",
    "<":  "$lt",  "<=": "$lte",
    "In": "$in",  "Not In": "$nin",
    "Match phrase": "$match_phrase",
    "Match all":    "$match_all",
    "Match any":    "$match_any",
}
# Compact operator labels for the one-line filter summary chips.
_OP_SHORT = {
    "==": "=", "!=": "≠", ">": ">", ">=": "≥", "<": "<", "<=": "≤",
    "In": "in", "Not In": "not in",
    "Match phrase": "phrase", "Match all": "has all", "Match any": "has any",
}
ALL_FIELDS = ["comic_id", "book_id", "page_id", "page_num", "panel_num", "source", "ocr_text", "search_text", "image_path", "is_ad_page"]

@st.cache_resource(show_spinner="Connecting to Pinecone…")
def _index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))

@st.cache_resource(show_spinner="Loading CLIP model…")
def _load_clip():
    from src.embeddings.embed_images import _init_model
    _init_model()

# The sound index is a SEPARATE, standard dense index (classic API), unlike the
# comic-panels document-schema index above — see src/sounds/.
@st.cache_resource(show_spinner="Connecting to sound index…")
def _sound_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ.get("PINECONE_SOUND_INDEX_NAME", "comic-sounds"))

# ── helpers ───────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _coerce(s: str):
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    if s.lower() in ("true", "yes"): return True
    if s.lower() in ("false", "no"): return False
    return s

def _add_filter():
    fid = st.session_state.filter_next_id
    st.session_state.filter_rows = st.session_state.filter_rows + [fid]
    st.session_state.filter_next_id += 1
    st.session_state[f"fk_{fid}"] = "search_text"

def _remove_filter(fid: int):
    st.session_state.filter_rows = [r for r in st.session_state.filter_rows if r != fid]

def _clear_filters():
    """Drop every filter row and its backing widget state — fresh slate for a new query."""
    for fid in st.session_state.filter_rows:
        for prefix in ("fk_", "fo_", "fv_"):
            st.session_state.pop(f"{prefix}{fid}", None)
    st.session_state.filter_rows = []

def _build_filters() -> dict:
    row_conds = []
    for fid in st.session_state.filter_rows:
        k       = st.session_state.get(f"fk_{fid}", "").strip()
        op      = st.session_state.get(f"fo_{fid}", "==")
        val_str = st.session_state.get(f"fv_{fid}", "").strip()
        if not k or not val_str:
            continue
        if op in ("In", "Not In"):
            vals = [_coerce(v.strip()) for v in val_str.split(",")]
            row_conds.append({k: {"$in" if op == "In" else "$nin": vals}})
        elif op in _MATCH_OPS:
            row_conds.append({k: {_OP_MAP[op]: val_str}})
        else:
            row_conds.append({k: {_OP_MAP[op]: _coerce(val_str)}})

    combinator = st.session_state.get("filter_combinator", "And")
    if len(row_conds) == 0:
        user_filter = None
    elif len(row_conds) == 1:
        user_filter = row_conds[0]
    elif combinator == "Or":
        user_filter = {"$or": row_conds}
    else:
        user_filter = {"$and": row_conds}

    exclude = st.session_state.get("exclude_ads", True)
    if exclude and user_filter:
        return {"$and": [{"is_ad_page": {"$eq": False}}, user_filter]}
    elif exclude:
        return {"is_ad_page": {"$eq": False}}
    elif user_filter:
        return user_filter
    return {}

def _card(hit: dict, rank: int, show_fields: list):
    score   = hit.get("rrf_score") or hit.get("_score") or 0.0
    sources = hit.get("sources") or []
    hit_id  = _esc(str(hit.get("_id", "—")))

    tags = "".join(f'<span class="rc-tag">{s}</span>' for s in sources)

    img_html = ""
    if src := image_src(hit.get("image_path", "")):
        img_html = f'<img class="rc-img" src="{src}" loading="lazy"/>'

    fields = ""
    for k in show_fields:
        v = hit.get(k)
        if v is None or str(v).strip() == "":
            continue
        s = str(v)
        if len(s) > 500:
            s = s[:500] + "…"
        fields += f'<div class="rc-field"><span class="rc-key">{k}</span><span class="rc-val">{_esc(s)}</span></div>'

    st.markdown(f"""
<div class="rc">
  <div class="rc-left">
    <div class="rc-rank">{rank}</div>
    <div class="rc-score-lbl">Score</div>
    <div class="rc-score-val">{score:.4f}</div>
    {tags}
  </div>
  <div class="rc-right">
    {img_html}
    <div class="rc-id">_id: {hit_id}</div>
    {fields}
  </div>
</div>""", unsafe_allow_html=True)


def _render_sounds(snd: dict) -> None:
    """Render the sound-effect results stored for a panel (under its card)."""
    if snd.get("error"):
        st.error(f"Sound search failed: {snd['error']}")
        return
    packet = snd.get("packet")
    if packet is None:
        st.caption("No clear sound cue in this panel (no drawn SFX, no confident visual match).")
        return

    src = "drawn SFX (OCR)" if packet["source"] == "ocr" else "depicted content (CLIP tags · experimental)"
    matched = ", ".join(packet["matched"])
    matches = snd.get("matches") or []
    with st.expander(f"Sounds · {matched} · via {src}", expanded=True):
        st.caption(f"CLAP query: “{packet['sound_query']}”")
        if not matches:
            st.caption("No commercial-safe clips matched the filters.")
            return
        if matches[0]["score"] < 0.4:
            st.caption("Weak matches — this is the 500-clip sample; the full 8,403-clip index will improve coverage.")
        for m in matches:
            md = m["md"]
            labels = ", ".join(md.get("labels", [])[:4])
            st.markdown(f"**{m['score']:.3f}** · {labels} · `{md.get('license')}` · {md.get('duration_sec')}s")
            path = md.get("audio_path")
            if path and Path(path).exists():
                st.audio(path)
            else:
                st.caption("Audio file not available locally.")
            if md.get("requires_attribution") and md.get("attribution"):
                st.caption(f"Attribution: {md['attribution']}")

# ── example queries ───────────────────────────────────────────────────────────

# Combined examples — each fires multiple signals at once; the ranked lists
# are re-ranked client-side with RRF. Each entry maps signal name → query text.
COMBINED_EXAMPLES = [
    ("Explosion + BOOM",         {"dense": "explosion fire destruction chaos", "fts": 'search_text:(BOOM^2 OR BLAST OR "KA-BOOM")'}),
    ("Villain + escape phrase",  {"dense": "villain sinister evil grin",        "sparse": "villain escape capture"}),
    ("Detective + murder",       {"dense": "detective investigating crime scene", "sparse": "crime mystery clue evidence", "fts": "search_text:(detective AND murder)"}),
    ("Formula + exact phrase",   {"sparse": "secret formula chemical",          "fts": 'search_text:("secret formula")'}),
]

# Filter ($match_*) → dense pipeline: a native text-match filter narrows the
# candidate set server-side, then the dense vector ranks what survives.
# (field, match-operator, filter value, dense query)
FILTER_DENSE_EXAMPLES = [
    ("'formula' text → lab scene",   "search_text", "Match any", "formula chemical experiment", "scientist in a laboratory"),
    ("'space' phrase → rocket art",  "search_text", "Match phrase", "outer space",              "spaceship rocket among the stars"),
    ("'monster attack' → creature",  "search_text", "Match all", "monster attack",              "giant monster creature attacking"),
    ("'jungle' text → wild beasts",  "search_text", "Match any", "jungle wild beast",           "wild animal prowling in the jungle"),
]

# "More examples" popover: (section title, one-line caption, signal kind, items).
# kind "fts" (query_string), "fts_text" (BM25), "dense", "sparse" items are
# (label, query); "filter_dense" items are FILTER_DENSE_EXAMPLES rows;
# "combined" items are COMBINED_EXAMPLES rows.
EXAMPLE_SECTIONS = [
    ("FTS · Phrases & keywords",
     "FTS type “text”: plain BM25 relevance over search_text — no query syntax needed.",
     "fts_text", [
        ("secret formula",       "secret formula"),
        ("POW BANG ZAP",         "POW BANG ZAP"),
        ("mad scientist",        "mad scientist"),
        ("buried treasure",      "buried treasure"),
     ]),
    ("FTS · Lucene syntax",
     "FTS type “query_string”: phrase slop (~N), term boosting (^N), per-field targeting.",
     "fts", [
        ('"detective murder"~5 · slop',       'search_text:("detective murder"~5)'),
        ("explosion^2 OR fire · boost",       "search_text:(explosion^2 OR fire)"),
        ('"secret formula"~3 OR treasure^2',  'search_text:("secret formula"~3 OR treasure^2)'),
        ("ocr_text:(hero AND villain)",       "ocr_text:(hero AND villain)"),
     ]),
    ("Dense · Visual (CLIP)",
     "Finds panels by what's drawn: OpenCLIP embeds the text and matches image vectors.",
     "dense", [
        ("Hero flying",          "hero in cape flying through sky"),
        ("Fistfight",            "two men punching fighting brawl"),
        ("Elephant",             "elephant in the jungle"),
        ("Rocket in space",      "spaceship rocket outer space stars"),
     ]),
    ("Filter ($match_*) → Dense",
     "A server-side text-match filter narrows candidates, then CLIP ranks what survives.",
     "filter_dense", FILTER_DENSE_EXAMPLES),
    ("Sparse · Keyword expansion",
     "Learned keyword expansion over OCR text (pinecone-sparse-english-v0).",
     "sparse", [
        ("Secret formula",       "secret formula"),
        ("Villain escape",       "villain escape capture"),
        ("Crime mystery",        "crime mystery clue evidence"),
        ("Hidden treasure",      "hidden treasure map gold"),
     ]),
    ("Hybrid · Client-side RRF",
     "Each signal queries the index separately; the ranked lists are re-ranked client-side with Reciprocal Rank Fusion.",
     "combined", COMBINED_EXAMPLES),
]

# Header strip chips: label → (signal kind, payload), one equal-width button each.
STRIP_EXAMPLES = {
    "Hero flying":                 ("dense",        "hero in cape flying through sky"),
    "POW / BANG / ZAP":            ("fts_text",     "POW BANG ZAP"),
    "'space' phrase → rocket art": ("filter_dense", FILTER_DENSE_EXAMPLES[1][1:]),
    "Explosion + BOOM":            ("combined",     COMBINED_EXAMPLES[0][1]),
    "Secret formula":              ("sparse",       "secret formula"),
}

# ── session state ─────────────────────────────────────────────────────────────

for k, v in {
    "fts_on": True, "dense_on": False, "sparse_on": False,
    "fts_q": "*", "dense_q": "", "sparse_q": "",
    "fts_type": "query_string",
    "top_k": 20, "exclude_ads": True,
    "filter_rows": [], "filter_next_id": 0, "filter_combinator": "And",
    "show_fields": ["search_text"],
    "_similar_vec": None, "_similar_id": None,
    "_sounds": {},
    "results": None, "result_meta": {},
    "run": False,
}.items():
    st.session_state.setdefault(k, v)


def _reset_query():
    """Clear every signal toggle, query box, filter row, and the 'similar'
    vector so a freshly-clicked example or 'See more like this' starts from a
    clean slate — no stale signal or text leaks into the new search."""
    _clear_filters()
    st.session_state.fts_on    = False
    st.session_state.dense_on  = False
    st.session_state.sparse_on = False
    st.session_state.fts_q     = "*"
    st.session_state.dense_q   = ""
    st.session_state.sparse_q  = ""
    st.session_state.fts_type  = "query_string"
    st.session_state["_similar_vec"] = None
    st.session_state["_similar_id"]  = None


def _find_sounds(hit_id: str):
    """Find FSD50K sound effects for a panel: OCR onomatopoeia first, CLIP-tag
    fallback, then CLAP search over the comic-sounds index."""
    from src.sounds.search.panel_to_sound_query import panel_to_sound_query
    from src.sounds.search.search_sounds_dense import search_sounds_dense
    try:
        doc = _index().documents.fetch(ids=[hit_id], namespace=NAMESPACE).documents[hit_id]
        panel = {"ocr_text": doc.get("ocr_text"), "search_text": doc.get("search_text")}
        packet = panel_to_sound_query(panel, image_vec=doc.get("image_dense"))
        if packet is None:
            st.session_state["_sounds"][hit_id] = {"packet": None}
            return
        res = search_sounds_dense(packet["sound_query"], top_k=6,
                                  filters=packet["filters"], index=_sound_index())
        matches = [{"id": m.id, "score": m.score, "md": m.metadata or {}} for m in res.matches]
        st.session_state["_sounds"][hit_id] = {"packet": packet, "matches": matches}
    except Exception as exc:
        st.session_state["_sounds"][hit_id] = {"error": str(exc)}

def _use_similar(hit_id: str):
    try:
        result = _index().documents.fetch(ids=[hit_id], namespace=NAMESPACE)
        vec = result.documents[hit_id].get("image_dense")
        if not vec:
            raise ValueError("No dense vector stored for this record")
        _reset_query()
        st.session_state["_similar_vec"] = vec
        st.session_state["_similar_id"]  = hit_id
        st.session_state.dense_on        = True
        st.session_state.run             = True
    except Exception as exc:
        st.session_state.result_meta = {"error": f"Could not fetch vector: {exc}"}

def _use_example(query: str, signal: str):
    """on_click callback — runs before rerun so widget keys can be set safely.

    signal "fts" runs query_string (Lucene); "fts_text" runs the text type
    (plain BM25 over search_text) — _reset_query defaults back to query_string.
    """
    _reset_query()
    if signal == "dense":
        st.session_state.dense_on = True
        st.session_state.dense_q  = query
    elif signal == "sparse":
        st.session_state.sparse_on = True
        st.session_state.sparse_q  = query
    else:
        st.session_state.fts_on = True
        st.session_state.fts_q  = query
        if signal == "fts_text":
            st.session_state.fts_type = "text"
    st.session_state.run = True

def _use_combined(queries: dict):
    """on_click callback for multi-signal examples — enables each signal in the dict."""
    _reset_query()
    st.session_state.dense_on  = "dense"  in queries
    st.session_state.sparse_on = "sparse" in queries
    st.session_state.fts_on    = "fts"    in queries
    if "dense" in queries:
        st.session_state.dense_q  = queries["dense"]
    if "sparse" in queries:
        st.session_state.sparse_q = queries["sparse"]
    if "fts" in queries:
        st.session_state.fts_q    = queries["fts"]
    st.session_state.run = True

def _use_filter_dense(field: str, op: str, value: str, dense_query: str):
    """on_click callback for $match_* → dense pipeline examples.

    Replaces the filter rows with a single text-match filter and runs a
    dense-only search, so the match operator narrows candidates server-side
    and the dense vector ranks them.
    """
    _reset_query()
    fid = st.session_state.filter_next_id
    st.session_state.filter_next_id += 1
    st.session_state.filter_rows = [fid]
    st.session_state[f"fk_{fid}"] = field
    st.session_state[f"fo_{fid}"] = op
    st.session_state[f"fv_{fid}"] = value
    st.session_state.dense_on  = True
    st.session_state.dense_q   = dense_query
    st.session_state.run = True

def _use_strip(label: str):
    """on_click callback for the header example chips."""
    kind, payload = STRIP_EXAMPLES[label]
    if kind == "combined":
        _use_combined(payload)
    elif kind == "filter_dense":
        _use_filter_dense(*payload)
    else:
        _use_example(payload, kind)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero">
  <div class="hero-title">💥 Comic Panel Search</div>
  <div class="hero-tag">Search 1,229,664 golden-age comic panels by what's drawn, what's said, or both.</div>
  <div class="hero-caps">
    <span class="hero-chip" title="OpenCLIP ViT-B/16 image embeddings, queried by text description">Dense · CLIP visual</span>
    <span class="hero-chip" title="pinecone-sparse-english-v0 keyword expansion over OCR text">Sparse · keywords</span>
    <span class="hero-chip" title="Two FTS types: text (plain BM25 relevance) and query_string (Lucene: slop, boosting, per-field)">Full-text · BM25 / Lucene</span>
    <span class="hero-chip" title="Enable multiple signals; the ranked lists are re-ranked client-side with Reciprocal Rank Fusion">Hybrid · RRF</span>
    <span class="hero-meta">comic-panels · 1,229,664 panels · namespace {NAMESPACE} · COMICS dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── example strip ─────────────────────────────────────────────────────────────
# One chip per capability; always visible, even with results on screen.
# Equal-width chips plus the "More examples" trigger span the full bar.

with st.container(key="strip"):
    cols = st.columns([0.55, 1, 1, 1, 1, 1, 1], gap="small", vertical_alignment="center")
    with cols[0]:
        st.markdown('<p class="section-label">Try:</p>', unsafe_allow_html=True)
    for i, (col, label) in enumerate(zip(cols[1:], STRIP_EXAMPLES)):
        with col:
            st.button(label, key=f"strip_{i}", use_container_width=True,
                      on_click=_use_strip, args=(label,))
    with cols[6], st.popover("More examples", key="strip_more", use_container_width=True):
        for title, caption, kind, items in EXAMPLE_SECTIONS:
            st.markdown(f'<p class="section-label" style="margin:8px 0 4px">{title}</p>',
                        unsafe_allow_html=True)
            st.caption(caption)
            for item in items:
                if kind == "combined":
                    label, queries = item
                    st.button(label, key=f"ex·{label}", use_container_width=True,
                              on_click=_use_combined, args=(queries,))
                elif kind == "filter_dense":
                    label = item[0]
                    st.button(label, key=f"ex·{label}", use_container_width=True,
                              on_click=_use_filter_dense, args=item[1:])
                else:
                    label, query = item
                    st.button(label, key=f"ex·{label}", use_container_width=True,
                              on_click=_use_example, args=(query, kind))

# ── two-column layout ─────────────────────────────────────────────────────────

left, right = st.columns([1, 2.8], gap="large")

with left:
    # ── Search signals ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Search signals</p>', unsafe_allow_html=True)

    # Full-text
    fts_on = st.toggle("Full-text search", key="fts_on")
    if fts_on:
        st.selectbox(
            "FTS type", ["query_string", "text"], key="fts_type",
            label_visibility="collapsed",
            help="query_string: multi-field Lucene syntax (AND/OR/phrases)  ·  text: BM25 on a single field",
        )
        st.text_input(
            "fts_query", key="fts_q", label_visibility="collapsed",
            placeholder=(
                'search_text:("secret formula")  ·  search_text:(explosion^2 OR fire)  ·  ocr_text:(hero AND villain)'
                if st.session_state.fts_type == "query_string"
                else "plain text, ranked by BM25: secret formula"
            ),
        )

    # Dense
    dense_on = st.toggle("Dense — visual similarity", key="dense_on")
    if dense_on:
        st.text_input(
            "dense_query", key="dense_q", label_visibility="collapsed",
            placeholder="describe what you see: hero flying through sky",
            help="Embedded with OpenCLIP ViT-B/16",
        )

    # Sparse
    sparse_on = st.toggle("Sparse — keyword expansion", key="sparse_on")
    if sparse_on:
        st.text_input(
            "sparse_query", key="sparse_q", label_visibility="collapsed",
            placeholder="keywords: secret formula villain escape",
            help="Embedded with pinecone-sparse-english-v0",
        )

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────
    n_active = len(st.session_state.filter_rows)
    fhdr_label = f"Filters · {n_active}" if n_active else "Filters"
    fhdr_l, fhdr_r = st.columns([1.4, 1.6], vertical_alignment="center")
    with fhdr_l:
        st.markdown(f'<p class="section-label">{fhdr_label}</p>', unsafe_allow_html=True)
    if st.session_state.filter_rows:
        with fhdr_r:
            st.selectbox("Combine", ["And", "Or"], key="filter_combinator",
                         label_visibility="collapsed",
                         help="How to combine multiple filter conditions")

    for fid in st.session_state.filter_rows:
        k   = st.session_state.get(f"fk_{fid}", "search_text")
        op  = st.session_state.get(f"fo_{fid}", "==")
        val = st.session_state.get(f"fv_{fid}", "").strip()
        summary = f"{k} · {_OP_SHORT.get(op, op)} · {val or '…'}"
        c1, c2 = st.columns([5.6, 0.8])
        with c1:
            with st.popover(summary, key=f"fp_{fid}", use_container_width=True):
                st.selectbox("Field", ALL_FIELDS, key=f"fk_{fid}")
                st.selectbox("Operator", OPS, key=f"fo_{fid}",
                             help="Match phrase / all / any run full-text matching server-side")
                st.text_input("Value", key=f"fv_{fid}",
                              placeholder="comma-separated for In / Not In")
        with c2:
            st.button("✕", key=f"fr_{fid}",
                      on_click=_remove_filter, args=(fid,),
                      use_container_width=True)

    st.button("+ Add filter", key="add_filter",
              on_click=_add_filter, use_container_width=True)
    st.checkbox("Exclude ad pages", key="exclude_ads")

    st.divider()

    # ── Results ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)
    st.number_input("Top K", min_value=1, max_value=200, step=5, key="top_k")
    st.multiselect("Metadata to show", ALL_FIELDS, key="show_fields",
                   placeholder="Fields to display in results",
                   help="Which metadata fields appear on each result card")
    st.caption("Metadata fields shown on each result card.")

    st.write("")
    if st.button("Run query", type="primary", use_container_width=True):
        st.session_state.run = True

# ── run search ────────────────────────────────────────────────────────────────

if st.session_state.run:
    st.session_state.run = False

    fts_on    = st.session_state.fts_on
    dense_on  = st.session_state.dense_on
    sparse_on = st.session_state.sparse_on

    if not any([fts_on, dense_on, sparse_on]):
        st.session_state.result_meta = {"error": "Enable at least one search signal."}
        st.session_state.results = []
    else:
        idx     = _index()
        filters = _build_filters()
        top_k   = int(st.session_state.top_k)

        from src.search.search_dense  import search_dense
        from src.search.search_sparse import search_sparse
        from src.search.search_fts    import search_fts
        from src.search.fusion        import rrf_merge

        sim_vec = st.session_state.get("_similar_vec")
        sim_id  = st.session_state.get("_similar_id")
        if sim_vec is not None:
            st.session_state["_similar_vec"] = None
            st.session_state["_similar_id"]  = None

        groups, error = [], None
        try:
            if dense_on and (sim_vec is not None or st.session_state.dense_q.strip()):
                if sim_vec is not None:
                    vec = sim_vec
                else:
                    _load_clip()  # CLIP (torch) is only needed to embed a dense *text* query
                    from src.embeddings.embed_images import embed_text_query
                    vec = embed_text_query(st.session_state.dense_q.strip())
                r = search_dense(idx, NAMESPACE, vec, top_k=top_k, filters=filters)
                groups.append(("dense", r.get("result", {}).get("hits", [])))

            if sparse_on and st.session_state.sparse_q.strip():
                from src.embeddings.embed_sparse_text import embed_sparse_query
                svec = embed_sparse_query(st.session_state.sparse_q.strip())
                if svec:
                    r = search_sparse(idx, NAMESPACE, svec, top_k=top_k, filters=filters)
                    groups.append(("sparse", r.get("result", {}).get("hits", [])))

            if fts_on and st.session_state.fts_q.strip():
                r = search_fts(idx, NAMESPACE, st.session_state.fts_q.strip(),
                               top_k=top_k, filters=filters,
                               fts_type=st.session_state.fts_type)
                groups.append(("fts", r.get("result", {}).get("hits", [])))

        except Exception as exc:
            error = str(exc)

        if error:
            st.session_state.results     = []
            st.session_state.result_meta = {"error": error}
        elif not groups:
            st.session_state.results     = []
            st.session_state.result_meta = {"info": "No query text provided for the enabled signals."}
        else:
            merged = rrf_merge(groups)
            src    = "hybrid · RRF" if len(groups) > 1 else groups[0][0]
            if sim_id:
                src = f"similar to {sim_id} · {src}"
            st.session_state.results     = merged[:top_k]
            st.session_state.result_meta = {"source": src, "top_k": top_k}

# ── results panel ─────────────────────────────────────────────────────────────

with right:
    results = st.session_state.results
    meta    = st.session_state.result_meta

    if meta.get("error"):
        st.error(meta["error"])
    elif meta.get("info"):
        st.warning(meta["info"])
    elif results is None:
        st.markdown('<div class="ph">Pick an example above, or configure a query and click <strong>Run query</strong></div>',
                    unsafe_allow_html=True)
    elif len(results) == 0:
        st.markdown('<div class="ph">No results — try adjusting your query or enabling more signals.</div>',
                    unsafe_allow_html=True)
    else:
        src = meta.get("source", "")
        tk  = meta.get("top_k", "?")
        st.markdown(
            f'<div class="rh">Search: <strong>{len(results)} results</strong>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;top_k={tk}'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;{src}</div>',
            unsafe_allow_html=True,
        )
        show_fields = st.session_state.show_fields or SHOW_FIELDS
        for i, hit in enumerate(results, 1):
            c_card, c_sim = st.columns([10, 2])
            with c_card:
                _card(hit, i, show_fields)
            with c_sim:
                if hit.get("_id"):
                    st.button(
                        "See more like this...",
                        key=f"sim_{i}_{hit['_id']}",
                        on_click=_use_similar,
                        args=(hit["_id"],),
                        use_container_width=True,
                    )
                    # "Find sounds" button unwired for the initial release. The
                    # backend (_find_sounds / _render_sounds here, and src/sounds/*)
                    # is left intact so the feature can be re-enabled later by
                    # restoring the button and the _render_sounds call below.
