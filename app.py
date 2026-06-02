"""Comic panel search UI."""

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Comic Panel Search",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  #MainMenu, footer, [data-testid="collapsedControl"] { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

  /* result card */
  .rc {
    background: #fff;
    border: 1px solid #e5e5e5;
    border-radius: 2px;
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
    background: rgba(0,43,255,0.05); border-radius: 2px;
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
    color: #002BFF; border-radius: 2px;
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
  .rc-img {
    max-width: 100%; max-height: 65vh;
    width: auto; height: auto;
    border-radius: 2px; display: block;
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
    color: #737373; margin: 0;
  }

  /* disable typing in all selectboxes — click still opens the dropdown */
  div[data-testid="stSelectbox"] input { pointer-events: none; }

  /* uniform 3px radius across all native Streamlit components */
  button, input, textarea, select,
  [data-testid="stTextInput"] > div,
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stNumberInput"] > div,
  [data-testid="stMultiSelect"] > div,
  [data-testid="stExpander"],
  [data-testid="stAlert"] { border-radius: 2px !important; }
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
ALL_FIELDS = ["comic_id", "book_id", "page_num", "panel_num", "source", "ocr_text", "image_path", "is_ad_page"]

@st.cache_resource(show_spinner="Connecting to Pinecone…")
def _index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))

@st.cache_resource(show_spinner="Loading CLIP model…")
def _load_clip():
    from src.embeddings.embed_images import _init_model
    _init_model()

# ── helpers ───────────────────────────────────────────────────────────────────

def _b64(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        img = Image.open(p).convert("RGB")
        scale = min(1.0, 900 / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

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

def _remove_filter(fid: int):
    st.session_state.filter_rows = [r for r in st.session_state.filter_rows if r != fid]

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
    if b64 := _b64(hit.get("image_path", "")):
        img_html = f'<img class="rc-img" src="data:image/jpeg;base64,{b64}"/>'

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

# ── example queries ───────────────────────────────────────────────────────────

EXAMPLES = {
    "🎨 Visual (Dense)": [
        ("Hero flying",             "hero in cape flying through sky"),
        ("Fistfight",               "two men punching fighting brawl"),
        ("Explosion",               "explosion fire destruction chaos"),
        ("Scientist in lab",        "scientist laboratory chemicals"),
        ("Rocket in space",         "spaceship rocket outer space stars"),
        ("Villain menacing",        "villain sinister evil grin"),
        ("Romance",                 "couple romance embrace kissing"),
        ("Detective",               "detective investigating crime scene"),
        ("Car chase",               "car chase street pursuit"),
        ("Monster horror",          "monster creature attacking horror"),
        ("Woman captured",          "woman captured tied up danger"),
        ("Man with gun",            "man pointing gun threatening"),
    ],
    "🔤 Keyword (Sparse)": [
        ("Secret formula",          "secret formula"),
        ("Help / danger",           "help me danger"),
        ("Villain escape",          "villain escape capture"),
        ("Mysterious stranger",     "mysterious stranger"),
        ("Great power",             "great power"),
        ("Bank robbery",            "robbery bank heist money"),
        ("Crime mystery",           "crime mystery clue evidence"),
        ("Space adventure",         "space adventure rocket planet"),
    ],
    "💥 Sound effects (FTS)": [
        ("POW / BANG / ZAP",        "BANG OR POW OR ZAP"),
        ("KAPOW / WHAM / CRASH",    "KAPOW OR WHAM OR CRASH"),
        ("BOOM / BLAST",            "BOOM OR BLAST OR KA-BOOM"),
        ("THUD / SMASH",            "THUD OR CRUNCH OR SMASH"),
    ],
    "💬 Phrases (FTS)": [
        ('"secret formula"',        '"secret formula"'),
        ('"great danger"',          '"great danger"'),
        ("detective AND murder",    "detective AND murder"),
        ("villain AND escape",      "villain AND escape"),
        ("danger AND rescue",       "danger AND rescue"),
    ],
}

# ── session state ─────────────────────────────────────────────────────────────

for k, v in {
    "fts_on": True, "dense_on": False, "sparse_on": False,
    "fts_q": "*", "dense_q": "", "sparse_q": "",
    "fts_type": "query_string",
    "top_k": 20, "exclude_ads": True,
    "filter_rows": [], "filter_next_id": 0, "filter_combinator": "And",
    "show_fields": ["ocr_text"],
    "_similar_vec": None, "_similar_id": None,
    "results": None, "result_meta": {},
    "run": False,
}.items():
    st.session_state.setdefault(k, v)


def _use_similar(hit_id: str):
    try:
        result = _index().documents.fetch(ids=[hit_id], namespace=NAMESPACE)
        vec = result.documents[hit_id].get("image_dense")
        if not vec:
            raise ValueError("No dense vector stored for this record")
        st.session_state["_similar_vec"] = vec
        st.session_state["_similar_id"]  = hit_id
        st.session_state.dense_on        = True
        st.session_state.run             = True
    except Exception as exc:
        st.session_state.result_meta = {"error": f"Could not fetch vector: {exc}"}

def _use_example(query: str, signal: str):
    """on_click callback — runs before rerun so widget keys can be set safely."""
    if signal == "dense":
        st.session_state.dense_on = True
        st.session_state.dense_q  = query
    elif signal == "sparse":
        st.session_state.sparse_on = True
        st.session_state.sparse_q  = query
    else:
        st.session_state.fts_on = True
        st.session_state.fts_q  = query
    st.session_state.run = True

# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## Browse your index")
st.caption(f"comic-panels · 1,229,664 panels · namespace: {NAMESPACE}")

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
            placeholder='"phrase" OR keyword  ·  BANG OR POW  ·  field AND value',
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
    fhdr_l, fhdr_r = st.columns([1, 2])
    with fhdr_l:
        st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)
    if st.session_state.filter_rows:
        with fhdr_r:
            st.selectbox("Combine", ["And", "Or"], key="filter_combinator",
                         label_visibility="collapsed",
                         help="How to combine multiple filter conditions")

    for fid in st.session_state.filter_rows:
        op  = st.session_state.get(f"fo_{fid}", "==")
        op_w = max(2.0, min(3.2, len(op) * 0.25))
        fv_w = (6.4 - op_w - 0.4) / 2
        c1, c2, c3, c4 = st.columns([fv_w, op_w, fv_w, 0.4])
        with c1:
            st.text_input("Key", key=f"fk_{fid}",
                          label_visibility="collapsed", placeholder="field")
        with c2:
            st.selectbox("Op", OPS, key=f"fo_{fid}", label_visibility="collapsed")
        with c3:
            st.text_input("Value", key=f"fv_{fid}",
                          label_visibility="collapsed", placeholder="value")
        with c4:
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
    st.multiselect("Fields", ALL_FIELDS, key="show_fields",
                   label_visibility="collapsed",
                   placeholder="Fields to display in results")

    st.write("")
    if st.button("Run query", type="primary", use_container_width=True):
        st.session_state.run = True

    st.divider()

    # ── Examples ──────────────────────────────────────────────────────────
    with st.expander("💡 Example queries"):
        for category, items in EXAMPLES.items():
            st.markdown(f'<p class="section-label" style="margin:8px 0 4px">{category}</p>',
                        unsafe_allow_html=True)
            for label, query in items:
                if "Visual" in category:
                    sig = "dense"
                elif "Keyword" in category:
                    sig = "sparse"
                else:
                    sig = "fts"
                st.button(label, key=f"ex·{label}", use_container_width=True,
                          on_click=_use_example, args=(query, sig))

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
        _load_clip()
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
        st.markdown('<div class="ph">Configure your query and click <strong>Run query</strong></div>',
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
                    )
