"""Turn a panel's CLIP image vector into sound-relevant words — no LLM.

CLIP and CLAP live in different vector spaces (a CLIP image vector queried
against CLAP audio returns noise — verified). The only bridge between them is
text. So we use CLIP *within its own space*: score the panel image against the
FSD50K class names (CLIP image vs CLIP text), take the top labels, and hand
those plain words to the CLAP text encoder for the actual sound search.

The CLIP-text vectors for the ~200 FSD50K classes are computed once and cached
to parquet.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.sounds import config

_CACHE = config.PROCESSED_DIR / "clip_label_vectors.parquet"
_labels: list[str] | None = None
_clean: list[str] | None = None
_matrix: np.ndarray | None = None


def _clean_label(raw: str) -> str:
    """'Bathtub_(filling_or_washing)' -> 'bathtub filling or washing'."""
    s = raw.replace("_", " ").replace("(", " ").replace(")", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _load_vocabulary() -> list[tuple[str, str]]:
    """[(raw_label, clean_label)] from FSD50K vocabulary.csv (idx,label,mid)."""
    path = hf_hub_download(config.SOUND_DATASET_NAME, "labels/vocabulary.csv", repo_type="dataset")
    df = pd.read_csv(path, header=None, names=["idx", "label", "mid"])
    return [(r.label, _clean_label(r.label)) for r in df.itertuples(index=False)]


def _build_cache() -> pd.DataFrame:
    from src.embeddings.embed_images import embed_text_query  # OpenCLIP text encoder

    vocab = _load_vocabulary()
    rows = []
    for raw, clean in vocab:
        rows.append({"label": raw, "clean": clean, "vec": embed_text_query(clean)})
    df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE, index=False)
    return df


def _ensure_loaded() -> None:
    global _labels, _clean, _matrix
    if _matrix is not None:
        return
    df = pd.read_parquet(_CACHE) if _CACHE.exists() else _build_cache()
    _labels = df["label"].tolist()
    _clean = df["clean"].tolist()
    _matrix = np.array(df["vec"].tolist(), dtype=np.float32)  # (N, 512) CLIP-text, L2-normed


def tag_image_vector(
    clip_image_vec,
    top_k: int = 3,
    min_score: float = 0.25,
    margin: float = 0.015,
    band: float = 0.02,
) -> list[str]:
    """Return cleaned FSD50K class phrases the image confidently looks like.

    `clip_image_vec` is the panel's stored OpenCLIP `image_dense` vector. CLIP
    cosines against label text are compressed (~0.24 even for a best guess), and
    on ambiguous images the top labels are near-tied — that's CLIP guessing, not
    recognizing. So we gate on confidence, not just an absolute floor:

      * top label must clear `min_score`, AND
      * top label must beat the 4th-ranked label by `margin` (a clear leader,
        not a near-tie among indifferent labels),
      * then return only labels within `band` of the top (drops weak siblings).

    Returns [] when the image has no confident sound-relevant content, so the
    caller can gate. (Vintage B/W comic art is hard for CLIP — expect many [].)
    """
    _ensure_loaded()
    v = np.asarray(clip_image_vec, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = _matrix @ v
    order = np.argsort(-sims)
    top = float(sims[order[0]])
    ref = float(sims[order[min(3, len(order) - 1)]])  # 4th-ranked
    if top < min_score or (top - ref) < margin:
        return []
    return [_clean[i] for i in order[:top_k] if sims[i] >= top - band]
