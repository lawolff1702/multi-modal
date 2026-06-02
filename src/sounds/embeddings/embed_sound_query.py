"""Embed a text query into the CLAP space for sound retrieval."""
from __future__ import annotations

from src.sounds.embeddings.clap_model import embed_text_batch


def embed_sound_query(query: str) -> list[float]:
    """Return the 512-dim L2-normalized CLAP text vector for `query`."""
    return embed_text_batch([query])[0]
