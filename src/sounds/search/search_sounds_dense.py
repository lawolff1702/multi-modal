"""Dense sound search against the CLAP sound index.

Always applies the commercial-safe license filter. Pass an existing `index`
(e.g. a Streamlit-cached `pc.Index(...)`) to avoid reconnecting per call.
"""
from __future__ import annotations

import os

from src.sounds import config
from src.sounds.embeddings.embed_sound_query import embed_sound_query


def search_sounds_dense(query: str, top_k: int = 10, filters: dict | None = None, index=None):
    if index is None:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(config.PINECONE_SOUND_INDEX_NAME)

    vector = embed_sound_query(query)

    base_filter = {"license_ok_for_commercial": {"$eq": True}}
    final_filter = {"$and": [base_filter, filters]} if filters else base_filter

    return index.query(
        namespace=config.PINECONE_SOUND_NAMESPACE,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=final_filter,
    )
