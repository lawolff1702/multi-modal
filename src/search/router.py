"""Route a query to the appropriate search signals."""


_EXACT_INTENT_PHRASES = {
    "says", "where they say", "exact", "phrase", "quote",
    "dialogue", "sound effect", "contains",
}


def route_query(query: str) -> dict[str, bool]:
    """
    Decide which search signals to run for a given query string.

    Returns a dict with keys: dense, sparse, fts — each True or False.

    Routing logic:
      - dense is always True (visual/semantic search runs for every query)
      - sparse is True for any non-empty query (keyword overlap)
      - fts is True when the query has quoted phrases, Boolean operators,
        all-caps sound effects, or explicit exact-match intent language
    """
    q = query.strip()
    if not q:
        return {"dense": False, "sparse": False, "fts": False}

    tokens = q.split()
    has_quotes = '"' in q or "'" in q
    has_boolean = any(t.upper() in {"AND", "OR", "NOT"} for t in tokens)
    has_all_caps = any(t.isupper() and len(t) >= 3 for t in tokens)
    has_exact_intent = any(phrase in q.lower() for phrase in _EXACT_INTENT_PHRASES)

    return {
        "dense": True,
        "sparse": True,
        "fts": has_quotes or has_boolean or has_all_caps or has_exact_intent,
    }
