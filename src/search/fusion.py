"""Reciprocal Rank Fusion for merging results from multiple search signals."""


def rrf_merge(result_groups: list[tuple[str, list]], k: int = 60) -> list[dict]:
    """
    Merge ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_groups: list of (source_name, hits) tuples where each hit is a
                       dict with at least '_id'.
        k: RRF smoothing constant (default 60 per original paper).

    Returns:
        List of hit dicts sorted by descending rrf_score, with a 'sources' key
        added to each hit indicating which signals returned it.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for source_name, hits in result_groups:
        for rank, hit in enumerate(hits, start=1):
            panel_id = hit.get("_id") or hit.get("panel_id")
            if panel_id is None:
                continue

            scores[panel_id] = scores.get(panel_id, 0.0) + 1.0 / (k + rank)

            if panel_id not in payloads:
                payloads[panel_id] = dict(hit)
                payloads[panel_id]["sources"] = []

            payloads[panel_id]["sources"].append(source_name)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    output = []
    for panel_id, score in ranked:
        item = payloads[panel_id]
        item["rrf_score"] = score
        output.append(item)

    return output
