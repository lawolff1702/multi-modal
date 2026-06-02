"""Full-text search over panel OCR/dialogue text."""

import os
from pinecone import Pinecone


_INCLUDE_FIELDS = [
    "ocr_text", "search_text", "comic_id", "book_id",
    "page_id", "page_num", "panel_num", "image_path", "source", "is_ad_page",
]


def search_fts(
    index,
    namespace: str,
    query: str,
    top_k: int = 20,
    filters: dict | None = None,
    fts_type: str = "query_string",
) -> dict:
    if fts_type == "query_string":
        score_by = [{"type": "query_string", "query": query}]
    else:
        score_by = [{"type": "text", "field": "search_text", "query": query}]

    response = index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=score_by,
        filter=filters or {},
        include_fields=_INCLUDE_FIELDS,
    )
    return {"result": {"hits": [h.to_dict() for h in response.matches]}}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full-text search over comic dialogue")
    parser.add_argument("--query", required=True, help='e.g. "secret formula" or BANG or formula AND detective')
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-ads", action="store_true", default=True)
    args = parser.parse_args()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "comics-v1")

    filters = {"is_ad_page": False} if args.no_ads else {}
    response = search_fts(index, namespace, args.query, top_k=args.top_k, filters=filters)

    hits = response.get("result", {}).get("hits", [])
    print(f"Query: {args.query!r}  ({len(hits)} hits)")
    for i, hit in enumerate(hits, 1):
        fields = hit.get("fields", {})
        print(f"  {i}. [{hit.get('_score', 0):.4f}] {hit['_id']}  |  {fields.get('ocr_text', '')[:80]}")


if __name__ == "__main__":
    main()
