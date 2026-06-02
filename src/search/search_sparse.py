"""Sparse vector search over panel OCR/dialogue text."""

import os
from pinecone import Pinecone


_INCLUDE_FIELDS = [
    "ocr_text", "search_text", "comic_id", "book_id",
    "page_id", "page_num", "panel_num", "image_path", "source", "is_ad_page",
]


def search_sparse(index, namespace: str, query_sparse: dict, top_k: int = 20, filters: dict | None = None) -> dict:
    response = index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[{"type": "sparse_vector", "field": "text_sparse", "sparse_values": query_sparse}],
        filter=filters or {},
        include_fields=_INCLUDE_FIELDS,
    )
    return {"result": {"hits": [h.to_dict() for h in response.matches]}}


def main():
    import argparse
    from src.embeddings.embed_sparse_text import embed_sparse_query

    parser = argparse.ArgumentParser(description="Sparse text search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-ads", action="store_true", default=True)
    args = parser.parse_args()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "comics-v1")

    query_sparse = embed_sparse_query(args.query)
    if not query_sparse:
        print("Empty query — no sparse vector produced")
        return

    filters = {"is_ad_page": False} if args.no_ads else {}
    response = search_sparse(index, namespace, query_sparse, top_k=args.top_k, filters=filters)

    hits = response.get("result", {}).get("hits", [])
    print(f"Query: {args.query!r}  ({len(hits)} hits)")
    for i, hit in enumerate(hits, 1):
        fields = hit.get("fields", {})
        print(f"  {i}. [{hit.get('_score', 0):.4f}] {hit['_id']}  |  {fields.get('ocr_text', '')[:80]}")


if __name__ == "__main__":
    main()
