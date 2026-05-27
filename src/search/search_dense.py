"""Dense vector search over panel image embeddings."""

import os
from pinecone import Pinecone


_INCLUDE_FIELDS = [
    "ocr_text", "search_text", "comic_id", "book_id",
    "page_id", "page_num", "panel_num", "image_path", "source", "is_ad_page",
]


def search_dense(index, namespace: str, query_vector: list[float], top_k: int = 20, filters: dict | None = None) -> dict:
    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "image_dense", "values": query_vector}],
        filter=filters or {},
        include_fields=_INCLUDE_FIELDS,
    )


def search_dense_with_phrase_filter(
    index,
    namespace: str,
    query_vector: list[float],
    phrase: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> dict:
    merged_filter = dict(filters) if filters else {}
    merged_filter["search_text"] = {"$match_phrase": phrase}
    return index.documents.search(
        namespace=namespace,
        top_k=top_k,
        score_by=[{"type": "dense_vector", "field": "image_dense", "values": query_vector}],
        filter=merged_filter,
        include_fields=_INCLUDE_FIELDS,
    )


def main():
    import argparse
    from src.embeddings.embed_images import embed_text_query

    parser = argparse.ArgumentParser(description="Dense image search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-ads", action="store_true", default=True)
    args = parser.parse_args()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "comics-v1")

    query_vector = embed_text_query(args.query)
    filters = {"is_ad_page": False} if args.no_ads else {}
    response = search_dense(index, namespace, query_vector, top_k=args.top_k, filters=filters)

    hits = response.get("result", {}).get("hits", [])
    print(f"Query: {args.query!r}  ({len(hits)} hits)")
    for i, hit in enumerate(hits, 1):
        fields = hit.get("fields", {})
        print(f"  {i}. [{hit.get('_score', 0):.4f}] {hit['_id']}  |  {fields.get('ocr_text', '')[:80]}")


if __name__ == "__main__":
    main()
