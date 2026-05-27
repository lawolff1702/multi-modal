"""Combined search: run dense, sparse, and FTS in parallel and merge with RRF."""

import os
from pinecone import Pinecone

from src.search.search_dense import search_dense
from src.search.search_sparse import search_sparse
from src.search.search_fts import search_fts
from src.search.fusion import rrf_merge


def search_combined(
    index,
    namespace: str,
    query_text: str,
    dense_query_vector: list[float] | None = None,
    sparse_query_vector: dict | None = None,
    top_k: int = 20,
    filters: dict | None = None,
    run_fts: bool = True,
) -> list[dict]:
    result_groups = []

    if dense_query_vector is not None:
        dense_response = search_dense(index, namespace, dense_query_vector, top_k=top_k, filters=filters)
        result_groups.append(("dense", dense_response.get("result", {}).get("hits", [])))

    if sparse_query_vector is not None:
        sparse_response = search_sparse(index, namespace, sparse_query_vector, top_k=top_k, filters=filters)
        result_groups.append(("sparse", sparse_response.get("result", {}).get("hits", [])))

    if run_fts and query_text.strip():
        fts_response = search_fts(index, namespace, query_text, top_k=top_k, filters=filters)
        result_groups.append(("fts", fts_response.get("result", {}).get("hits", [])))

    if not result_groups:
        return []

    return rrf_merge(result_groups)[:top_k]


def main():
    import argparse
    from src.search.router import route_query
    from src.embeddings.embed_images import embed_text_query
    from src.embeddings.embed_sparse_text import embed_sparse_query

    parser = argparse.ArgumentParser(description="Combined dense+sparse+FTS search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-ads", action="store_true", default=True)
    args = parser.parse_args()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.preview.index(name=os.environ.get("PINECONE_INDEX_NAME", "comic-panels"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "comics-v1")

    route = route_query(args.query)
    dense_vector = embed_text_query(args.query) if route["dense"] else None
    sparse_vector = embed_sparse_query(args.query) if route["sparse"] else None
    filters = {"is_ad_page": False} if args.no_ads else {}

    hits = search_combined(
        index=index,
        namespace=namespace,
        query_text=args.query,
        dense_query_vector=dense_vector,
        sparse_query_vector=sparse_vector,
        top_k=args.top_k,
        filters=filters,
        run_fts=route["fts"],
    )

    print(f"Query: {args.query!r}  route={route}  ({len(hits)} hits)")
    for i, hit in enumerate(hits, 1):
        fields = hit.get("fields", hit)
        sources = hit.get("sources", [])
        print(f"  {i}. [rrf={hit.get('rrf_score', 0):.4f} src={sources}] {hit.get('_id')}  |  {fields.get('ocr_text', '')[:80]}")


if __name__ == "__main__":
    main()
