import reciprocal_rank_fusion
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(chunks, query, cand_ids, k=5):
    pairs = [(query, chunks[cid]["text"]) for cid in cand_ids]
    scores = reranker.predict(pairs)
    order = sorted(range(len(cand_ids)), key=lambda i: scores[i], reverse=True)
    return [cand_ids[i] for i in order[:k]]


def retrieve(bm25: BM25Okapi, chunks, query, k=5, pool=10):
    # Full pipeline: hybrid candidate pool, then cross-encoder rerank.
    return rerank(
        chunks,
        query,
        reciprocal_rank_fusion.hybrid_search(bm25, query, k=pool, pool=pool),
        k=k,
    )
