import dense_semantic_search
from rank_bm25 import BM25Okapi


def bm25_search(bm25: BM25Okapi, query, k=10):
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def rrf_fuse(rankings, k=10, c=60):
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def hybrid_search(bm25: BM25Okapi, query, k=10, pool=10):
    return rrf_fuse(
        [
            dense_semantic_search.dense_search(query, pool),
            bm25_search(bm25, query, pool),
        ],
        k=k,
    )
