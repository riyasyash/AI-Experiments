import dense_semantic_search, reciprocal_rank_fusion, cross_encoder_reranking
import numpy as np

GOLDEN = [
    # --- exact event id embedded in a natural question ---
    {
        "q": "What was the magnitude of the earthquake with event id us7000t1bu?",
        "answer_contains": "us7000t1bu",
    },
    {
        "q": "How deep was the earthquake recorded as event us6000t8pa?",
        "answer_contains": "us6000t8pa",
    },
    {
        "q": "Where did the earthquake with event id us7000szpb occur?",
        "answer_contains": "us7000szpb",
    },
    {
        "q": "What was the magnitude of event us6000t7zp?",
        "answer_contains": "us6000t7zp",
    },
    {
        "q": "What is the depth of the earthquake with id us6000t9r5?",
        "answer_contains": "us6000t9r5",
    },
    {
        "q": "At what magnitude was event us7000t0d0 recorded?",
        "answer_contains": "us7000t0d0",
    },
    {
        "q": "What time did the earthquake with id us6000t7zq occur?",
        "answer_contains": "us6000t7zq",
    },
    {
        "q": "Which earthquake was logged under event id us7000t01n?",
        "answer_contains": "us7000t01n",
    },
    # --- unique place name (proper noun) ---
    {
        "q": "What was the magnitude of the earthquake near Yumare, Venezuela?",
        "answer_contains": "Yumare",
    },
    {
        "q": "How strong was the earthquake near Jurm, Afghanistan?",
        "answer_contains": "Jurm",
    },
    {
        "q": "What was the depth of the earthquake near Tobelo, Indonesia?",
        "answer_contains": "Tobelo",
    },
    {
        "q": "What magnitude was the earthquake near Balangonan, Philippines?",
        "answer_contains": "Balangonan",
    },
]


def gold_chunks(chunks, answer_contains):
    key = answer_contains.lower()
    return {c["cid"] for c in chunks if key in c["text"].lower()}


def precision_at_k(retrieved, gold, k):
    topk = retrieved[:k]
    return sum(1 for s in topk if s in gold) / k


def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def mrr(retrieved, gold):
    for i, s in enumerate(retrieved, start=1):
        if s in gold:
            return 1.0 / i
    return 0.0


def evaluate(bm25, chunks, retrieve_fn, k=5):
    rows = []
    for g in GOLDEN:
        gold = gold_chunks(chunks, g["answer_contains"])
        got = retrieve_fn(bm25, chunks, g["q"], k=k)  # ordered chunk ids
        rows.append(
            (precision_at_k(got, gold, k), recall_at_k(got, gold, k), mrr(got, gold))
        )
    p, r, m = (float(np.mean(x)) for x in zip(*rows))
    return {"precision@k": p, "recall@k": r, "mrr": m}


def v_dense(bm25, chunks, q, k=5):
    return dense_semantic_search.dense_search(q, k=k)  # semantic only


def v_fused(bm25, chunks, q, k=5):
    return reciprocal_rank_fusion.hybrid_search(
        bm25, q, k=k
    )  # dense + BM25, fused (no rerank)


def v_full(bm25, chunks, q, k=5):
    return cross_encoder_reranking.retrieve(
        bm25, chunks, q, k=k
    )  # hybrid + cross-encoder rerank
