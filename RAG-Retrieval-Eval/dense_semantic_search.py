import qdrant, embed, chunking
from qdrant_client.models import Filter, FieldCondition, MatchValue


def dense_search(query, k=10, doc_type=None):
    client = qdrant.CLIENT
    qfilter = (
        Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))])
        if doc_type
        else None
    )
    hits = client.query_points(
        collection_name=qdrant.COLLECTION,
        query=embed.embed_query(query),
        query_filter=qfilter,
        limit=k,
    ).points
    return [h.id for h in hits]


def show(ids, n=3):
    for cid in ids[:n]:
        c = chunking.CHUNKS[cid]
        print(f"  [{cid}] ({c['source']}/{c['doc_type']}) {c['text'][:130].strip()}")
