import chunking


def rank_of(ids, needle):
    for r, cid in enumerate(ids, start=1):
        if needle in chunking.CHUNKS[cid]["text"]:
            return f"rank {r}"
    return "NOT in top-5"
