PAPER_CHUNKS = 60
FILING_SKIP = 4
FILING_CHUNKS = 40
CSV_ROWS = 80

CHUNKS = []


def add_chunks(texts, source, doc_type):
    for t in texts:
        t = t.strip()
        if t:
            CHUNKS.append(
                {"cid": len(CHUNKS), "text": t, "source": source, "doc_type": doc_type}
            )
