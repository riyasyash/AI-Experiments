import os, hashlib, json, time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_base = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)  # reads GOOGLE_API_KEY


CACHE_DIR = ".embcache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _ckey(text):
    return hashlib.sha1(("gemini-embedding-001::" + text).encode("utf-8")).hexdigest()


def _cget(text):
    p = os.path.join(CACHE_DIR, _ckey(text) + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _cput(text, vec):
    with open(os.path.join(CACHE_DIR, _ckey(text) + ".json"), "w") as f:
        json.dump(vec, f)


def _with_backoff(fn):
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(2**attempt)  # 1,2,4,8s backoff on rate limits
            else:
                raise


BATCH, PAUSE = 90, 60  # <=100 requests/min: embed 90 texts, then pause 60s


def embed_texts(texts):
    # Embed many texts; cache hits cost nothing, misses are embedded in throttled batches.
    out, todo = [None] * len(texts), []
    for i, t in enumerate(texts):
        v = _cget(t)
        out[i] = v
        if v is None:
            todo.append(i)
    if todo:
        print(
            f"  {len(todo)} new chunks to embed ({len(texts) - len(todo)} served from ./{CACHE_DIR})."
        )
    for b in range(0, len(todo), BATCH):
        idx = todo[b : b + BATCH]
        vecs = _with_backoff(lambda: _base.embed_documents([texts[i] for i in idx]))
        for i, v in zip(idx, vecs):
            out[i] = v
            _cput(texts[i], v)
        if b + BATCH < len(todo):
            print(
                f"  ...embedded {b + len(idx)}/{len(todo)}; pausing {PAUSE}s (rate limit)"
            )
            time.sleep(PAUSE)
    return out


def embed_query(text):
    v = _cget(text)
    if v is None:
        v = _with_backoff(lambda: _base.embed_query(text))
        _cput(text, v)
    return v
