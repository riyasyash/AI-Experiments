import truststore
truststore.inject_into_ssl()

import supermemory, time, os, config, truststore
mem = supermemory.Supermemory(api_key=config.SUPERMEMORY_API_KEY)
def remember(text, kind, **extra):
    return mem.add(config.USER_ID, content=text, metadata={"type": kind, **extra})

def recall(query, client=None, k=3):
    client = client or mem

    response = client.search(
        config.USER_ID,
        query=query,
        search_mode="hybrid",
        limit=k,
    )

    out = []

    for r in response.results or []:
        metadata = r.metadata or {}
        similarity = float(r.similarity or 0.0)

        if r.memory:
            out.append((
                metadata.get("type", "memory"),
                r.memory,
                similarity,
            ))

        elif r.chunk:
            out.append((
                metadata.get("type", "document"),
                r.chunk,
                similarity,
            ))

    return out

# Episodic — a specific, timestamped event ("what happened").
remember("On 2026-08-01, Alex booked flight AI-302 from Bangalore to Delhi, seat 14A, for a work trip.", "episodic", date="2026-08-01")

# Semantic — stable facts / preferences ("what is true about the user").
remember("Alex is a data scientist. Alex prefers an aisle seat and is allergic to peanuts.", "semantic")

# Procedural — how to carry out a recurring task ("how to").
remember("Travel-booking procedure: (1) check the quarterly budget freeze, (2) use the approved corporate vendor, (3) economy class for flights under 6 hours, (4) submit expenses within 30 days.", "procedural")

# The store extracts memories asynchronously — poll until they're searchable (up to ~60s) rather than guessing a fixed delay. (On a re-run this returns instantly, since the memories already exist.)
print("Waiting for the store to extract the memories (usually a few seconds)...")

for _ in range(20):
    if mem.search(config.USER_ID, query="flight", search_mode="memories", limit=1).results:
        break
    time.sleep(3)

print("Stored 3 long-term memories (episodic, semantic, procedural) under namespace:", config.USER_ID)

print("Episodic recall — a specific past event:")

for kind, text, score in recall("what flight did I book for the Delhi trip?", k=2):
    print(f"  {score:.3f}  [{kind}]  {text}")

print("Semantic recall — stable facts / preferences:")
for kind, text, score in recall("what are the user's seating preferences and allergies?", k=1):
    print(f"  [{kind}]  {text}")

print("Procedural recall — how to do the task:")
for kind, text, score in recall("how should I book a work trip?", k=1):
    print(f"  [{kind}]  {text}")