import tiktoken

def token_count(text: str) -> int:
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)

working_memory = [
    {"role": "user",      "content": "I'm planning a work trip to Delhi next week."},
    {"role": "assistant", "content": "Sure — shall I check the travel policy and your seat preference first?"},
    {"role": "user",      "content": "Yes please, and book me an aisle seat."},
]
used = sum(token_count(m["content"]) for m in working_memory)
print(f"Working memory: {len(working_memory)} turns, ~{used} tokens sitting in the context window.")
print("It's transient — end the session and it's gone, unless a fact graduates to long-term memory.")