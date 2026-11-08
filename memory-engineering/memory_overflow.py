import litellm
import working_memory
import config

RECENT_KEEP  = 4
TOKEN_BUDGET = 220

def summarize_turns(turns, prev_summary=""):
    convo = "\n".join(f'{t["role"]}: {t["content"]}' for t in turns)
    system = ("You maintain a running summary of a conversation. Update the existing summary with the "
              "new turns, preserving concrete facts (names, numbers, preferences). Return only the "
              "updated summary, under 100 words.")
    user = f"Existing summary:\n{prev_summary or '(none)'}\n\nNew turns:\n{convo}\n\nUpdated summary:"
    resp = litellm.completion(model=config.LLM_MODEL, temperature=0, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}],custom_llm_provider="openai",
                api_base=config.API_BASE,
                api_key=config.API_KEY)
    return resp.choices[0].message.content.strip()

history = []
history.append({"role": "user", "content": "My name is Alex and I have a pet parrot named Kiwi."})
history.append({"role": "assistant", "content": "Nice to meet you, Alex!"})

for i in range(8):
    history.append({"role": "user", "content": f"Explain how volcanoes form, part {i}."})
    history.append({"role": "assistant", "content": "Detailed explanation. " * 12})


assert sum(working_memory.token_count(m["content"]) for m in history) > TOKEN_BUDGET, "budget was not exceeded"
old = history[:-RECENT_KEEP]
summary = summarize_turns(old)

assert "kiwi" in summary.lower(), "Early fact 'Kiwi' was lost — the summary must preserve it."
print("PASS — summarisation fired and the planted fact 'Kiwi' survived the compression.")
print("Rolling summary:", summary)