# usage: python3 spec_bench.py PORT LABEL -> sends fixed chat prompts at temp 0, prints decode tok/s and draft acceptance
import json, sys, urllib.request

PROMPTS = [
    ("prose", "Explain why the sky is blue in three short paragraphs."),
    ("prose", "Write a short history of the printing press."),
    ("prose", "What are the main causes of inflation? Answer briefly."),
    ("code",  "Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests."),
    ("copy",  "Rewrite this file in full with the function `healthy` renamed to `is_healthy` (call sites too). "
              "Output only the code.\n\n```python\n" + "".join(open(__file__.rsplit("/", 2)[0] + "/ui/server.py").readlines()) + "```"),
]

port, label = sys.argv[1], sys.argv[2]
for kind, p in PROMPTS:
    req = {"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 256, "seed": 1}
    if kind == "copy":  # thinking off, long reply
        req.update(max_tokens=1600, chat_template_kwargs={"reasoning_effort": "none"}, logit_bias=[[200008, False]])
    body = json.dumps(req).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    t = r.get("timings", {})
    with open(f"/private/tmp/claude-501/-Users-kushjuvekar/f2b09734-e90c-4999-87d2-b0619a1a7cdd/scratchpad/out_{label}_{kind}.txt", "a") as f:
        f.write(r["choices"][0]["message"].get("reasoning_content", "") + "\n=====\n" + r["choices"][0]["message"].get("content", "") + "\n#####\n")
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    acc = f"{da}/{dn} ({100*da/dn:.0f}%)" if dn else "-"
    print(f"{label:22s} {kind:5s} {t.get('predicted_per_second', 0):6.2f} tok/s  n={t.get('predicted_n')}  accepted {acc}")
