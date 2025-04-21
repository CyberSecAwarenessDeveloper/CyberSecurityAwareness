# test_ollama.py  ─────────────────────────────────────────────
import json, requests, sys, itertools

resp = requests.post(
    "http://localhost:11434/api/chat",
    json = {
        "model"   : "deepseek-r1:7b",    # <- use the tag you pulled
        "stream"  : True,
        "messages": [{"role":"user","content":"ping"}]
    },
    stream=True, timeout=180            # keep‑alive 3 min
)
resp.raise_for_status()

for line in resp.iter_lines():
    if not line or not line.startswith(b"data:"):
        continue
    data = json.loads(line[5:])
    if data.get("done"):
        break
    sys.stdout.write(data["message"]["content"])
    sys.stdout.flush()
