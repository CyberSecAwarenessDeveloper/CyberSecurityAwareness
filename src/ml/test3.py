import requests, json, time

print("→ contacting Ollama …")
t  = time.time()
r  = requests.post(
        "http://localhost:11434/api/chat",
        json = {
            "model"   : "deepseek-r1:7b",
            "stream"  : True,
            "messages": [{"role":"user","content":"Ping"}]
        },
        stream  = True,
        timeout = (10, 600)          # 10 s connect • 600 s read
     )

print("HTTP", r.status_code)
token = None
for ln in r.iter_lines():
    if ln and ln.startswith(b"data:"):
        token = json.loads(ln[5:])["message"]["content"]
        print("first token after",
              round(time.time()-t,1), "s →", token[:60])
        break
print("TOKEN:", bool(token))
