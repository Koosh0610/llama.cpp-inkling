# Side-by-side chat UI: serves index.html and streams chat completions from two llama-servers.
# usage: python3 server.py  (expects the engines on the ports below, see start.sh)
import http.server
import json
import os
import socketserver
import subprocess
import threading
import time
import urllib.request

HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/inkling2-mini/mainline-Inkling-2-Mini-UD-Q3_K_XL-f16ctrl.gguf"
ENGINES = {
    "stock": {"port": 18101, "label": "llama.cpp (upstream)", "bin": f"{HOME}/src/llama-stock/build/bin/llama-server"},
    "ours":  {"port": 18102, "label": "ours (M5 kernels)",    "bin": f"{HOME}/src/llama.cpp/build/bin/llama-server"},
}
UI_PORT = 8090
THINKING_TOKEN = 200008  # <|content_thinking|>
HERE = os.path.dirname(os.path.abspath(__file__))

# both engines do not fit in the GPU working set at once (24 GB Mac), so only one runs at a time
engine_lock = threading.Lock()
running = {"name": None, "proc": None}


def healthy(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except Exception:
        return False


def ensure_engine(name):
    """Stops the other engine and starts this one; returns the load time in seconds."""
    if running["name"] == name and running["proc"] and running["proc"].poll() is None and healthy(ENGINES[name]["port"]):
        return 0.0
    t0 = time.time()
    if running["proc"] and running["proc"].poll() is None:
        running["proc"].terminate()
        running["proc"].wait(timeout=30)
    e = ENGINES[name]
    log = open(os.path.join(HERE, "logs", f"{name}.log"), "w")
    running["proc"] = subprocess.Popen(
        [e["bin"], "-m", MODEL, "-fa", "on", "-ngl", "99", "-c", "8192", "--no-webui", "--port", str(e["port"])],
        stdout=log, stderr=subprocess.STDOUT)
    running["name"] = name
    for _ in range(600):
        if running["proc"].poll() is not None:
            raise RuntimeError(f"{name} engine exited, see ui/logs/{name}.log")
        if healthy(e["port"]):
            return time.time() - t0
        time.sleep(0.25)
    raise RuntimeError(f"{name} engine did not become ready")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            status = {k: ("loaded" if running["name"] == k and healthy(v["port"]) else "on demand") for k, v in ENGINES.items()}
            self.send_json(200, {"engines": {k: {"label": v["label"], "status": status[k]} for k, v in ENGINES.items()}})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        name = self.path.rsplit("/", 1)[-1]
        if not self.path.startswith("/api/chat/") or name not in ENGINES:
            self.send_json(404, {"error": "unknown engine"})
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        body = json.dumps({
            "messages": req.get("messages", []),
            "max_tokens": int(req.get("max_tokens", 512)),
            "temperature": float(req.get("temperature", 0)),
            "stream": True,
            "cache_prompt": False,
            "chat_template_kwargs": {"reasoning_effort": "high" if req.get("reasoning") else "none"},
            # effort 0 is only a hint to this model; banning the thinking marker makes it answer directly
            "logit_bias": [] if req.get("reasoning") else [[THINKING_TOKEN, False]],
        }).encode()
        with engine_lock:
            try:
                load_s = ensure_engine(name)
                upstream = urllib.request.Request(
                    f"http://127.0.0.1:{ENGINES[name]['port']}/v1/chat/completions", body, {"Content-Type": "application/json"})
                resp = urllib.request.urlopen(upstream, timeout=900)
            except Exception as e:
                self.send_json(502, {"error": f"{name} engine unavailable: {e}"})
                return
            self.stream(resp, load_s)

    def stream(self, resp, load_s):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Engine-Load-Seconds", f"{load_s:.2f}")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for line in resp:
                chunk = line
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"UI on http://127.0.0.1:{UI_PORT}")
    try:
        Server(("127.0.0.1", UI_PORT), Handler).serve_forever()
    finally:
        if running["proc"] and running["proc"].poll() is None:
            running["proc"].terminate()
