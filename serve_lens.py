"""Local web UI: type a prompt (or chat with the model), recompute the
Jacobian-lens slice, view it.

Loads the model and lens once, then serves on http://127.0.0.1:<port>/.
The page itself is lens_ui.html, re-read on every request, so UI edits only
need a browser refresh.

    .venv/bin/python serve_lens.py            # port 8765
    .venv/bin/python serve_lens.py --port 9000
"""

import argparse
import base64
import gzip
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import transformers

import jlens
from jlens import vis
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice
from run_local import LENS_FILE, LENS_REPO, LENS_REVISION, MODEL_NAME, pick_device

UI_HTML = Path(__file__).with_name("lens_ui.html")
D3_CACHE = Path.home() / ".cache" / "jlens" / "d3-7.9.0.min.js"

#: Longest conversation (in tokens) a chat turn may reach. compute_slice
#: truncates from the right, so anything longer would drop the newest turn.
MAX_CONTEXT = 4096
#: Cap on tokens that get a full rank tensor (pinnable / rank-heatmap). The
#: default (every token in any top-K cell) makes long-chat pages huge.
MAX_TRACKED = 300


def seed_d3_from_cache() -> None:
    """Embed mode downloads d3 on first use; inline the local copy instead so
    pages build offline. Skipped (falls back to the download) if the cached
    file is missing or fails the same integrity check vis.py applies."""
    if not D3_CACHE.exists():
        return
    body = D3_CACHE.read_bytes()
    sri = "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode()
    if sri != vis._D3_SRI:
        print(f"warning: {D3_CACHE} failed integrity check; will download d3")
        return
    vis._TEMPLATE_FOR_MODE["embed"] = vis.PAGE_TEMPLATE.replace(
        "__D3__", f"<script>\n{body.decode()}\n</script>"
    )


class LensApp:
    def __init__(self) -> None:
        self.device = pick_device()
        print(f"device: {self.device}; loading {MODEL_NAME} ...")
        self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = jlens.from_hf(self.hf_model, self.tokenizer)
        self.lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
        )
        with gzip.open("assets/qwen_gloss.json.gz") as f:
            self.gloss = {int(k): v for k, v in json.load(f).items()}
        self.examples = [
            {"slug": e.slug, "section": e.section, "prompt": resolve_prompt(e, self.tokenizer)}
            for e in EXAMPLES
        ]
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # One forward pass at a time: the model and activation hooks are shared.
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"model": MODEL_NAME, "device": str(self.device), "examples": self.examples}

    def _page(self, slice_data, text: str, title: str) -> str:
        page, _, _ = build_page(
            slice_data, text, title=title, description="", alt_token=self.gloss
        )
        return page

    def slice_page(self, req: dict) -> tuple[str, int]:
        prompt = req["prompt"]
        if req.get("chat"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        with self.lock:
            slice_data = compute_slice(
                self.model,
                self.lens,
                prompt,
                layer_stride=max(1, int(req.get("layer_stride", 1))),
                mask_display=bool(req.get("mask_display", True)),
            )
        page = self._page(slice_data, prompt, req.get("title") or "Custom prompt")
        return page, len(slice_data.context_token_ids)

    def chat(self, req: dict) -> dict:
        """Generate the assistant's next reply, then slice the whole
        conversation including that reply."""
        messages = [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in req.get("messages", [])
        ]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("last message must be from the user")
        if any(m["role"] not in ("user", "assistant") for m in messages):
            raise ValueError("roles must be 'user' or 'assistant'")
        max_new = min(max(1, int(req.get("max_new_tokens", 200))), 1024)
        window = max(16, int(req.get("window", 256)))

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        if input_ids.shape[1] + max_new > MAX_CONTEXT:
            raise ValueError(
                f"conversation is {input_ids.shape[1]} tokens; with a {max_new}-token "
                f"reply it would exceed {MAX_CONTEXT}. Start a new chat."
            )

        with self.lock:
            t0 = time.perf_counter()
            output = self.hf_model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,  # greedy: the same conversation gives the same reply
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_secs = time.perf_counter() - t0
            new_ids = output[0, input_ids.shape[1] :].tolist()
            if self.end_of_turn in new_ids:  # keep <|im_end|>, drop what follows
                new_ids = new_ids[: new_ids.index(self.end_of_turn) + 1]
            reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            # Slice exactly what the model saw plus what it wrote.
            text = prompt + self.tokenizer.decode(new_ids, skip_special_tokens=False)

            t0 = time.perf_counter()
            slice_data = compute_slice(
                self.model,
                self.lens,
                text,
                layer_stride=max(1, int(req.get("layer_stride", 1))),
                mask_display=bool(req.get("mask_display", True)),
                last_n_tokens=window,
                max_seq_len=MAX_CONTEXT,
                max_tracked=MAX_TRACKED,
            )
            slice_secs = time.perf_counter() - t0

        total = len(slice_data.context_token_ids)
        return {
            "reply": reply,
            "page": self._page(slice_data, text, "Chat"),
            "reply_tokens": len(new_ids),
            "total_tokens": total,
            "shown_tokens": min(window, total),
            "gen_secs": round(gen_secs, 1),
            "slice_secs": round(slice_secs, 1),
        }


def make_handler(app: LensApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: str, content_type: str, **headers) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, UI_HTML.read_text(encoding="utf-8"), "text/html")
            elif self.path == "/info":
                self._send(200, json.dumps(app.info(), ensure_ascii=False), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            if self.path not in ("/slice", "/chat"):
                self._send(404, "not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length))
                t0 = time.perf_counter()
                if self.path == "/slice":
                    if not str(req.get("prompt", "")).strip():
                        raise ValueError("prompt is empty")
                    page, n_tokens = app.slice_page(req)
                    print(f"slice: {n_tokens} tokens in {time.perf_counter() - t0:.1f}s")
                else:
                    result = app.chat(req)
                    print(
                        f"chat: {result['reply_tokens']} reply tokens, "
                        f"{result['total_tokens']} total, "
                        f"gen {result['gen_secs']}s + slice {result['slice_secs']}s"
                    )
            except Exception as exc:  # report any failure to the page
                self._send(400, f"{type(exc).__name__}: {exc}", "text/plain")
                return
            if self.path == "/slice":
                self._send(200, page, "text/html", X_Tokens=str(n_tokens))
            else:
                self._send(200, json.dumps(result, ensure_ascii=False), "application/json")

        def log_message(self, format: str, *args) -> None:
            pass  # keep the console for slice timings

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    jlens.configure_logging()
    seed_d3_from_cache()
    app = LensApp()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"-> http://localhost:{args.port}/   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
