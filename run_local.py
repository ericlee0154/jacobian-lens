"""Apply a pre-fitted Jacobian lens on whatever accelerator this machine has."""

import torch
import transformers
import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"



import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import gzip
import json

from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    jlens.configure_logging()
    device = pick_device()
    print(f"device: {device}")

    # Load causal model from Hugging Face per device type
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to(device)
    # Download tokenizer from hugging face based on loaded model name (if model comes from hugging face
    # then tokenizer should be downloaded from hugging face as well)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    # Calling jlens from_hf() to load Hugging Face model
    model = jlens.from_hf(hf_model, tokenizer)

    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )

    prompt = "Fact: The currency used in the country shaped like a boot is"
    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]
    jlens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=layers, positions=[-2]
    )
    logit_lens, _, _ = lens.apply(
        model, prompt, layers=layers, positions=[-2], use_jacobian=False
    )

    def top5(logits):
        return [tokenizer.decode([t]) for t in logits.topk(5).indices]

    for layer in layers:
        print(f"L{layer:>3} logit-lens: {top5(logit_lens[layer][0])}")
        print(f"L{layer:>3} J-lens:     {top5(jlens_logits[layer][0])}")
    print(f"model:           {top5(model_logits[0])}")



    # English gloss for Qwen's Chinese/Japanese/Korean vocab tokens (machine-
    # generated, best-effort), shown next to
    # the token in the page (alt_token=).
    gloss = {
        int(k): v for k, v in json.load(gzip.open("assets/qwen_gloss.json.gz")).items()
    }

    example = next(e for e in EXAMPLES if e.slug == "multihop")
    prompt = resolve_prompt(example, tokenizer)

    slice_data = compute_slice(
        model,
        lens,
        prompt,
        layer_stride=2,
        # Empirically on Qwen, the interesting word tokens trail punctuation and
        # single-character tokens in the raw top-K; mask to word-like tokens only.
        mask_display=True,
    )
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=example.section,
        description=example.description,
        alt_token=gloss,
    )
    # notebook_iframe() only renders inside Jupyter; in a script, write the
    # self-contained (embed mode) page to a file and open it in the browser.
    Path("slices").mkdir(exist_ok=True)
    Path("slices/multihop.html").write_text(page)
    print("-> wrote slices/multihop.html")

    # create a static page for the "ascii-face" example, and serve it on a local HTTP server
    example = next(e for e in EXAMPLES if e.slug == "ascii-face")
    prompt = resolve_prompt(example, tokenizer)

    slice_data = compute_slice(model, lens, prompt, mask_display=True)
    out_dir = Path("slices") / example.slug
    page, _, _ = build_page(
        slice_data,
        prompt,
        title=example.section,
        description=example.description,
        alt_token=gloss,
        mode="fetch",
        out_dir=out_dir,
    )
    (out_dir / "index.html").write_text(page)

    # In a script, serve in the foreground: a daemon thread would die as soon as
    # main() returns. Ctrl+C stops the server.
    handler = partial(SimpleHTTPRequestHandler, directory=os.path.abspath("slices"))
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    print(f"-> http://localhost:{httpd.server_address[1]}/{example.slug}/")
    print("   serving; press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
