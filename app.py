import json
import os
import random
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Some Space builds intermittently miss transitive deps; recover at startup.
try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests>=2.31.0"])

import gradio as gr

from scripts.generate_openclaw_examples import generate


OUTPUT_DIR = Path("artifacts/space")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_CONFIG: dict[str, str] = {}


def build_dataset(count_per_style: int, seed: int):
    random.seed(seed)
    rows = generate(count_per_style)
    random.shuffle(rows)

    out_path = OUTPUT_DIR / f"openclaw_dataset_cps{count_per_style}_seed{seed}.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    stats: dict[str, int] = {}
    for row in rows:
        src = row["source"]
        stats[src] = stats.get(src, 0) + 1

    summary_lines = [
        f"Generated {len(rows)} rows",
        f"Output: {out_path}",
        "",
        "Style counts:",
    ]
    for style, count in sorted(stats.items()):
        summary_lines.append(f"- {style}: {count}")

    preview = []
    for row in rows[:10]:
        user_text = ""
        assistant_text = ""
        for message in row.get("messages", []):
            if message.get("role") == "user" and not user_text:
                user_text = message.get("content", "")
            if message.get("role") == "assistant" and not assistant_text:
                assistant_text = message.get("content", "")
        preview.append([row.get("source", ""), user_text, assistant_text])

    return "\n".join(summary_lines), preview, str(out_path)


def _cfg(key: str) -> str:
    return RUNTIME_CONFIG.get(key, os.getenv(key, "")).strip()


def _masked(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:3] + ("*" * (len(value) - 6)) + value[-3:]


def save_runtime_config(
    model_base_url: str,
    model_name: str,
    model_api_key: str,
    data_base_url: str,
    data_token: str,
) -> str:
    RUNTIME_CONFIG["LOCAL_MODEL_BASE_URL"] = model_base_url.strip()
    RUNTIME_CONFIG["LOCAL_MODEL_NAME"] = model_name.strip()
    RUNTIME_CONFIG["LOCAL_MODEL_API_KEY"] = model_api_key.strip()
    RUNTIME_CONFIG["LOCAL_DATA_BASE_URL"] = data_base_url.strip()
    RUNTIME_CONFIG["LOCAL_DATA_TOKEN"] = data_token.strip()
    return (
        "Runtime config saved.\n"
        f"LOCAL_MODEL_BASE_URL={_cfg('LOCAL_MODEL_BASE_URL') or '(empty)'}\n"
        f"LOCAL_MODEL_NAME={_cfg('LOCAL_MODEL_NAME') or '(empty)'}\n"
        f"LOCAL_MODEL_API_KEY={_masked(_cfg('LOCAL_MODEL_API_KEY'))}\n"
        f"LOCAL_DATA_BASE_URL={_cfg('LOCAL_DATA_BASE_URL') or '(empty)'}\n"
        f"LOCAL_DATA_TOKEN={_masked(_cfg('LOCAL_DATA_TOKEN'))}"
    )


def check_model_health() -> str:
    base_url = _cfg("LOCAL_MODEL_BASE_URL").rstrip("/")
    api_key = _cfg("LOCAL_MODEL_API_KEY")
    if not base_url:
        return "LOCAL_MODEL_BASE_URL is empty."
    req = urllib.request.Request(url=f"{base_url}/v1/models", method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return f"Model endpoint reachable.\n{raw[:1200]}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP error {exc.code}: {body}"
    except Exception as exc:
        return f"Request failed: {exc}"


def check_data_health() -> str:
    base_url = _cfg("LOCAL_DATA_BASE_URL").rstrip("/")
    token = _cfg("LOCAL_DATA_TOKEN")
    if not base_url:
        return "LOCAL_DATA_BASE_URL is empty."
    req = urllib.request.Request(url=f"{base_url}/health", method="GET")
    if token:
        req.add_header("X-Bridge-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return f"Data bridge reachable.\n{raw[:1200]}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP error {exc.code}: {body}"
    except Exception as exc:
        return f"Request failed: {exc}"


def call_remote_local_model(system_prompt: str, user_prompt: str, temperature: float, max_tokens: int):
    base_url = _cfg("LOCAL_MODEL_BASE_URL").rstrip("/")
    model = _cfg("LOCAL_MODEL_NAME")
    api_key = _cfg("LOCAL_MODEL_API_KEY")

    if not base_url or not model:
        return (
            "Missing configuration. Set Space secrets/vars:\n"
            "- LOCAL_MODEL_BASE_URL (public HTTPS URL to your local model server)\n"
            "- LOCAL_MODEL_NAME (model id on that server)\n"
            "Optional: LOCAL_MODEL_API_KEY"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    req = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP error {exc.code}: {body}"
    except Exception as exc:
        return f"Request failed: {exc}"

    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except Exception:
        return f"Unexpected response:\n{raw[:2000]}"


def fetch_local_data_via_bridge(relative_path: str):
    base_url = _cfg("LOCAL_DATA_BASE_URL").rstrip("/")
    token = _cfg("LOCAL_DATA_TOKEN")

    if not base_url:
        return (
            "Missing LOCAL_DATA_BASE_URL. Configure Space variable with your "
            "Tailscale-exposed local data bridge URL."
        )

    rel = relative_path.strip().lstrip("/")
    if not rel:
        return "Provide a relative file path."

    req = urllib.request.Request(
        url=f"{base_url}/read?path={urllib.parse.quote(rel)}",
        method="GET",
    )
    if token:
        req.add_header("X-Bridge-Token", token)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP error {exc.code}: {body}"
    except Exception as exc:
        return f"Request failed: {exc}"

    try:
        data = json.loads(raw)
        content = str(data.get("content", ""))
    except Exception:
        return f"Unexpected response:\n{raw[:2000]}"

    preview = content[:8000]
    if len(content) > 8000:
        preview += "\n\n[truncated]"
    return preview


with gr.Blocks(title="OpenClaw Dataset Builder") as demo:
    gr.Markdown(
        """
# OpenClaw Dataset Builder

Generate synthetic OpenClaw-format training data directly in this Space.
This UI runs the local template generator from `scripts/generate_openclaw_examples.py`.
"""
    )

    with gr.Tab("Dataset Builder"):
        with gr.Row():
            count_per_style = gr.Slider(
                minimum=1,
                maximum=300,
                step=1,
                value=40,
                label="Count Per Style",
            )
            seed = gr.Number(value=42, precision=0, label="Seed")

        run_btn = gr.Button("Generate Dataset", variant="primary")
        summary = gr.Textbox(label="Run Summary", lines=10)
        preview = gr.Dataframe(
            headers=["source", "user", "assistant"],
            datatype=["str", "str", "str"],
            label="Preview (first 10 rows)",
            wrap=True,
        )
        download = gr.File(label="Download JSONL")

        run_btn.click(
            fn=build_dataset,
            inputs=[count_per_style, seed],
            outputs=[summary, preview, download],
        )

    with gr.Tab("Local Model Proxy"):
        gr.Markdown(
            "Use this Space as the service/UI while keeping model weights on your own machine. "
            "Set `LOCAL_MODEL_BASE_URL` and `LOCAL_MODEL_NAME` in Space variables."
        )
        system_prompt = gr.Textbox(label="System prompt", value="You are OpenClaw assistant. Be accurate, concise, and safe.")
        user_prompt = gr.Textbox(label="User prompt", lines=6, placeholder="Ask something...")
        with gr.Row():
            temperature = gr.Slider(minimum=0.0, maximum=1.5, step=0.1, value=0.2, label="Temperature")
            max_tokens = gr.Slider(minimum=32, maximum=4096, step=32, value=512, label="Max tokens")
        ask_btn = gr.Button("Run via local model", variant="primary")
        model_output = gr.Textbox(label="Model output", lines=16)

        ask_btn.click(
            fn=call_remote_local_model,
            inputs=[system_prompt, user_prompt, temperature, max_tokens],
            outputs=[model_output],
        )

    with gr.Tab("Local Data (Tailscale)"):
        gr.Markdown(
            "Fetch local files through your Tailscale-exposed bridge. "
            "Set `LOCAL_DATA_BASE_URL` and optional `LOCAL_DATA_TOKEN` in Space variables/secrets."
        )
        relative_path = gr.Textbox(
            label="Relative path",
            value="data/openclaw_workflow_tools.jsonl",
            placeholder="e.g. data/train.jsonl",
        )
        fetch_btn = gr.Button("Fetch local file", variant="primary")
        file_preview = gr.Textbox(label="File preview", lines=18)

        fetch_btn.click(
            fn=fetch_local_data_via_bridge,
            inputs=[relative_path],
            outputs=[file_preview],
        )

    with gr.Tab("Control Panel"):
        gr.Markdown("Update runtime endpoint settings and run health checks.")
        ctrl_model_base = gr.Textbox(label="LOCAL_MODEL_BASE_URL", value=os.getenv("LOCAL_MODEL_BASE_URL", ""))
        ctrl_model_name = gr.Textbox(label="LOCAL_MODEL_NAME", value=os.getenv("LOCAL_MODEL_NAME", ""))
        ctrl_model_key = gr.Textbox(label="LOCAL_MODEL_API_KEY", value=os.getenv("LOCAL_MODEL_API_KEY", ""), type="password")
        ctrl_data_base = gr.Textbox(label="LOCAL_DATA_BASE_URL", value=os.getenv("LOCAL_DATA_BASE_URL", ""))
        ctrl_data_token = gr.Textbox(label="LOCAL_DATA_TOKEN", value=os.getenv("LOCAL_DATA_TOKEN", ""), type="password")
        with gr.Row():
            save_btn = gr.Button("Save Runtime Config", variant="primary")
            model_health_btn = gr.Button("Check Model Health")
            data_health_btn = gr.Button("Check Data Health")
        control_output = gr.Textbox(label="Control output", lines=12)

        save_btn.click(
            fn=save_runtime_config,
            inputs=[ctrl_model_base, ctrl_model_name, ctrl_model_key, ctrl_data_base, ctrl_data_token],
            outputs=[control_output],
        )
        model_health_btn.click(fn=check_model_health, outputs=[control_output])
        data_health_btn.click(fn=check_data_health, outputs=[control_output])


demo.queue(default_concurrency_limit=1)
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
