import json
import random
from pathlib import Path

import gradio as gr

from scripts.generate_openclaw_examples import generate


OUTPUT_DIR = Path("artifacts/space")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


with gr.Blocks(title="OpenClaw Dataset Builder") as demo:
    gr.Markdown(
        """
# OpenClaw Dataset Builder

Generate synthetic OpenClaw-format training data directly in this Space.
This UI runs the local template generator from `scripts/generate_openclaw_examples.py`.
"""
    )

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


demo.queue(default_concurrency_limit=1)
demo.launch(server_name="0.0.0.0", server_port=7860)
