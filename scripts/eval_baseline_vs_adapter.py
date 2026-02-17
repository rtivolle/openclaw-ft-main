import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = Path(path).read_text(encoding="utf-8").splitlines()
    for line in data:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message["role"] == "assistant":
            continue
        parts.append(f"<|{message['role']}|>\n{message['content']}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def reference_answer(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"].strip()
    return ""


def score_row(prediction: str, truth: str) -> dict[str, float]:
    pred = prediction.strip().lower()
    ans = truth.strip().lower()
    exact = 1.0 if pred == ans else 0.0
    pred_tokens = set(re.findall(r"\w+", pred))
    ans_tokens = set(re.findall(r"\w+", ans))
    overlap = len(pred_tokens & ans_tokens)
    denom = max(1, len(ans_tokens))
    keyword_recall = overlap / denom
    precision = overlap / max(1, len(pred_tokens))
    token_f1 = 0.0 if (precision + keyword_recall) == 0 else (2 * precision * keyword_recall) / (precision + keyword_recall)
    return {"exact_match": exact, "keyword_recall": keyword_recall, "token_f1": token_f1}


def run_model(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_new_tokens: int) -> dict[str, float]:
    exact_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for row in rows:
        messages = row["messages"]
        prompt = render_prompt(messages)
        truth = reference_answer(messages)
        pred = run_model(model, tokenizer, prompt, max_new_tokens)
        s = score_row(pred, truth)
        exact_sum += s["exact_match"]
        recall_sum += s["keyword_recall"]
        f1_sum += s["token_f1"]
    n = max(1, len(rows))
    return {
        "exact_match": exact_sum / n,
        "keyword_recall": recall_sum / n,
        "token_f1": f1_sum / n,
    }


def load_base(base_model: str) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation. No CUDA device detected.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs adapter checkpoint.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--eval-file", default="openclaw-ft/data/val_20.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="openclaw-ft/data/eval_results.json")
    args = parser.parse_args()

    rows = load_jsonl(args.eval_file)

    base_model, tokenizer = load_base(args.base_model)
    baseline_metrics = evaluate(base_model, tokenizer, rows, args.max_new_tokens)

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    finetuned_metrics = evaluate(tuned_model, tokenizer, rows, args.max_new_tokens)

    delta_exact = finetuned_metrics["exact_match"] - baseline_metrics["exact_match"]
    delta_recall = finetuned_metrics["keyword_recall"] - baseline_metrics["keyword_recall"]
    delta_f1 = finetuned_metrics["token_f1"] - baseline_metrics["token_f1"]

    result = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "delta": {
            "exact_match": delta_exact,
            "keyword_recall": delta_recall,
            "token_f1": delta_f1,
        },
        "is_real_improvement": delta_exact >= 0.05 and delta_recall >= 0.05,
    }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
