import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = Path(path).read_text(encoding="utf-8").splitlines()
    for line in data:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    # Filter out assistant messages to build the prompt for generation
    chat = [m for m in messages if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


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


def run_model_batch(model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Run batched inference with left-padding for consistent generation."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = original_padding_side

    # Decode only the generated tokens (skip the input)
    input_len = encoded["input_ids"].shape[1]
    results = []
    for i in range(out.shape[0]):
        text = tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        results.append(text.strip())
    return results


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_new_tokens: int, batch_size: int = 4) -> dict[str, float]:
    exact_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for i in tqdm(range(0, len(rows), batch_size), desc="Evaluating"):
        batch = rows[i : i + batch_size]
        prompts = [render_prompt(r["messages"], tokenizer) for r in batch]
        truths = [reference_answer(r["messages"]) for r in batch]
        preds = run_model_batch(model, tokenizer, prompts, max_new_tokens)
        for pred, truth in zip(preds, truths):
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="openclaw-ft/data/eval_results.json")
    args = parser.parse_args()

    rows = load_jsonl(args.eval_file)

    base_model, tokenizer = load_base(args.base_model)
    baseline_metrics = evaluate(base_model, tokenizer, rows, args.max_new_tokens, args.batch_size)

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    finetuned_metrics = evaluate(tuned_model, tokenizer, rows, args.max_new_tokens, args.batch_size)

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
