"""Generate training data using a local model on GPU (no API calls).

Uses the base model already on disk (e.g. Qwen2.5-Coder-7B) with 4-bit
quantization to generate openclaw-format training examples entirely locally.

Usage:
    python scripts/generate_from_local_model.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --count 200 \
        --out data/local_generated.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = "You are OpenClaw assistant. Be accurate, concise, and safe."

TOPICS = [
    "password reset", "account lockout", "billing mismatch",
    "subscription cancel", "api token rotation", "rate limit error",
    "tool timeout", "json parse failure", "build failure",
    "test flake", "deployment rollback", "database migration",
    "cache invalidation", "permissions error", "session expiry",
    "container OOM kill", "pod crash loop", "DNS resolution failure",
    "TLS handshake timeout", "memory leak detection",
    "merge conflict resolution", "dependency version conflict",
    "circular import", "deadlock in transaction", "race condition",
    "schema drift", "ETL job timeout", "replication lag",
    "data corruption detection", "secret rotation",
]

SCENARIOS = [
    "in a production Kubernetes cluster",
    "on a staging server after a deploy",
    "in a CI/CD pipeline",
    "during a live customer demo",
    "in a microservices architecture",
    "on a legacy monolith",
    "in a multi-tenant SaaS platform",
    "after a database migration",
]

GENERATION_PROMPT = """You are generating a training example for an AI coding assistant called OpenClaw.

Topic: {topic} {scenario}

Generate a realistic conversation where:
1. The user describes a specific, realistic problem (include error messages, file paths, or commands)
2. You (as OpenClaw) provide an actionable, structured response with numbered steps
3. Your response should be 3-6 sentences, concise and practical

Respond with ONLY a JSON object (no markdown, no code fences):
{{"user": "the user's question", "assistant": "your helpful response"}}"""


def load_model(model_name: str) -> tuple[Any, Any]:
    """Load model with 4-bit quantization to fit in 12GB VRAM."""
    print(f"Loading {model_name} with 4-bit quantization ...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for local model generation.\n"
            "For CPU-only environments, use: python scripts/generate_openclaw_examples.py"
        )

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
    total_gb = torch.cuda.mem_get_info(0)[1] / (1024**3)
    print(f"Model loaded. VRAM: {total_gb - free_gb:.1f}/{total_gb:.1f} GB used")
    return model, tokenizer


def generate_single(
    model: Any,
    tokenizer: Any,
    topic: str,
    scenario: str,
) -> dict[str, str] | None:
    """Generate one training example via local model inference."""
    prompt = GENERATION_PROMPT.format(topic=topic, scenario=scenario)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()

    # Try to parse JSON from the response
    try:
        # Strip markdown fences if present
        text = generated
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        parsed = json.loads(text)
        if "user" in parsed and "assistant" in parsed:
            return {"user": str(parsed["user"]), "assistant": str(parsed["assistant"])}
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data using a local model on GPU."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        help="Local model name or path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.getenv("LOCAL_GEN_COUNT", "200")),
        help="Number of examples to generate.",
    )
    parser.add_argument(
        "--out",
        default=os.getenv("LOCAL_GEN_FILE", "data/local_generated.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model)
    source_tag = f"local:{args.model.split('/')[-1]}"

    rows: list[dict[str, Any]] = []
    errors = 0
    max_errors = max(10, args.count // 5)

    print(f"\n🤖 Generating {args.count} examples locally ...")

    for i in range(args.count):
        topic = TOPICS[i % len(TOPICS)]
        scenario = SCENARIOS[i % len(SCENARIOS)]

        result = generate_single(model, tokenizer, topic, scenario)

        if result is None:
            errors += 1
            if errors >= max_errors:
                print(f"  ❌ Too many parse errors ({errors}), stopping early.")
                break
            continue

        row = {
            "source": source_tag,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": result["user"]},
                {"role": "assistant", "content": result["assistant"]},
            ],
        }
        rows.append(row)

        if (i + 1) % 25 == 0:
            print(f"  📝 {i + 1}/{args.count} generated ({errors} errors)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"\n🎉 Wrote {len(rows)} examples to {out_path} ({errors} errors)")


if __name__ == "__main__":
    main()
