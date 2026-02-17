"""Fetch and normalise HuggingFace datasets into the openclaw JSONL format.

Usage:
    python scripts/fetch_hf_dataset.py \
        --datasets Open-Orca/OpenOrca,teknium/OpenHermes-2.5,sahil2801/CodeAlpaca-20k \
        --max-per-dataset 2000 \
        --out data/hf_combined.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = "You are OpenClaw assistant. Be accurate, concise, and safe."

DEFAULT_DATASETS = [
    "Open-Orca/OpenOrca",
    "teknium/OpenHermes-2.5",
    "sahil2801/CodeAlpaca-20k",
]


# ---------------------------------------------------------------------------
# Normalisation helpers — each HF dataset has different column schemas.
# ---------------------------------------------------------------------------


def normalise_row(row: dict[str, Any], dataset_id: str) -> list[dict[str, str]] | None:
    """Convert a single HF row into an openclaw messages list.

    Returns None if the row cannot be normalised (skip it).
    """
    # ----- Format 1: already has "messages" (list of dicts) -----
    if "messages" in row and isinstance(row["messages"], list):
        msgs = []
        for m in row["messages"]:
            if isinstance(m, dict) and "role" in m and "content" in m:
                msgs.append({"role": str(m["role"]), "content": str(m["content"])})
        if len(msgs) >= 2:
            return msgs
        return None

    # ----- Format 2: "conversations" (ShareGPT / OpenHermes style) -----
    if "conversations" in row and isinstance(row["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        msgs: list[dict[str, str]] = []
        for turn in row["conversations"]:
            if isinstance(turn, dict):
                from_key = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                role = role_map.get(str(from_key).lower(), str(from_key).lower())
                if role and value:
                    msgs.append({"role": role, "content": str(value)})
        if len(msgs) >= 2:
            # Ensure system prompt exists
            if msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            return msgs
        return None

    # ----- Format 3: instruction / input / output (Alpaca style) -----
    instruction = row.get("instruction", "")
    output = row.get("output", "")
    if instruction and output:
        inp = row.get("input", "")
        user_content = f"{instruction}\n\n{inp}".strip() if inp else str(instruction)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(output)},
        ]

    # ----- Format 4: question / response (OpenOrca style) -----
    question = row.get("question", "")
    response = row.get("response", "")
    if question and response:
        system = row.get("system_prompt", SYSTEM_PROMPT) or SYSTEM_PROMPT
        return [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(response)},
        ]

    # ----- Format 5: prompt / completion -----
    prompt = row.get("prompt", "")
    completion = row.get("completion", row.get("chosen", ""))
    if prompt and completion:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(completion)},
        ]

    return None


def fetch_and_normalise(
    dataset_id: str,
    max_rows: int,
    seed: int,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Download a HF dataset and convert it into openclaw JSONL rows."""
    print(f"\n📦 Fetching {dataset_id} (split={split}, max={max_rows}) ...")

    try:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception:
        # Some datasets only have specific configs; try without split selection
        try:
            ds = load_dataset(dataset_id, trust_remote_code=True)
            if isinstance(ds, dict):
                ds = ds.get("train", ds.get(list(ds.keys())[0]))
        except Exception as exc:
            print(f"  ⚠ Could not load {dataset_id}: {exc}")
            return []

    # Subsample if dataset is larger than max_rows
    total = len(ds)
    if total > max_rows:
        rng = random.Random(seed)
        indices = rng.sample(range(total), max_rows)
        ds = ds.select(indices)

    rows: list[dict[str, Any]] = []
    source_tag = f"hf:{dataset_id}"
    skipped = 0

    for item in tqdm(ds, desc=f"  Normalising {dataset_id}", leave=False):
        messages = normalise_row(item, dataset_id)
        if messages is None:
            skipped += 1
            continue
        rows.append({"source": source_tag, "messages": messages})

    print(f"  ✅ {len(rows)} rows normalised, {skipped} skipped")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch HuggingFace datasets and convert to openclaw JSONL format."
    )
    parser.add_argument(
        "--datasets",
        default=os.getenv("HF_DATASETS", ",".join(DEFAULT_DATASETS)),
        help="Comma-separated HuggingFace dataset IDs.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=int(os.getenv("HF_MAX_PER_DATASET", "2000")),
        help="Maximum rows to sample per dataset.",
    )
    parser.add_argument(
        "--out",
        default=os.getenv("HF_FILE", "data/hf_combined.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_rows: list[dict[str, Any]] = []

    for ds_id in dataset_ids:
        rows = fetch_and_normalise(ds_id, args.max_per_dataset, args.seed)
        all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"\n🎉 Wrote {len(all_rows)} total rows to {out_path}")
    print(f"   Sources: {', '.join(dataset_ids)}")


if __name__ == "__main__":
    main()
