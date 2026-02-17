import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<REDACTED_EMAIL>"),
    (re.compile(r"\b(ghp|github_pat)_[A-Za-z0-9_]{20,}\b"), "<REDACTED_GH_TOKEN>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "<REDACTED_API_KEY>"),
    (re.compile(r"(?i)\b(bearer|token|api[_-]?key)\s*[:=]\s*[A-Za-z0-9._-]{8,}\b"), "<REDACTED_SECRET>"),
    (re.compile(r"https?://[^\s]+"), "<REDACTED_URL>"),
    (re.compile(r"(?:(?:/Users|/home|/root)/[^\s'\"`]+)"), "<REDACTED_PATH>"),
    (re.compile(r"(?:[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+)"), "<REDACTED_PATH>"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def redact_text(text: str) -> tuple[str, int]:
    replaced = text
    replacements = 0
    for pattern, token in REDACTION_RULES:
        replaced, count = pattern.subn(token, replaced)
        replacements += count
    return replaced, replacements


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in row["messages"] if "role" in m and "content" in m]

    if isinstance(row.get("conversation"), list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in row["conversation"] if "role" in m and "content" in m]

    user = row.get("user")
    assistant = row.get("assistant")
    if isinstance(user, str) and isinstance(assistant, str):
        system = row.get("system", "You are OpenClaw assistant. Be accurate, concise, and safe.")
        return [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    return []


def redact_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    cleaned: list[dict[str, str]] = []
    replaced_total = 0
    for message in messages:
        content, count = redact_text(message["content"])
        cleaned.append({"role": message["role"], "content": content})
        replaced_total += count
    return cleaned, replaced_total


def assistant_len(messages: list[dict[str, str]]) -> int:
    for message in messages:
        if message["role"].strip().lower() == "assistant":
            return len(message["content"].strip())
    return 0


def user_len(messages: list[dict[str, str]]) -> int:
    for message in messages:
        if message["role"].strip().lower() == "user":
            return len(message["content"].strip())
    return 0


def fingerprint(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clamp_count(count: int, available: int) -> int:
    return min(max(0, count), max(0, available))


def choose_bucket(total: int, ratio: float, available: int) -> int:
    return clamp_count(int(round(total * ratio)), available)


def add_rows(
    rows: list[dict[str, Any]],
    bucket: str,
    default_source: str,
    default_category: str,
    parsed: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    stats["raw_rows"] += len(rows)
    for row in rows:
        messages = normalize_messages(row)
        if not messages:
            stats["rejected_malformed"] += 1
            continue

        cleaned_messages, replacements = redact_messages(messages)
        stats["redactions"] += replacements
        if replacements > 0:
            stats["rows_with_redactions"] += 1

        if user_len(cleaned_messages) < 10 or assistant_len(cleaned_messages) < 20:
            stats["rejected_low_signal"] += 1
            continue

        parsed.append(
            {
                "source_bucket": bucket,
                "source": str(row.get("source", default_source)),
                "task_category": str(row.get("task_category", default_category)),
                "outcome": str(row.get("outcome", "unknown")),
                "messages": cleaned_messages,
            }
        )


def read_optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return load_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenCode/OpenClaw protocol train/val datasets with QC and manifest.")
    parser.add_argument("--raw-trace-glob", default="data/raw_traces/*.jsonl")
    parser.add_argument("--synthetic-file", default="data/train_200.jsonl")
    parser.add_argument("--adversarial-file", default="data/adversarial.jsonl")
    parser.add_argument("--target-examples", type=int, default=20000)
    parser.add_argument("--min-examples", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--ratio-real", type=float, default=0.6)
    parser.add_argument("--ratio-synthetic", type=float, default=0.3)
    parser.add_argument("--ratio-adversarial", type=float, default=0.1)
    parser.add_argument("--max-per-source", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--min-redaction-rate", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)

    ratios_sum = args.ratio_real + args.ratio_synthetic + args.ratio_adversarial
    if abs(ratios_sum - 1.0) > 0.01:
        raise ValueError(f"Mix ratios must sum to 1.0, got {ratios_sum:.3f}")

    parsed: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "raw_rows": 0,
        "rejected_malformed": 0,
        "rejected_low_signal": 0,
        "redactions": 0,
        "rows_with_redactions": 0,
    }

    raw_files = sorted(glob.glob(args.raw_trace_glob))
    for raw_path in raw_files:
        add_rows(
            rows=load_jsonl(Path(raw_path)),
            bucket="real",
            default_source="trace",
            default_category="agent_task",
            parsed=parsed,
            stats=stats,
        )

    synthetic_rows = read_optional_jsonl(Path(args.synthetic_file))
    if synthetic_rows:
        add_rows(
            rows=synthetic_rows,
            bucket="synthetic",
            default_source="synthetic",
            default_category="coverage",
            parsed=parsed,
            stats=stats,
        )

    adversarial_rows = read_optional_jsonl(Path(args.adversarial_file))
    if adversarial_rows:
        add_rows(
            rows=adversarial_rows,
            bucket="adversarial",
            default_source="adversarial",
            default_category="unsafe_request",
            parsed=parsed,
            stats=stats,
        )

    source_limited: dict[str, int] = {}
    filtered: list[dict[str, Any]] = []
    for row in parsed:
        source = row["source"]
        count = source_limited.get(source, 0)
        if count >= args.max_per_source:
            continue
        source_limited[source] = count + 1
        filtered.append(row)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in filtered:
        fp = fingerprint(row["messages"])
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(row)

    real_rows = [row for row in deduped if row["source_bucket"] == "real"]
    synthetic_rows = [row for row in deduped if row["source_bucket"] == "synthetic"]
    adversarial_rows = [row for row in deduped if row["source_bucket"] == "adversarial"]

    target = min(args.target_examples, len(deduped))
    n_real = choose_bucket(target, args.ratio_real, len(real_rows))
    n_synthetic = choose_bucket(target, args.ratio_synthetic, len(synthetic_rows))
    n_adversarial = choose_bucket(target, args.ratio_adversarial, len(adversarial_rows))

    selected = (
        random.sample(real_rows, n_real) if n_real > 0 else []
    ) + (
        random.sample(synthetic_rows, n_synthetic) if n_synthetic > 0 else []
    ) + (
        random.sample(adversarial_rows, n_adversarial) if n_adversarial > 0 else []
    )

    if len(selected) < target:
        remainder_pool = [row for row in deduped if row not in selected]
        needed = min(target - len(selected), len(remainder_pool))
        if needed > 0:
            selected += random.sample(remainder_pool, needed)

    random.shuffle(selected)
    val_count = int(round(len(selected) * args.val_ratio))
    val_count = clamp_count(val_count, len(selected))
    val_rows = selected[:val_count]
    train_rows = selected[val_count:]

    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    manifest_path = Path(args.manifest_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in train_rows:
            handle.write(json.dumps({"source": row["source"], "messages": row["messages"]}, ensure_ascii=True) + "\n")

    with val_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in val_rows:
            handle.write(json.dumps({"source": row["source"], "messages": row["messages"]}, ensure_ascii=True) + "\n")

    category_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for row in selected:
        category = row["task_category"]
        outcome = row["outcome"]
        category_counts[category] = category_counts.get(category, 0) + 1
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    redaction_rate = 0.0 if stats["raw_rows"] == 0 else stats["rows_with_redactions"] / stats["raw_rows"]
    min_example_ok = len(selected) >= args.min_examples
    redaction_ok = redaction_rate >= args.min_redaction_rate

    manifest = {
        "contract_version": "1.0",
        "inputs": {
            "raw_trace_glob": args.raw_trace_glob,
            "raw_trace_files": raw_files,
            "synthetic_file": args.synthetic_file,
            "adversarial_file": args.adversarial_file,
        },
        "build": {
            "target_examples": args.target_examples,
            "selected_examples": len(selected),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "min_examples": args.min_examples,
            "min_example_ok": min_example_ok,
            "val_ratio": args.val_ratio,
            "ratios": {
                "real": args.ratio_real,
                "synthetic": args.ratio_synthetic,
                "adversarial": args.ratio_adversarial,
            },
            "min_redaction_rate": args.min_redaction_rate,
            "redaction_ok": redaction_ok,
            "redaction_rate": redaction_rate,
        },
        "qc": {
            "raw_rows": stats["raw_rows"],
            "rejected_malformed": stats["rejected_malformed"],
            "rejected_low_signal": stats["rejected_low_signal"],
            "rows_with_redactions": stats["rows_with_redactions"],
            "redactions_applied": stats["redactions"],
            "post_dedup_rows": len(deduped),
            "category_counts": category_counts,
            "outcome_counts": outcome_counts,
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote train dataset: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote val dataset: {val_path} ({len(val_rows)} rows)")
    print(f"Wrote manifest: {manifest_path}")

    if not min_example_ok:
        raise RuntimeError(f"Selected examples {len(selected)} are below minimum {args.min_examples}")
    if not redaction_ok:
        raise RuntimeError(
            f"Redaction rate {redaction_rate:.4f} is below configured minimum {args.min_redaction_rate:.4f}"
        )


if __name__ == "__main__":
    main()
