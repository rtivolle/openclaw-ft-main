import argparse
import json
import random
from pathlib import Path
from typing import TypedDict, cast


class Message(TypedDict):
    role: str
    content: str


class Record(TypedDict):
    source: str
    messages: list[Message]


class Args(TypedDict):
    out: str
    count_per_style: int
    seed: int

TOPICS = [
    "password reset",
    "account lockout",
    "billing mismatch",
    "subscription cancel",
    "invoice export",
    "api token rotation",
    "rate limit error",
    "tool timeout",
    "json parse failure",
    "build failure",
    "test flake",
    "deployment rollback",
    "database migration",
    "cache invalidation",
    "search indexing",
    "notification delay",
    "permissions error",
    "multi-factor setup",
    "session expiry",
    "slow response",
]


def codex_style(topic: str, index: int) -> tuple[str, str]:
    openers = [
        "Help with",
        "Need a fix for",
        "Troubleshoot",
        "Assist on",
    ]
    endings = [
        "Keep it actionable.",
        "Make it concise and practical.",
        "Focus on safe recovery.",
        "Give minimal-risk steps.",
    ]
    user = f"{random.choice(openers)} {topic} issue #{index + 1}. {random.choice(endings)}"
    step2 = random.choice(
        [
            "Check recent config changes and logs for correlated failures.",
            "Compare recent deploys against logs to isolate the first bad change.",
            "Validate dependencies and permissions against expected state.",
        ]
    )
    assistant = (
        "1) Confirm exact error and timestamp. "
        f"2) {step2} "
        "3) Apply the smallest safe fix, then run validation. "
        "4) Record root cause and guardrails. "
        f"Focus now: {topic}."
    )
    return user, assistant


def gemini_style(topic: str, index: int) -> tuple[str, str]:
    starter = random.choice(
        [
            "Can you walk me through resolving",
            "Guide me through fixing",
            "Help me diagnose",
            "Please break down",
        ]
    )
    user = f"{starter} {topic} case {index + 1} with context?"
    assistant = random.choice(
        [
            (
                f"For {topic}, begin with symptom, impact, and time window. "
                "Then test one likely cause at a time, validate with evidence, and keep rollback ready. "
                "After recovery, capture what changed, why it happened, and which monitor should trigger earlier next time."
            ),
            (
                f"In a {topic} incident, first scope blast radius and affected users. "
                "Run one hypothesis at a time with observable checks, apply the least disruptive remediation, "
                "then document prevention controls and alerts."
            ),
        ]
    )
    return user, assistant


def build_record(source: str, user: str, assistant: str) -> Record:
    return {
        "source": source,
        "messages": [
            {
                "role": "system",
                "content": "You are OpenClaw assistant. Be accurate, concise, and safe.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def generate(count_per_style: int) -> list[Record]:
    rows: list[Record] = []

    for i in range(count_per_style):
        topic = TOPICS[i % len(TOPICS)]
        user, assistant = codex_style(topic, i)
        rows.append(build_record("codex5.3", user, assistant))

    for i in range(count_per_style):
        topic = TOPICS[(i * 3) % len(TOPICS)]
        user, assistant = gemini_style(topic, i)
        rows.append(build_record("gemini-pro-3", user, assistant))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenClaw training examples.")
    _ = parser.add_argument("--out", default="openclaw-ft/data/train_200.jsonl")
    _ = parser.add_argument("--count-per-style", type=int, default=100)
    _ = parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args()
    out = cast(str, getattr(parsed, "out"))
    count_per_style = int(cast(int, getattr(parsed, "count_per_style")))
    seed = int(cast(int, getattr(parsed, "seed")))
    args: Args = {"out": out, "count_per_style": count_per_style, "seed": seed}
    random.seed(args["seed"])

    rows = generate(args["count_per_style"])

    out_path = Path(args["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            _ = handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
