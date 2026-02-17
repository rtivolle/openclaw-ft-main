"""Generate training data using Gemini CLI and Codex CLI.

Drives both locally-installed CLIs in non-interactive mode to produce
high-quality OpenClaw/OpenCode training examples.

Requirements:
    - gemini CLI: brew install gemini
    - codex CLI: brew install codex

Usage:
    python scripts/generate_from_cli.py --count 100 --out data/cli_generated.jsonl
    python scripts/generate_from_cli.py --count 50 --gemini-only
    python scripts/generate_from_cli.py --count 50 --codex-only
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = "You are OpenClaw assistant. Be accurate, concise, and safe."

TOPICS = [
    "password reset flow", "account lockout recovery", "billing mismatch investigation",
    "subscription cancellation", "API token rotation", "rate limit mitigation",
    "tool timeout debugging", "JSON parse failure", "CI build failure",
    "flaky test investigation", "deployment rollback", "database migration",
    "cache invalidation strategy", "permissions error diagnosis", "session management",
    "container OOM debugging", "pod crash loop", "DNS resolution failure",
    "TLS certificate renewal", "memory leak detection", "merge conflict resolution",
    "dependency version conflict", "circular import fix", "deadlock debugging",
    "race condition diagnosis", "schema drift detection", "ETL pipeline failure",
    "replication lag investigation", "secret rotation procedure", "webhook retry storm",
    "load balancer misconfiguration", "CORS policy debugging", "OAuth flow troubleshooting",
    "log aggregation setup", "alerting threshold tuning", "canary deployment validation",
    "feature flag rollout", "API versioning strategy", "database connection pooling",
    "queue backlog management", "service mesh configuration",
]

SCENARIOS = [
    "in a production Kubernetes cluster",
    "on a staging server after a recent deploy",
    "during a CI/CD pipeline run",
    "in a microservices architecture",
    "on a legacy monolith being migrated",
    "in a multi-tenant SaaS platform",
    "after a failed database migration",
    "during an on-call incident",
]

PROMPT_TEMPLATE = """Generate a realistic support conversation for an AI coding assistant called OpenClaw.

Topic: {topic} {scenario}

Requirements:
1. The user message should describe a specific, realistic scenario with technical details (error messages, file paths, commands)
2. The assistant response must be actionable, concise, and structured with numbered steps (3-6 steps)
3. Include specific technical details where appropriate
4. Vary the tone naturally

Return ONLY a JSON object with this exact structure (no markdown fences, no extra text):
{{"user": "the user's question here", "assistant": "the assistant's response here"}}"""


def call_gemini(prompt: str, timeout: int = 60) -> dict[str, str] | None:
    """Call Gemini CLI in headless mode with JSON output."""
    try:
        result = subprocess.run(
            ["gemini", "-p", prompt, "-o", "text"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"    ⚠ Gemini CLI error: {result.stderr[:200]}")
            return None

        text = result.stdout.strip()
        return _parse_json_response(text)

    except subprocess.TimeoutExpired:
        print("    ⚠ Gemini CLI timed out")
        return None
    except FileNotFoundError:
        print("    ⚠ Gemini CLI not found — install with: brew install gemini")
        return None
    except Exception as exc:
        print(f"    ⚠ Gemini error: {exc}")
        return None


def call_codex(prompt: str, timeout: int = 60) -> dict[str, str] | None:
    """Call Codex CLI in non-interactive exec mode."""
    try:
        result = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"    ⚠ Codex CLI error: {result.stderr[:200]}")
            return None

        text = result.stdout.strip()
        return _parse_json_response(text)

    except subprocess.TimeoutExpired:
        print("    ⚠ Codex CLI timed out")
        return None
    except FileNotFoundError:
        print("    ⚠ Codex CLI not found — install with: brew install codex")
        return None
    except Exception as exc:
        print(f"    ⚠ Codex error: {exc}")
        return None


def _parse_json_response(text: str) -> dict[str, str] | None:
    """Parse JSON from CLI output, handling markdown fences and extra text."""
    if not text:
        return None

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        cleaned = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not line.strip().startswith("```"):
                cleaned.append(line)
        text = "\n".join(cleaned).strip()

    # Find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None

    try:
        parsed = json.loads(text[start:end])
        if isinstance(parsed, dict) and "user" in parsed and "assistant" in parsed:
            return {"user": str(parsed["user"]), "assistant": str(parsed["assistant"])}
    except json.JSONDecodeError:
        pass

    return None


def generate_examples(
    count: int,
    use_gemini: bool,
    use_codex: bool,
    timeout: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate training examples by calling CLI tools."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    errors = 0
    max_errors = max(10, count // 3)

    # Build list of (cli_name, call_fn) pairs
    cli_pool: list[tuple[str, Any]] = []
    if use_gemini:
        cli_pool.append(("gemini", call_gemini))
    if use_codex:
        cli_pool.append(("codex", call_codex))

    if not cli_pool:
        print("❌ No CLI tools selected. Use --gemini-only or --codex-only, or enable both (default).")
        return []

    print(f"\n🤖 Generating {count} examples using: {', '.join(name for name, _ in cli_pool)}")

    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        scenario = rng.choice(SCENARIOS)
        prompt = PROMPT_TEMPLATE.format(topic=topic, scenario=scenario)

        # Alternate between CLIs
        cli_name, call_fn = cli_pool[i % len(cli_pool)]

        # Retry with backoff
        result = None
        for attempt in range(3):
            result = call_fn(prompt, timeout=timeout)
            if result is not None:
                break
            wait = (2 ** attempt) + rng.random()
            print(f"    Retry {attempt + 1}/3 for {cli_name} in {wait:.1f}s ...")
            time.sleep(wait)

        if result is None:
            errors += 1
            if errors >= max_errors:
                print(f"  ❌ Too many errors ({errors}), stopping early.")
                break
            continue

        row: dict[str, Any] = {
            "source": f"cli:{cli_name}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": result["user"]},
                {"role": "assistant", "content": result["assistant"]},
            ],
        }
        rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  📝 {i + 1}/{count} generated ({errors} errors)")

    print(f"  ✅ {len(rows)} examples generated, {errors} errors")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data using Gemini CLI and Codex CLI."
    )
    parser.add_argument(
        "--count", type=int,
        default=int(os.getenv("CLI_GEN_COUNT", "100")),
        help="Total number of examples to generate.",
    )
    parser.add_argument(
        "--out",
        default=os.getenv("CLI_GEN_FILE", "data/cli_generated.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument("--gemini-only", action="store_true", help="Only use Gemini CLI.")
    parser.add_argument("--codex-only", action="store_true", help="Only use Codex CLI.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per CLI call in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine which CLIs to use
    use_gemini = True
    use_codex = True
    if args.gemini_only:
        use_codex = False
    elif args.codex_only:
        use_gemini = False

    # Check CLI availability
    if use_gemini and not shutil.which("gemini"):
        print("⚠ gemini CLI not found in PATH. Falling back to codex-only.")
        use_gemini = False
    if use_codex and not shutil.which("codex"):
        print("⚠ codex CLI not found in PATH. Falling back to gemini-only.")
        use_codex = False

    rows = generate_examples(args.count, use_gemini, use_codex, args.timeout, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Print summary
    source_counts: dict[str, int] = {}
    for row in rows:
        src = row["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n🎉 Wrote {len(rows)} examples to {out_path}")
    for src, cnt in sorted(source_counts.items()):
        print(f"   {src}: {cnt}")


if __name__ == "__main__":
    main()
