"""Generate diverse OpenClaw training examples using templates (fully local).

Produces high-quality synthetic training data with 5 conversation styles,
50 topics, multi-turn variants, and contextual randomisation.

Usage:
    python scripts/generate_openclaw_examples.py \
        --out data/train_200.jsonl \
        --count-per-style 100 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, TypedDict, cast


class Message(TypedDict):
    role: str
    content: str


class Record(TypedDict):
    source: str
    messages: list[Message]


# ---------------------------------------------------------------------------
# Topics — 50 covering ops, dev, security, data, and infra
# ---------------------------------------------------------------------------

TOPICS = [
    # Original ops topics
    "password reset", "account lockout", "billing mismatch",
    "subscription cancel", "invoice export", "api token rotation",
    "rate limit error", "tool timeout", "json parse failure",
    "build failure", "test flake", "deployment rollback",
    "database migration", "cache invalidation", "search indexing",
    "notification delay", "permissions error", "multi-factor setup",
    "session expiry", "slow response",
    # Security & auth
    "JWT token expiry", "CORS misconfiguration", "SQL injection attempt",
    "XSS vulnerability", "secret rotation", "RBAC policy conflict",
    "certificate pinning failure", "OAuth redirect loop",
    # Infrastructure
    "container OOM kill", "pod crash loop", "DNS resolution failure",
    "load balancer health check", "disk space exhaustion", "network partition",
    "TLS handshake timeout", "CDN cache purge",
    # Data & pipelines
    "data pipeline stall", "schema drift", "ETL job timeout",
    "replication lag", "backup restoration", "data corruption detection",
    # Development
    "merge conflict resolution", "dependency version conflict",
    "circular import", "memory leak detection", "deadlock in transaction",
    "regex catastrophic backtracking", "race condition", "flaky integration test",
    "webpack bundle size", "type mismatch error",
]

# Contextual fragments for realism
SERVICES = ["auth-service", "payment-api", "user-gateway", "analytics-worker",
            "notification-hub", "search-indexer", "billing-engine", "cdn-proxy"]
ERROR_CODES = ["ERR_TIMEOUT_503", "ERR_AUTH_401", "ERR_RATE_429", "ERR_NOT_FOUND_404",
               "ERR_CONFLICT_409", "ERR_INTERNAL_500", "ERR_BAD_GATEWAY_502", "ERR_CONN_REFUSED"]
ENVS = ["production", "staging", "dev", "canary"]
PATHS = ["/var/log/app.log", "/etc/nginx/nginx.conf", "src/handlers/main.py",
         "config/database.yml", ".github/workflows/ci.yml", "k8s/deployment.yaml"]


# ---------------------------------------------------------------------------
# Style 1: Codex — terse, action-oriented
# ---------------------------------------------------------------------------

def codex_style(topic: str, index: int) -> tuple[str, str]:
    openers = ["Help with", "Need a fix for", "Troubleshoot", "Assist on",
               "Quick fix for", "Resolve", "Debug"]
    endings = ["Keep it actionable.", "Make it concise and practical.",
               "Focus on safe recovery.", "Give minimal-risk steps.",
               "Prioritise uptime.", "Avoid data loss."]
    svc = random.choice(SERVICES)
    env = random.choice(ENVS)
    user = f"{random.choice(openers)} {topic} in {svc} ({env}) issue #{index + 1}. {random.choice(endings)}"
    step2 = random.choice([
        f"Check `{random.choice(PATHS)}` for recent config changes and logs.",
        f"Compare recent deploys in {env} against {svc} logs to isolate the root cause.",
        "Validate dependencies and permissions against expected state.",
        f"Inspect {svc} health endpoint and correlate with `{random.choice(ERROR_CODES)}`.",
    ])
    assistant = (
        f"1) Confirm exact error and timestamp from {svc} logs. "
        f"2) {step2} "
        "3) Apply the smallest safe fix, then run validation. "
        f"4) Record root cause and add a monitor for early detection. "
        f"Focus now: {topic} in {env}."
    )
    return user, assistant


# ---------------------------------------------------------------------------
# Style 2: Gemini — guided walkthrough
# ---------------------------------------------------------------------------

def gemini_style(topic: str, index: int) -> tuple[str, str]:
    starters = ["Can you walk me through resolving", "Guide me through fixing",
                "Help me diagnose", "Please break down", "Walk me through debugging"]
    svc = random.choice(SERVICES)
    env = random.choice(ENVS)
    user = f"{random.choice(starters)} {topic} in {svc} on {env} (case {index + 1})?"
    templates = [
        (f"For {topic} in {svc}, begin with symptom, impact, and time window. "
         f"Then test one likely cause at a time, validate with evidence from `{random.choice(PATHS)}`, and keep rollback ready. "
         "After recovery, capture what changed, why it happened, and which monitor should trigger earlier next time."),
        (f"In a {topic} incident on {env}, first scope blast radius and affected users. "
         f"Run one hypothesis at a time with observable checks against {svc}, apply the least disruptive remediation, "
         "then document prevention controls and alerts."),
        (f"Start by checking if {topic} is isolated to {svc} or cross-service. "
         f"Gather evidence: recent deploys, config diffs, and error rates. "
         f"Narrow to root cause, fix with minimal blast radius, verify in {env}, then update runbook."),
    ]
    return user, random.choice(templates)


# ---------------------------------------------------------------------------
# Style 3: Debug — error-message-first problem solving
# ---------------------------------------------------------------------------

def debug_style(topic: str, index: int) -> tuple[str, str]:
    err = random.choice(ERROR_CODES)
    svc = random.choice(SERVICES)
    path = random.choice(PATHS)
    env = random.choice(ENVS)
    user = (
        f"I'm getting `{err}` in {svc} on {env} related to {topic}. "
        f"Stack trace points to `{path}`. What should I check?"
    )
    assistant = (
        f"**Error `{err}` in {svc}**\n\n"
        f"1. Check `{path}` at the line mentioned in the stack trace — look for recent changes in git blame.\n"
        f"2. Verify {svc} config hasn't drifted from expected state: `diff {path} {path}.bak`\n"
        f"3. Check if this correlates with a recent deploy: `git log --oneline -5`\n"
        f"4. If the error is transient, check resource limits (CPU/memory) on the {env} pod.\n"
        f"5. Apply fix, verify with `curl -I https://{svc}.internal/health`, then monitor for 15 minutes."
    )
    return user, assistant


# ---------------------------------------------------------------------------
# Style 4: Explain — conceptual understanding
# ---------------------------------------------------------------------------

def explain_style(topic: str, index: int) -> tuple[str, str]:
    starters = ["Explain how", "What causes", "Why does", "Help me understand",
                "What's the root cause behind"]
    user = f"{random.choice(starters)} {topic} happens and how to prevent it?"
    svc = random.choice(SERVICES)
    assistant = (
        f"**{topic.title()}** typically occurs when:\n\n"
        f"- **Immediate cause**: A misconfiguration or resource exhaustion in services like {svc}.\n"
        f"- **Contributing factors**: Missing validation, stale caches, or deployment race conditions.\n"
        f"- **Detection**: Set up alerts on error rate spikes and latency P99 in your monitoring stack.\n\n"
        f"**Prevention checklist**:\n"
        f"1. Add input validation at the boundary layer\n"
        f"2. Implement circuit breakers between dependent services\n"
        f"3. Use canary deployments to catch issues before full rollout\n"
        f"4. Add integration tests that specifically cover {topic} scenarios"
    )
    return user, assistant


# ---------------------------------------------------------------------------
# Style 5: Refactor — code improvement advice
# ---------------------------------------------------------------------------

def refactor_style(topic: str, index: int) -> tuple[str, str]:
    path = random.choice(PATHS)
    user = (
        f"The code handling {topic} in `{path}` is getting complex and hard to test. "
        f"How should I refactor it?"
    )
    assistant = (
        f"**Refactoring {topic} logic in `{path}`**:\n\n"
        f"1. **Extract**: Pull the {topic} handling into its own module/class with a clear interface.\n"
        f"2. **Inject dependencies**: Pass config and clients as constructor args instead of importing globals.\n"
        f"3. **Add a retry wrapper**: Wrap the flaky parts (network calls, DB queries) in a retry decorator with exponential backoff.\n"
        f"4. **Test seams**: Create a mock/stub for external dependencies so you can unit test the core logic.\n"
        f"5. **Guard clause**: Replace nested conditionals with early returns for readability.\n\n"
        f"Start with step 1 — get it into a separate file, keep the existing tests passing, then improve incrementally."
    )
    return user, assistant


# ---------------------------------------------------------------------------
# Multi-turn style: Follow-up conversation
# ---------------------------------------------------------------------------

def multiturn_style(topic: str, index: int) -> list[Message]:
    """Generate a 2-turn conversation (user asks, assistant answers, user follows up)."""
    svc = random.choice(SERVICES)
    env = random.choice(ENVS)
    err = random.choice(ERROR_CODES)

    messages: list[Message] = [
        {"role": "system", "content": "You are OpenClaw assistant. Be accurate, concise, and safe."},
        {"role": "user", "content": f"We're seeing {topic} issues in {svc} on {env}. Getting `{err}` intermittently. Where do I start?"},
        {"role": "assistant", "content": (
            f"Start with these checks:\n"
            f"1. Check {svc} error logs for the last hour — filter by `{err}`\n"
            f"2. Verify no recent config or deployment changes in {env}\n"
            f"3. Check resource utilisation (CPU, memory, connections) on affected pods\n"
            f"Let me know what you find and I'll help narrow it down."
        )},
        {"role": "user", "content": f"Found that memory usage spiked right before the errors started. The pod is at 95% memory."},
        {"role": "assistant", "content": (
            f"Memory spike + `{err}` strongly suggests an OOM-related issue. Here's your action plan:\n\n"
            f"1. **Immediate**: Restart the affected pods or scale horizontally: `kubectl rollout restart deployment/{svc}`\n"
            f"2. **Investigate**: Check for memory leaks — look at heap dumps or profile with `pprof`\n"
            f"3. **Prevent**: Set memory limits in the pod spec, add a memory usage alert at 80%\n"
            f"4. **Verify**: Monitor the new pods for 30 minutes to confirm stability"
        )},
    ]
    return messages


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

STYLE_FNS = {
    "codex5.3": codex_style,
    "gemini-pro-3": gemini_style,
    "debug-trace": debug_style,
    "explain-mode": explain_style,
    "refactor-guide": refactor_style,
}


def build_record(source: str, user: str, assistant: str) -> Record:
    return {
        "source": source,
        "messages": [
            {"role": "system", "content": "You are OpenClaw assistant. Be accurate, concise, and safe."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def generate(count_per_style: int) -> list[Record]:
    rows: list[Record] = []

    # Single-turn styles
    for style_name, style_fn in STYLE_FNS.items():
        for i in range(count_per_style):
            topic = TOPICS[i % len(TOPICS)]
            user, assistant = style_fn(topic, i)
            rows.append(build_record(style_name, user, assistant))

    # Multi-turn conversations (20% of count_per_style)
    multiturn_count = max(1, count_per_style // 5)
    for i in range(multiturn_count):
        topic = TOPICS[i % len(TOPICS)]
        messages = multiturn_style(topic, i)
        rows.append({"source": "multiturn-debug", "messages": messages})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenClaw training examples (local templates).")
    parser.add_argument("--out", default="data/train_200.jsonl")
    parser.add_argument("--count-per-style", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args()
    out = cast(str, getattr(parsed, "out"))
    count_per_style = int(cast(int, getattr(parsed, "count_per_style")))
    seed = int(cast(int, getattr(parsed, "seed")))
    random.seed(seed)

    rows = generate(count_per_style)
    random.shuffle(rows)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    style_counts: dict[str, int] = {}
    for row in rows:
        src = row["source"]
        style_counts[src] = style_counts.get(src, 0) + 1

    print(f"Wrote {len(rows)} examples to {out_path}")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count}")


if __name__ == "__main__":
    main()
