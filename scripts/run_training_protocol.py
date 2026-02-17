import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full OpenCode/OpenClaw training protocol.")
    parser.add_argument("--protocol-env", default="configs/protocol.env")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    protocol_env = parse_env_file(Path(args.protocol_env))
    merged_env = dict(os.environ)
    merged_env.update(protocol_env)

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_root = Path(protocol_env.get("OUTPUT_ROOT", "artifacts/protocol-runs"))
    run_dir = output_root / ts
    dataset_dir = run_dir / "dataset"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    manifest_file = dataset_dir / "dataset_manifest.json"
    model_out = run_dir / "adapter"
    eval_out = eval_dir / "eval_results.json"
    gate_out = eval_dir / "gate_decision.json"

    build_cmd = [
        "python3",
        "scripts/build_protocol_dataset.py",
        "--raw-trace-glob",
        protocol_env.get("RAW_TRACE_GLOB", "data/raw_traces/*.jsonl"),
        "--synthetic-file",
        protocol_env.get("SYNTHETIC_FILE", "data/train_200.jsonl"),
        "--adversarial-file",
        protocol_env.get("ADVERSARIAL_FILE", "data/adversarial.jsonl"),
        "--target-examples",
        protocol_env.get("TARGET_EXAMPLES", "20000"),
        "--min-examples",
        protocol_env.get("MIN_EXAMPLES", "2000"),
        "--val-ratio",
        protocol_env.get("VAL_RATIO", "0.1"),
        "--ratio-real",
        protocol_env.get("RATIO_REAL", "0.6"),
        "--ratio-synthetic",
        protocol_env.get("RATIO_SYNTHETIC", "0.3"),
        "--ratio-adversarial",
        protocol_env.get("RATIO_ADVERSARIAL", "0.1"),
        "--max-per-source",
        protocol_env.get("MAX_PER_SOURCE", "4000"),
        "--seed",
        protocol_env.get("SEED", "42"),
        "--train-out",
        str(train_file),
        "--val-out",
        str(val_file),
        "--manifest-out",
        str(manifest_file),
        "--min-redaction-rate",
        protocol_env.get("DATASET_MANIFEST_MIN_REDACTION_RATE", "0.0"),
    ]
    t0 = time.time()
    run_command(build_cmd, merged_env)
    dataset_duration = time.time() - t0
    print(f"Dataset build completed in {dataset_duration:.1f}s")

    training_env = dict(merged_env)
    training_env.update(parse_env_file(Path(protocol_env.get("TRAIN_CONFIG", "configs/qlora_12gb.env"))))
    training_env["BASE_MODEL"] = protocol_env.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    training_env["TRAIN_FILE"] = str(train_file)
    training_env["VAL_FILE"] = str(val_file)
    training_env["OUTPUT_DIR"] = str(model_out)

    train_duration = 0.0
    if not args.skip_train:
        t1 = time.time()
        run_command(["python3", "scripts/train_qlora_12gb.py"], training_env)
        train_duration = time.time() - t1
        print(f"Training completed in {train_duration:.1f}s")

    eval_duration = 0.0
    if not args.skip_eval:
        eval_cmd = [
            "python3",
            "scripts/eval_baseline_vs_adapter.py",
            "--base-model",
            training_env["BASE_MODEL"],
            "--adapter-dir",
            str(model_out),
            "--eval-file",
            str(val_file),
            "--max-new-tokens",
            protocol_env.get("MAX_NEW_TOKENS", "256"),
            "--out",
            str(eval_out),
        ]
        t2 = time.time()
        run_command(eval_cmd, training_env)
        eval_duration = time.time() - t2
        print(f"Evaluation completed in {eval_duration:.1f}s")

    eval_result = json.loads(eval_out.read_text(encoding="utf-8")) if eval_out.exists() else {}
    min_exact = float(protocol_env.get("GATE_MIN_DELTA_EXACT_MATCH", "0.05"))
    min_recall = float(protocol_env.get("GATE_MIN_DELTA_KEYWORD_RECALL", "0.05"))
    delta = eval_result.get("delta", {})
    exact_ok = float(delta.get("exact_match", -1.0)) >= min_exact
    recall_ok = float(delta.get("keyword_recall", -1.0)) >= min_recall

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    dataset_ok = manifest.get("build", {}).get("min_example_ok", False) and manifest.get("build", {}).get("redaction_ok", False)
    promotion_pass = exact_ok and recall_ok and dataset_ok
    gate = {
        "timestamp_utc": ts,
        "run_dir": str(run_dir),
        "dataset_manifest": str(manifest_file),
        "eval_result": str(eval_out),
        "thresholds": {
            "min_delta_exact_match": min_exact,
            "min_delta_keyword_recall": min_recall,
        },
        "checks": {
            "dataset_min_examples_ok": manifest.get("build", {}).get("min_example_ok", False),
            "dataset_redaction_ok": manifest.get("build", {}).get("redaction_ok", False),
            "exact_delta_ok": exact_ok,
            "keyword_recall_delta_ok": recall_ok,
        },
        "is_real_improvement": eval_result.get("is_real_improvement", False),
        "promote": promotion_pass,
        "durations_seconds": {
            "dataset_build": round(dataset_duration, 1),
            "training": round(train_duration, 1),
            "evaluation": round(eval_duration, 1),
        },
    }
    gate_out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Wrote gate decision: {gate_out}")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
