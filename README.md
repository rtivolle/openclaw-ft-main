# OpenCode/OpenClaw Local Training Protocol

## Goal

Train local adapters that improve OpenCode/OpenClaw task success while enforcing dataset quality and release gates.

## Protocol Overview

This repo now supports an end-to-end protocol:

1. Build dataset snapshot from real traces + synthetic + adversarial examples.
2. Apply deterministic redaction and quality filters.
3. Deduplicate and generate train/val splits with a manifest.
4. Train QLoRA adapter.
5. Evaluate baseline vs adapter.
6. Produce promotion decision JSON from configured thresholds.

## Key Files

- `configs/qlora_12gb.env`: base QLoRA training settings.
- `configs/protocol.env`: protocol-level ratios, quality gates, and run defaults.
- `scripts/build_protocol_dataset.py`: dataset creation, QC, redaction, and manifest.
- `scripts/train_qlora_12gb.py`: QLoRA training entrypoint (supports multi-file shards and loader tuning).
- `scripts/eval_baseline_vs_adapter.py`: baseline vs adapter comparison.
- `scripts/run_training_protocol.py`: orchestrates build -> train -> eval -> gate report.

## Data Contracts

### Raw traces (`data/raw_traces/*.jsonl`)

Each row should include:

- `messages` (or `conversation`) with `role` and `content`, or `user` + `assistant`.
- Optional metadata: `source`, `task_category`, `outcome`.

### Processed output per run

Each run writes:

- `artifacts/protocol-runs/<timestamp>/dataset/train.jsonl`
- `artifacts/protocol-runs/<timestamp>/dataset/val.jsonl`
- `artifacts/protocol-runs/<timestamp>/dataset/dataset_manifest.json`
- `artifacts/protocol-runs/<timestamp>/eval/eval_results.json`
- `artifacts/protocol-runs/<timestamp>/eval/gate_decision.json`

## Run Protocol

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install datasets peft transformers trl bitsandbytes accelerate torch
python3 scripts/run_training_protocol.py --protocol-env configs/protocol.env
```

Use `--skip-train` or `--skip-eval` for partial pipeline runs.

## Multi-Disk Loading

In `configs/qlora_12gb.env`, set shard lists across disks:

- `TRAIN_FILES=/mnt/disk1/train_a.jsonl,/mnt/disk2/train_b.jsonl`
- `VAL_FILES=/mnt/disk1/val_a.jsonl,/mnt/disk2/val_b.jsonl`

Optional host-side throughput knobs:

- `DATASET_MAP_NUM_PROC`
- `DATALOADER_NUM_WORKERS`
- `DATALOADER_PREFETCH_FACTOR`
- `DATALOADER_PERSISTENT_WORKERS`
- `DATALOADER_PIN_MEMORY`

## Promotion Gate

`scripts/run_training_protocol.py` promotes only when both are true:

- `delta.exact_match >= GATE_MIN_DELTA_EXACT_MATCH`
- `delta.keyword_recall >= GATE_MIN_DELTA_KEYWORD_RECALL`

And dataset gates in `dataset_manifest.json` must pass:

- `min_example_ok`
- `redaction_ok`

## Daily Loop Recommendation

1. Ingest previous-day traces to `data/raw_traces/`.
2. Run `scripts/run_training_protocol.py`.
3. Review `gate_decision.json`.
4. If blocked, add hard counterexamples and rerun.
