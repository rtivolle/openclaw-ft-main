---
title: OpenClaw Dataset Builder
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

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

## Hugging Face CLI on Debian (dataset import + jobs)

This repo includes a helper script for `hf` CLI workflows:

- `scripts/hf_cli_workflow.sh setup`: install `hf` CLI.
- `scripts/hf_cli_workflow.sh login`: authenticate with Hugging Face.
- `scripts/hf_cli_workflow.sh import ...`: download dataset repos locally.
- `scripts/hf_cli_workflow.sh run-job ...`: launch remote Hugging Face Jobs.
- `scripts/hf_cli_workflow.sh run-gpu-task ...`: launch the training protocol as a GPU Job.
- `scripts/hf_jobs_tui.py`: terminal UI for dataset creation, model pull, train, test, and HF Jobs operations.

Quick start:

```bash
chmod +x scripts/hf_cli_workflow.sh
scripts/hf_cli_workflow.sh setup
scripts/hf_cli_workflow.sh login
```

Import a dataset repo:

```bash
scripts/hf_cli_workflow.sh import HuggingFaceH4/ultrachat_200k data/hf/ultrachat --include "*.jsonl"
```

Run a remote job:

```bash
scripts/hf_cli_workflow.sh run-job python:3.12 -- python -c "print('hello from hf jobs')" --flavor cpu-basic --detach
scripts/hf_cli_workflow.sh ps -a
scripts/hf_cli_workflow.sh logs <job_id>
```

Run a GPU training protocol job:

```bash
scripts/hf_cli_workflow.sh run-gpu-task \
  --repo-url https://huggingface.co/spaces/<user>/<space-name> \
  --ref main \
  --flavor l4x1 \
  --protocol-env configs/protocol.env \
  --train-config configs/qlora_12gb.env \
  --detach
```

This command uses `hf jobs run --secrets HF_TOKEN`, clones your repo in the
remote container, installs dependencies, and executes:

```bash
python scripts/run_training_protocol.py --protocol-env <path>
```

Launch the TUI:

```bash
python3 scripts/hf_jobs_tui.py
```

TUI covers:

- Hugging Face auth status/login
- Model pull (`hf download`)
- Dataset creation (`fetch_hf_dataset.py`, `build_protocol_dataset.py`)
- Local train/test (`train_qlora_12gb.py`, `eval_baseline_vs_adapter.py`)
- Full local protocol (`run_training_protocol.py`)
- Remote jobs submit/list/logs/inspect/cancel

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

## Deploy as a Hugging Face Space

This repo is configured as a Docker Space. The Space UI runs `app.py`, which
generates OpenClaw-format synthetic datasets in-browser.

1. Create a new Hugging Face Space and choose **Docker** SDK.
2. Push this repository to the Space:
   - `git remote add space https://huggingface.co/spaces/<user>/<space-name>`
   - `git push space main`
3. Wait for the build to finish. The app is served on port `7860`.

Space runtime files added for deployment:

- `Dockerfile`
- `space_requirements.txt`
- `app.py`
