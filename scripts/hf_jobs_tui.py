#!/usr/bin/env python3
import curses
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SCRIPT = str((Path(__file__).resolve().parent / "hf_cli_workflow.sh"))
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
LMS_BIN = os.environ.get("LMS_BIN", "lms")


@dataclass
class Action:
    label: str
    fn: Callable[[curses.window], None]


def run_capture(cmd: list[str], env: dict[str, str] | None = None, cwd: str | None = None) -> tuple[int, str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=merged_env, cwd=cwd)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip() or "(no output)"


def read_input(stdscr: curses.window, prompt: str, default: str = "") -> str:
    curses.echo()
    stdscr.clear()
    stdscr.addstr(0, 0, prompt)
    if default:
        stdscr.addstr(1, 0, f"Default: {default}")
    stdscr.addstr(3, 0, "> ")
    stdscr.refresh()
    raw = stdscr.getstr(3, 2).decode("utf-8", errors="replace").strip()
    curses.noecho()
    return raw or default


def show_output(stdscr: curses.window, title: str, body: str) -> None:
    lines = [title, ""] + body.splitlines()
    top = 0
    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        view_h = max(1, max_y - 2)
        for idx in range(view_h):
            line_idx = top + idx
            if line_idx >= len(lines):
                break
            stdscr.addnstr(idx, 0, lines[line_idx], max_x - 1)
        stdscr.addnstr(max_y - 1, 0, "j/k scroll, q back", max_x - 1)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch in (ord("j"), curses.KEY_DOWN) and top + view_h < len(lines):
            top += 1
        if ch in (ord("k"), curses.KEY_UP) and top > 0:
            top -= 1


def action_whoami(stdscr: curses.window) -> None:
    token = read_input(stdscr, "HF token for login")
    code, output = run_capture(["hf", "auth", "login", "--token", token], cwd=REPO_ROOT)
    title = "HF token login completed" if code == 0 else f"HF token login failed (exit={code})"
    show_output(stdscr, title, output)


def action_status(stdscr: curses.window) -> None:
    code, output = run_capture(["hf", "auth", "whoami"], cwd=REPO_ROOT)
    title = "Auth status" if code == 0 else f"Auth check failed (exit={code})"
    show_output(stdscr, title, output)


def action_pull_model(stdscr: curses.window) -> None:
    model_id = read_input(stdscr, "Model repo id (e.g. deepseek-ai/deepseek-llm-7b-chat)")
    default_dir = f"/mnt/models/models/{model_id.split('/')[-1]}"
    local_dir = read_input(stdscr, "Local model dir", default_dir)
    hf_home = read_input(stdscr, "HF_HOME cache dir", "/mnt/models/hf_cache")
    include = read_input(stdscr, "Include glob (blank for all)", "")

    cmd = ["hf", "download", model_id, "--local-dir", local_dir]
    if include:
        cmd.extend(["--include", include])
    code, output = run_capture(cmd, env={"HF_HOME": hf_home}, cwd=REPO_ROOT)
    title = "Model download completed" if code == 0 else f"Model download failed (exit={code})"
    show_output(stdscr, title, output)


def action_lms_pull_model(stdscr: curses.window) -> None:
    model_id = read_input(stdscr, "LM Studio model id to pull")
    code, output = run_capture([LMS_BIN, "pull", model_id], cwd=REPO_ROOT)
    title = "LM Studio model pull completed" if code == 0 else f"LM Studio model pull failed (exit={code})"
    show_output(stdscr, title, output)


def action_lms_run_model(stdscr: curses.window) -> None:
    model_id = read_input(stdscr, "LM Studio model id to run")
    prompt = read_input(stdscr, "Prompt (required for non-interactive run)")
    if not prompt:
        show_output(stdscr, "LM Studio run cancelled", "Prompt is required to avoid interactive TUI blocking.")
        return
    code, output = run_capture([LMS_BIN, "run", model_id, prompt], cwd=REPO_ROOT)
    title = "LM Studio model run completed" if code == 0 else f"LM Studio model run failed (exit={code})"
    show_output(stdscr, title, output)


def action_fetch_hf_dataset(stdscr: curses.window) -> None:
    datasets = read_input(
        stdscr,
        "HF datasets CSV",
        "Open-Orca/OpenOrca,teknium/OpenHermes-2.5,sahil2801/CodeAlpaca-20k",
    )
    max_per = read_input(stdscr, "Max rows per dataset", "2000")
    out_file = read_input(stdscr, "Output jsonl", "data/hf_combined.jsonl")
    seed = read_input(stdscr, "Seed", "42")
    cmd = [
        "python3",
        "scripts/fetch_hf_dataset.py",
        "--datasets",
        datasets,
        "--max-per-dataset",
        max_per,
        "--out",
        out_file,
        "--seed",
        seed,
    ]
    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "HF dataset fetch completed" if code == 0 else f"HF dataset fetch failed (exit={code})"
    show_output(stdscr, title, output)


def action_build_dataset(stdscr: curses.window) -> None:
    raw_glob = read_input(stdscr, "Raw traces glob", "data/raw_traces/*.jsonl")
    synthetic = read_input(stdscr, "Synthetic file", "data/train_200.jsonl")
    adversarial = read_input(stdscr, "Adversarial file", "data/adversarial.jsonl")
    hf_file = read_input(stdscr, "HF merged file", "data/hf_combined.jsonl")
    train_out = read_input(stdscr, "Train output", "artifacts/local/train.jsonl")
    val_out = read_input(stdscr, "Val output", "artifacts/local/val.jsonl")
    manifest = read_input(stdscr, "Manifest output", "artifacts/local/dataset_manifest.json")
    target = read_input(stdscr, "Target examples", "20000")
    min_examples = read_input(stdscr, "Min examples", "2000")
    seed = read_input(stdscr, "Seed", "42")
    cmd = [
        "python3",
        "scripts/build_protocol_dataset.py",
        "--raw-trace-glob",
        raw_glob,
        "--synthetic-file",
        synthetic,
        "--adversarial-file",
        adversarial,
        "--hf-file",
        hf_file,
        "--target-examples",
        target,
        "--min-examples",
        min_examples,
        "--seed",
        seed,
        "--train-out",
        train_out,
        "--val-out",
        val_out,
        "--manifest-out",
        manifest,
    ]
    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "Dataset build completed" if code == 0 else f"Dataset build failed (exit={code})"
    show_output(stdscr, title, output)


def action_train_local(stdscr: curses.window) -> None:
    base_model = read_input(stdscr, "Base model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    train_file = read_input(stdscr, "Train file", "artifacts/local/train.jsonl")
    val_file = read_input(stdscr, "Val file", "artifacts/local/val.jsonl")
    output_dir = read_input(stdscr, "Adapter output dir", "artifacts/local/adapter")
    cmd = ["python3", "scripts/train_qlora_12gb.py"]
    env = {
        "BASE_MODEL": base_model,
        "TRAIN_FILE": train_file,
        "VAL_FILE": val_file,
        "OUTPUT_DIR": output_dir,
    }
    code, output = run_capture(cmd, env=env, cwd=REPO_ROOT)
    title = "Training completed" if code == 0 else f"Training failed (exit={code})"
    show_output(stdscr, title, output)


def action_test_eval(stdscr: curses.window) -> None:
    base_model = read_input(stdscr, "Base model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    adapter_dir = read_input(stdscr, "Adapter dir", "artifacts/local/adapter")
    eval_file = read_input(stdscr, "Eval jsonl file", "artifacts/local/val.jsonl")
    max_new_tokens = read_input(stdscr, "Max new tokens", "256")
    batch_size = read_input(stdscr, "Batch size", "4")
    out_file = read_input(stdscr, "Eval result json", "artifacts/local/eval_results.json")
    cmd = [
        "python3",
        "scripts/eval_baseline_vs_adapter.py",
        "--base-model",
        base_model,
        "--adapter-dir",
        adapter_dir,
        "--eval-file",
        eval_file,
        "--max-new-tokens",
        max_new_tokens,
        "--batch-size",
        batch_size,
        "--out",
        out_file,
    ]
    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "Evaluation completed" if code == 0 else f"Evaluation failed (exit={code})"
    show_output(stdscr, title, output)


def action_run_protocol(stdscr: curses.window) -> None:
    protocol_env = read_input(stdscr, "Protocol env", "configs/protocol.env")
    train_config = read_input(stdscr, "Train config", "configs/qlora_12gb.env")
    skip_hf = read_input(stdscr, "Skip HF fetch? (y/n)", "n").lower() in {"y", "yes", "1", "true"}
    skip_gen = read_input(stdscr, "Skip local generation? (y/n)", "n").lower() in {"y", "yes", "1", "true"}
    skip_train = read_input(stdscr, "Skip train? (y/n)", "n").lower() in {"y", "yes", "1", "true"}
    skip_eval = read_input(stdscr, "Skip eval? (y/n)", "n").lower() in {"y", "yes", "1", "true"}

    cmd = [
        "python3",
        "scripts/run_training_protocol.py",
        "--protocol-env",
        protocol_env,
        "--train-config",
        train_config,
    ]
    if skip_hf:
        cmd.append("--skip-hf-fetch")
    if skip_gen:
        cmd.append("--skip-local-gen")
    if skip_train:
        cmd.append("--skip-train")
    if skip_eval:
        cmd.append("--skip-eval")

    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "Protocol run completed" if code == 0 else f"Protocol run failed (exit={code})"
    show_output(stdscr, title, output)


def action_submit_gpu(stdscr: curses.window) -> None:
    repo_url = read_input(stdscr, "Repository URL required by run-gpu-task")
    ref = read_input(stdscr, "Git ref (branch/tag/sha)", "main")
    flavor = read_input(stdscr, "GPU flavor", "l4x1")
    protocol_env = read_input(stdscr, "Protocol env path", "configs/protocol.env")
    train_config = read_input(stdscr, "Train config path", "configs/qlora_12gb.env")
    timeout = read_input(stdscr, "Timeout (blank for default)", "")
    detach = read_input(stdscr, "Detach? (y/n)", "y").lower() in {"y", "yes", "1", "true"}

    cmd = [
        SCRIPT,
        "run-gpu-task",
        "--repo-url",
        repo_url,
        "--ref",
        ref,
        "--flavor",
        flavor,
        "--protocol-env",
        protocol_env,
        "--train-config",
        train_config,
    ]
    if timeout:
        cmd.extend(["--timeout", timeout])
    if detach:
        cmd.append("--detach")

    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "GPU task submitted" if code == 0 else f"GPU task submit failed (exit={code})"
    show_output(stdscr, title, output)


def action_ps(stdscr: curses.window) -> None:
    include_all = read_input(stdscr, "Include completed jobs? (y/n)", "y").lower() in {"y", "yes", "1", "true"}
    cmd = [SCRIPT, "ps"]
    if include_all:
        cmd.append("-a")
    code, output = run_capture(cmd, cwd=REPO_ROOT)
    title = "Job list" if code == 0 else f"Job list failed (exit={code})"
    show_output(stdscr, title, output)


def action_logs(stdscr: curses.window) -> None:
    job_id = read_input(stdscr, "Job ID for logs")
    code, output = run_capture([SCRIPT, "logs", job_id], cwd=REPO_ROOT)
    title = f"Logs: {job_id}" if code == 0 else f"Logs failed (exit={code})"
    show_output(stdscr, title, output)


def action_inspect(stdscr: curses.window) -> None:
    job_id = read_input(stdscr, "Job ID for inspect")
    code, output = run_capture([SCRIPT, "inspect", job_id], cwd=REPO_ROOT)
    title = f"Inspect: {job_id}" if code == 0 else f"Inspect failed (exit={code})"
    show_output(stdscr, title, output)


def action_cancel(stdscr: curses.window) -> None:
    job_id = read_input(stdscr, "Job ID to cancel")
    code, output = run_capture([SCRIPT, "cancel", job_id], cwd=REPO_ROOT)
    title = f"Cancel: {job_id}" if code == 0 else f"Cancel failed (exit={code})"
    show_output(stdscr, title, output)


def draw_menu(stdscr: curses.window, actions: list[Action], idx: int) -> None:
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()
    title = "OpenClaw Training + HF Jobs TUI"
    stdscr.addnstr(0, 0, title, max_x - 1, curses.A_BOLD)
    stdscr.addnstr(1, 0, "Arrows/jk: move  Enter: select  q: quit", max_x - 1)
    for i, action in enumerate(actions):
        attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
        stdscr.addnstr(3 + i, 0, action.label, max_x - 1, attr)
    stdscr.refresh()


def main(stdscr: curses.window) -> None:
    curses.curs_set(0)
    actions = [
        Action("Login (hf auth login)", action_whoami),
        Action("Auth status (hf auth whoami)", action_status),
        Action("Pull model from Hugging Face", action_pull_model),
        Action("Pull model via LM Studio CLI (lms pull)", action_lms_pull_model),
        Action("Run model via LM Studio CLI (lms run)", action_lms_run_model),
        Action("Create dataset: fetch HF datasets", action_fetch_hf_dataset),
        Action("Create dataset: build protocol train/val", action_build_dataset),
        Action("Train (local QLoRA)", action_train_local),
        Action("Test/Eval (baseline vs adapter)", action_test_eval),
        Action("Run full local protocol", action_run_protocol),
        Action("Submit GPU training protocol job", action_submit_gpu),
        Action("List jobs", action_ps),
        Action("Show job logs", action_logs),
        Action("Inspect job", action_inspect),
        Action("Cancel job", action_cancel),
        Action("Quit", lambda _s: None),
    ]
    idx = 0
    while True:
        draw_menu(stdscr, actions, idx)
        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            return
        if ch in (curses.KEY_DOWN, ord("j")) and idx < len(actions) - 1:
            idx += 1
            continue
        if ch in (curses.KEY_UP, ord("k")) and idx > 0:
            idx -= 1
            continue
        if ch in (10, 13, curses.KEY_ENTER):
            if actions[idx].label == "Quit":
                return
            actions[idx].fn(stdscr)


if __name__ == "__main__":
    curses.wrapper(main)
