#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Hugging Face CLI workflow helper for Debian.

Usage:
  scripts/hf_cli_workflow.sh setup
  scripts/hf_cli_workflow.sh login
  scripts/hf_cli_workflow.sh import <dataset_repo> <local_dir> [--revision REV] [--include PATTERN]
  scripts/hf_cli_workflow.sh run-job <image> -- <command...> [--flavor FLAVOR] [--namespace NS] [--detach]
  scripts/hf_cli_workflow.sh run-gpu-task --repo-url URL [--ref REF] [--flavor FLAVOR] [--protocol-env PATH] [--train-config PATH] [--timeout DUR] [--namespace NS] [--detach] [--skip-train] [--skip-eval] [--skip-hf-fetch] [--skip-local-gen]
  scripts/hf_cli_workflow.sh ps [-a]
  scripts/hf_cli_workflow.sh logs <job_id>
  scripts/hf_cli_workflow.sh inspect <job_id>
  scripts/hf_cli_workflow.sh cancel <job_id>

Examples:
  scripts/hf_cli_workflow.sh setup
  scripts/hf_cli_workflow.sh login
  scripts/hf_cli_workflow.sh import HuggingFaceH4/ultrachat_200k data/hf/ultrachat --include "*.jsonl"
  scripts/hf_cli_workflow.sh run-job python:3.12 -- python -c "print('hello from hf jobs')" --flavor cpu-basic --detach
  scripts/hf_cli_workflow.sh run-gpu-task --repo-url https://huggingface.co/spaces/<user>/<space-repo> --flavor l4x1 --detach
EOF
}

need_hf() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI not found. Run: scripts/hf_cli_workflow.sh setup" >&2
    exit 1
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

sub="$1"
shift

case "$sub" in
  setup)
    python3 -m pip install --upgrade "huggingface_hub[cli]"
    hf --help >/dev/null
    echo "hf CLI installed."
    ;;

  login)
    need_hf
    hf auth login
    ;;

  import)
    need_hf
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi
    dataset_repo="$1"
    local_dir="$2"
    shift 2

    revision=""
    include_pattern=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --revision)
          revision="${2:-}"
          shift 2
          ;;
        --include)
          include_pattern="${2:-}"
          shift 2
          ;;
        *)
          echo "Unknown option for import: $1" >&2
          exit 1
          ;;
      esac
    done

    mkdir -p "$local_dir"
    cmd=(hf download "$dataset_repo" --repo-type dataset --local-dir "$local_dir")
    if [[ -n "$revision" ]]; then
      cmd+=(--revision "$revision")
    fi
    if [[ -n "$include_pattern" ]]; then
      cmd+=(--include "$include_pattern")
    fi
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
    ;;

  run-job)
    need_hf
    if [[ $# -lt 2 ]]; then
      usage
      exit 1
    fi

    image="$1"
    shift

    if [[ "${1:-}" != "--" ]]; then
      echo "run-job requires '--' before command." >&2
      exit 1
    fi
    shift

    if [[ $# -lt 1 ]]; then
      echo "run-job requires a command after '--'." >&2
      exit 1
    fi

    cmd_args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --flavor|--namespace|--detach)
          break
          ;;
        *)
          cmd_args+=("$1")
          shift
          ;;
      esac
    done

    flavor=""
    namespace=""
    detach=false
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --flavor)
          flavor="${2:-}"
          shift 2
          ;;
        --namespace)
          namespace="${2:-}"
          shift 2
          ;;
        --detach)
          detach=true
          shift
          ;;
        *)
          echo "Unknown option for run-job: $1" >&2
          exit 1
          ;;
      esac
    done

    cmd=(hf jobs run)
    if [[ -n "$flavor" ]]; then
      cmd+=(--flavor "$flavor")
    fi
    if [[ -n "$namespace" ]]; then
      cmd+=(--namespace "$namespace")
    fi
    if [[ "$detach" == true ]]; then
      cmd+=(--detach)
    fi
    cmd+=("$image")
    cmd+=("${cmd_args[@]}")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
    ;;

  run-gpu-task)
    need_hf
    repo_url=""
    repo_ref="main"
    flavor="l4x1"
    namespace=""
    timeout=""
    detach=false
    protocol_env="configs/protocol.env"
    train_config="configs/qlora_12gb.env"
    skip_train=false
    skip_eval=false
    skip_hf_fetch=false
    skip_local_gen=false

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --repo-url)
          repo_url="${2:-}"
          shift 2
          ;;
        --ref)
          repo_ref="${2:-}"
          shift 2
          ;;
        --flavor)
          flavor="${2:-}"
          shift 2
          ;;
        --namespace)
          namespace="${2:-}"
          shift 2
          ;;
        --timeout)
          timeout="${2:-}"
          shift 2
          ;;
        --protocol-env)
          protocol_env="${2:-}"
          shift 2
          ;;
        --train-config)
          train_config="${2:-}"
          shift 2
          ;;
        --skip-train)
          skip_train=true
          shift
          ;;
        --skip-eval)
          skip_eval=true
          shift
          ;;
        --skip-hf-fetch)
          skip_hf_fetch=true
          shift
          ;;
        --skip-local-gen)
          skip_local_gen=true
          shift
          ;;
        --detach)
          detach=true
          shift
          ;;
        *)
          echo "Unknown option for run-gpu-task: $1" >&2
          exit 1
          ;;
      esac
    done

    if [[ -z "$repo_url" ]]; then
      echo "run-gpu-task requires --repo-url <URL>" >&2
      exit 1
    fi

    train_flags=()
    if [[ "$skip_train" == true ]]; then
      train_flags+=(--skip-train)
    fi
    if [[ "$skip_eval" == true ]]; then
      train_flags+=(--skip-eval)
    fi
    if [[ "$skip_hf_fetch" == true ]]; then
      train_flags+=(--skip-hf-fetch)
    fi
    if [[ "$skip_local_gen" == true ]]; then
      train_flags+=(--skip-local-gen)
    fi

    remote_cmd=$(cat <<EOF
set -euo pipefail
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir huggingface_hub[cli]
mkdir -p /workspace
cd /workspace
if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y --no-install-recommends git
fi
rm -rf /workspace/repo
git clone --depth 1 --branch '$repo_ref' '$repo_url' /workspace/repo
cd /workspace/repo
python -m pip install --no-cache-dir -r requirements.txt
python scripts/run_training_protocol.py --protocol-env '$protocol_env' --train-config '$train_config' ${train_flags[*]}
EOF
)

    cmd=(hf jobs run --secrets HF_TOKEN --flavor "$flavor")
    if [[ -n "$namespace" ]]; then
      cmd+=(--namespace "$namespace")
    fi
    if [[ -n "$timeout" ]]; then
      cmd+=(--timeout "$timeout")
    fi
    if [[ "$detach" == true ]]; then
      cmd+=(--detach)
    fi
    cmd+=(python:3.12 bash -lc "$remote_cmd")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
    ;;

  ps)
    need_hf
    hf jobs ps "$@"
    ;;

  logs)
    need_hf
    if [[ $# -ne 1 ]]; then
      echo "Usage: scripts/hf_cli_workflow.sh logs <job_id>" >&2
      exit 1
    fi
    hf jobs logs "$1"
    ;;

  inspect)
    need_hf
    if [[ $# -ne 1 ]]; then
      echo "Usage: scripts/hf_cli_workflow.sh inspect <job_id>" >&2
      exit 1
    fi
    hf jobs inspect "$1"
    ;;

  cancel)
    need_hf
    if [[ $# -ne 1 ]]; then
      echo "Usage: scripts/hf_cli_workflow.sh cancel <job_id>" >&2
      exit 1
    fi
    hf jobs cancel "$1"
    ;;

  *)
    usage
    exit 1
    ;;
esac
