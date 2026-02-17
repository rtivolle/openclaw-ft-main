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
  scripts/hf_cli_workflow.sh ps [-a]
  scripts/hf_cli_workflow.sh logs <job_id>
  scripts/hf_cli_workflow.sh inspect <job_id>
  scripts/hf_cli_workflow.sh cancel <job_id>

Examples:
  scripts/hf_cli_workflow.sh setup
  scripts/hf_cli_workflow.sh login
  scripts/hf_cli_workflow.sh import HuggingFaceH4/ultrachat_200k data/hf/ultrachat --include "*.jsonl"
  scripts/hf_cli_workflow.sh run-job python:3.12 -- python -c "print('hello from hf jobs')" --flavor cpu-basic --detach
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
