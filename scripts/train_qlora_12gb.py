import os
import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_paths(name: str) -> list[str]:
    value = os.getenv(name, "").strip()
    if not value:
        return []
    raw_parts = value.replace(os.pathsep, ",").replace(";", ",").split(",")
    paths = [part.strip() for part in raw_parts if part.strip()]
    return [str(Path(path).expanduser()) for path in paths]


def supported_sft_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    return {key: value for key, value in kwargs.items() if key in params}


def format_example(example: dict[str, Any]) -> dict[str, str]:
    parts = []
    for item in example["messages"]:
        role = item["role"].strip().lower()
        content = item["content"].strip()
        parts.append(f"<|{role}|>\n{content}\n")
    return {"text": "".join(parts)}


def pick_safe_runtime(batch_size: int, grad_accum: int, max_length: int) -> tuple[int, int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    safe_batch = batch_size
    safe_grad = grad_accum
    safe_len = max_length

    if free_gb < 8.0:
        safe_batch = 1
        safe_grad = max(12, grad_accum)
        safe_len = min(max_length, 1024)
    elif free_gb < 10.0:
        safe_batch = min(batch_size, 1)
        safe_grad = max(10, grad_accum)
        safe_len = min(max_length, 1536)
    else:
        safe_batch = min(batch_size, 2)
        safe_len = min(max_length, 2048)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM free/total: {free_gb:.2f} / {total_gb:.2f} GiB")
    print(f"Runtime settings -> batch={safe_batch}, grad_acc={safe_grad}, max_len={safe_len}")
    return safe_batch, safe_grad, safe_len


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No CUDA device detected.")

    base_model = env("BASE_MODEL", "unsloth/DeepSeek-R1-Distill-Llama-8B")
    output_dir = env("OUTPUT_DIR", "artifacts/openclaw-lora-12gb")
    train_file = env("TRAIN_FILE", "openclaw-ft/data/train_200.jsonl")
    val_file = env("VAL_FILE", "openclaw-ft/data/val_20.jsonl")
    resume_from_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT")
    train_files = env_paths("TRAIN_FILES") or [train_file]
    val_files = env_paths("VAL_FILES") or [val_file]
    map_num_proc = max(1, env_int("DATASET_MAP_NUM_PROC", min(8, os.cpu_count() or 1)))

    print(f"Training shards ({len(train_files)}): {train_files}")
    print(f"Validation shards ({len(val_files)}): {val_files}")
    print(f"Dataset map workers: {map_num_proc}")

    dataset = load_dataset("json", data_files={"train": train_files, "validation": val_files})
    dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
        num_proc=map_num_proc,
        desc="Formatting conversation examples",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Quantization mode: Q4 (4-bit, NF4)")

    req_batch = env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
    req_grad = env_int("GRADIENT_ACCUMULATION_STEPS", 16)
    req_len = env_int("MAX_SEQ_LENGTH", 1536)
    safe_batch, safe_grad, safe_len = pick_safe_runtime(req_batch, req_grad, req_len)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    preferred_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    target_modules = [name for name in preferred_targets if name in available_modules]
    if not target_modules:
        raise RuntimeError(
            "No compatible LoRA target modules found on this model. "
            f"Expected one of: {preferred_targets}"
        )

    lora = LoraConfig(
        r=env_int("LORA_R", 16),
        lora_alpha=env_int("LORA_ALPHA", 16),
        lora_dropout=env_float("LORA_DROPOUT", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    bf16_supported = torch.cuda.is_bf16_supported()

    sft_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": env_int("NUM_EPOCHS", 3),
        "max_steps": env_int("MAX_STEPS", -1),
        "per_device_train_batch_size": safe_batch,
        "gradient_accumulation_steps": safe_grad,
        "learning_rate": env_float("LEARNING_RATE", 2e-4),
        "warmup_ratio": env_float("WARMUP_RATIO", 0.03),
        "lr_scheduler_type": env("LR_SCHEDULER_TYPE", "cosine"),
        "logging_steps": 10,
        "eval_steps": env_int("EVAL_STEPS", 200),
        "save_steps": env_int("SAVE_STEPS", 200),
        "eval_strategy": "steps",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": env_bool("LOAD_BEST_MODEL_AT_END", True),
        "save_total_limit": 2,
        "optim": "paged_adamw_8bit",
        "max_length": safe_len,
        "packing": True,
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
        "use_cpu": False,
        "report_to": "none",
        # Speed up host-side sample loading; useful when shards are spread across disks.
        "dataloader_num_workers": env_int("DATALOADER_NUM_WORKERS", min(8, os.cpu_count() or 1)),
        "dataloader_prefetch_factor": env_int("DATALOADER_PREFETCH_FACTOR", 2),
        "dataloader_persistent_workers": env_bool("DATALOADER_PERSISTENT_WORKERS", True),
        "dataloader_pin_memory": env_bool("DATALOADER_PIN_MEMORY", True),
    }
    args = SFTConfig(**supported_sft_kwargs(sft_kwargs))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora,
        args=args,
    )

    patience = env_int("EARLY_STOPPING_PATIENCE", 3)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter to: {output_dir}")


if __name__ == "__main__":
    main()
