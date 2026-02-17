param(
  [string]$BaseModel = "unsloth/DeepSeek-R1-Distill-Llama-8B",
  [string]$TrainFile = "openclaw-ft/data/train_200.jsonl",
  [string]$ValFile = "openclaw-ft/data/val_20.jsonl",
  [string]$OutputDir = "artifacts/openclaw-lora-12gb",
  [string]$EvalOut = "openclaw-ft/data/eval_results.json",
  [string]$ConfigFile = "openclaw-ft/configs/qlora_12gb.env",
  [string]$ResumeFromCheckpoint = ""
)

function Import-EnvFile([string]$Path) {
  if (-not (Test-Path $Path)) { return }
  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    $pair = $line -split "=", 2
    if ($pair.Count -ne 2) { return }
    [System.Environment]::SetEnvironmentVariable($pair[0].Trim(), $pair[1].Trim())
  }
}

Import-EnvFile $ConfigFile
$env:BASE_MODEL = $BaseModel
$env:OUTPUT_DIR = $OutputDir
$env:TRAIN_FILE = $TrainFile
$env:VAL_FILE = $ValFile
$env:RESUME_FROM_CHECKPOINT = $ResumeFromCheckpoint

python -c "import torch,sys; sys.exit(0) if torch.cuda.is_available() else sys.exit('CUDA required but not detected')"
if ($LASTEXITCODE -ne 0) {
  throw "CUDA check failed"
}

python .\openclaw-ft\scripts\train_qlora_12gb.py
if ($LASTEXITCODE -ne 0) {
  throw "Training failed"
}

python .\openclaw-ft\scripts\eval_baseline_vs_adapter.py --base-model $BaseModel --adapter-dir $OutputDir --eval-file $ValFile --out $EvalOut
if ($LASTEXITCODE -ne 0) {
  throw "Evaluation failed"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Split-Path -Parent $EvalOut
$name = [System.IO.Path]::GetFileNameWithoutExtension($EvalOut)
$ext = [System.IO.Path]::GetExtension($EvalOut)
$versionedEvalOut = Join-Path $dir "$name-$timestamp$ext"
Copy-Item $EvalOut $versionedEvalOut -Force

Write-Host "Latest eval: $EvalOut"
Write-Host "Versioned eval: $versionedEvalOut"
Get-Content $versionedEvalOut
