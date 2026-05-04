param(
    [string]$HfToken,
    [ValidateSet("react", "cli")][string]$Mode = "react",
    [int]$Port = 7860,
    [int]$MaxNewTokens = 96,
    [string]$BaseModelId = "google/gemma-2b-it",
    [switch]$SkipAdapter,
    [switch]$CheckOnly,
    [switch]$SkipInstall,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not $SkipInstall) {
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt
    try {
        & $PythonExe -m pip install bitsandbytes
    } catch {
        Write-Warning "bitsandbytes install failed. The app will run without 4-bit quantization."
    }
}

$pythonArgs = @(
    ".\chat_interface.py",
    "--adapter-path", $projectRoot,
    "--base-model-id", $BaseModelId,
    "--mode", $Mode,
    "--port", $Port,
    "--max-new-tokens", $MaxNewTokens
)
if ($SkipAdapter) {
    $pythonArgs += "--skip-adapter"
}
if ($CheckOnly) {
    $pythonArgs += "--check-only"
    & $PythonExe @pythonArgs
    exit $LASTEXITCODE
}
if (-not $HfToken) {
    $HfToken = $env:HF_TOKEN
}
if ($HfToken) {
    $env:HF_TOKEN = $HfToken
} else {
    Write-Warning "HF token not set. The app will attempt unauthenticated/local-cache model loading."
}
& $PythonExe @pythonArgs
