[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON,
  [ValidateSet("fetch", "run")]
  [string]$Operation = "run"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
  throw "Set SOCIALGRAPH_FM_HOME or pass -RuntimeRoot explicitly."
}
if ([string]::IsNullOrWhiteSpace($GfmPython)) {
  throw "Set SOCIALGRAPH_GFM_PYTHON or pass -GfmPython explicitly."
}

$selectedRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$selectedPython = [System.IO.Path]::GetFullPath($GfmPython)
if (-not (Test-Path -LiteralPath $selectedPython -PathType Leaf)) {
  throw "GFM Python executable does not exist: $selectedPython"
}

$rootPath = [System.IO.Path]::GetPathRoot($selectedRoot)
$driveName = $rootPath.TrimEnd("\").TrimEnd(":")
$drive = Get-PSDrive -Name $driveName -PSProvider FileSystem
$minimumGiB = if ($Operation -eq "fetch") { 30 } else { 20 }
$minimumBytes = [int64]$minimumGiB * 1GB
if ($drive.Free -lt $minimumBytes) {
  $freeGiB = [math]::Round($drive.Free / 1GB, 2)
  throw "Insufficient free space for $Operation`: $freeGiB GiB available; $minimumGiB GiB required."
}

$directories = @(
  "datasets\raw\ogb",
  "datasets\raw\gfm\openalex",
  "datasets\raw\gfm\thgl-software",
  "datasets\raw\gfm\wikimedia-talk",
  "datasets\packages",
  "datasets\processed",
  "datasets\processed\gfm",
  "datasets\manifests",
  "datasets\manifests\gfm",
  "embeddings",
  "runs",
  "runs\gfm",
  "registry",
  "reports",
  "reports\gfm",
  "models\staging",
  "models\released",
  "cache\hf",
  "cache\pip",
  "cache\uv",
  "cache\torch",
  "cache\torchinductor",
  "cache\wandb",
  "tmp",
  "exports"
)
foreach ($relativePath in $directories) {
  $directory = Join-Path $selectedRoot $relativePath
  [void](New-Item -ItemType Directory -Path $directory -Force)
}

# These assignments are process-scoped and never edit the machine/user environment.
# Dot-source this script when the caller needs the values.
$env:SOCIALGRAPH_FM_HOME = $selectedRoot
$env:SOCIALGRAPH_GFM_PYTHON = $selectedPython
$env:PIP_CACHE_DIR = Join-Path $selectedRoot "cache\pip"
$env:UV_CACHE_DIR = Join-Path $selectedRoot "cache\uv"
$env:HF_HOME = Join-Path $selectedRoot "cache\hf"
$env:TORCH_HOME = Join-Path $selectedRoot "cache\torch"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $selectedRoot "cache\torchinductor"
$env:WANDB_DIR = Join-Path $selectedRoot "cache\wandb"
$env:TEMP = Join-Path $selectedRoot "tmp"
$env:TMP = Join-Path $selectedRoot "tmp"

[pscustomobject]@{
  RuntimeRoot = $selectedRoot
  GfmPython = $selectedPython
  Operation = $Operation
  FreeGiB = [math]::Round($drive.Free / 1GB, 3)
  MinimumFreeGiB = $minimumGiB
  Scope = "Process"
}
