[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON,
  [ValidateSet("cuda")]
  [string]$Device = "cuda",
  [switch]$SkipFetch,
  [switch]$SkipDev,
  [switch]$SkipInfrastructureSmoke
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Invoke-GfmJson {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  $output = & $GfmPython -m socialgraph_gfm.cli @Arguments --json
  if ($LASTEXITCODE -ne 0) {
    throw "socialgraph-gfm failed ($LASTEXITCODE): $($Arguments -join ' ')`n$output"
  }
  return ($output | Out-String | ConvertFrom-Json)
}

Push-Location $projectRoot
try {
  if (-not $SkipFetch) {
    . (Join-Path $PSScriptRoot "Enter-BaselineRuntime.ps1") `
      -RuntimeRoot $RuntimeRoot -GfmPython $GfmPython -Operation fetch | Out-Null
    Invoke-GfmJson -Arguments @(
      "corpus-fetch-ogbl-collab", "--accept-license", "ODC-BY-1.0",
      "--root", $RuntimeRoot
    ) | Out-Null
  } else {
    . (Join-Path $PSScriptRoot "Enter-BaselineRuntime.ps1") `
      -RuntimeRoot $RuntimeRoot -GfmPython $GfmPython -Operation run | Out-Null
  }

  if (-not $SkipInfrastructureSmoke) {
    Invoke-GfmJson -Arguments @(
      "smoke", "--fixture", "both", "--device", $Device, "--root", $RuntimeRoot
    ) | Out-Null
  }

  $package = Join-Path $RuntimeRoot "datasets\packages\ogbl-collab.sgfm.zip"
  if (-not (Test-Path -LiteralPath $package -PathType Leaf)) {
    throw "The safe corpus package is absent: $package"
  }
  Invoke-GfmJson -Arguments @(
    "corpus-prepare-ogbl-collab", "--package", $package, "--root", $RuntimeRoot
  ) | Out-Null
  Invoke-GfmJson -Arguments @(
    "corpus-check", "--corpus-id", "ogbl-collab", "--root", $RuntimeRoot
  ) | Out-Null

  if (-not $SkipDev) {
    Invoke-GfmJson -Arguments @(
      "baseline-run", "--phase", "dev", "--track", "both",
      "--device", $Device, "--root", $RuntimeRoot
    ) | Out-Null
  }

  $formal = Invoke-GfmJson -Arguments @(
    "baseline-run", "--phase", "formal", "--track", "both",
    "--device", $Device, "--root", $RuntimeRoot
  )
  $experimentId = $formal.experimentId
  if (-not $experimentId) {
    throw "Formal baseline output did not contain experimentId."
  }
  $acceptance = Invoke-GfmJson -Arguments @(
    "baseline-validate", "--experiment-id", $experimentId, "--root", $RuntimeRoot
  )
  $preflight = Invoke-GfmJson -Arguments @(
    "preflight", "--device", $Device, "--root", $RuntimeRoot
  )

  [pscustomobject]@{
    RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
    ExperimentId = $experimentId
    Acceptance = $acceptance
    Readiness = $preflight.readiness
  }
} finally {
  Pop-Location
}
