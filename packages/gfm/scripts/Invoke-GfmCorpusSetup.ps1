[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "Enter-GfmRuntime.ps1") `
  -RuntimeRoot $RuntimeRoot `
  -GfmPython $GfmPython `
  -Operation fetch `
  -PromptForSecrets `
  -SecretAction {
    param($Runtime)

    function Invoke-GfmCli {
      param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

      & $Runtime.GfmPython -m socialgraph_gfm.cli @Arguments
      if ($LASTEXITCODE -ne 0) {
        throw "GFM command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
      }
    }

    Invoke-GfmCli `
      "gfm-corpus-fetch-openalex" `
      "--spec" "graph-ai" `
      "--api-key-env" "OPENALEX_API_KEY" `
      "--root" $Runtime.RuntimeRoot `
      "--json"
    Invoke-GfmCli `
      "gfm-corpus-prepare" `
      "--domain" "openalex" `
      "--newcomer-overlay" "skip" `
      "--root" $Runtime.RuntimeRoot `
      "--json"
    Invoke-GfmCli `
      "gfm-corpus-prepare" `
      "--domain" "thgl-software" `
      "--root" $Runtime.RuntimeRoot `
      "--json"
    Invoke-GfmCli `
      "gfm-corpus-prepare" `
      "--domain" "wikimedia-talk" `
      "--root" $Runtime.RuntimeRoot `
      "--json"
  }

Write-Host "Three-domain corpus preparation finished successfully." -ForegroundColor Green
