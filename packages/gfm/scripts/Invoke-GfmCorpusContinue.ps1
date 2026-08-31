[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON
)

$ErrorActionPreference = "Stop"

# Continue from immutable raw/prepared artifacts. This path deliberately never
# fetches OpenAlex and never starts or resumes the optional newcomer overlay.
& (Join-Path $PSScriptRoot "Enter-GfmRuntime.ps1") `
  -RuntimeRoot $RuntimeRoot `
  -GfmPython $GfmPython `
  -Operation run `
  -PromptForWikimediaSalt `
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

Write-Host "Three-domain corpus continuation finished successfully." -ForegroundColor Green
