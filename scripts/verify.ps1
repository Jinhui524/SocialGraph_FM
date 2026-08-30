param(
    [ValidateSet("Offline", "Cpu", "Cuda")]
    [string]$Profile = "Offline",
    [switch]$SkipBuild,
    [switch]$SkipE2E
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "lib\UnifiedOperations.ps1")
$clearedLlmEnvironment = Get-ClearedLlmEnvironment

function Invoke-VerificationCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host "==> $Label" -ForegroundColor Cyan
    Invoke-WithProcessEnvironment -Environment $clearedLlmEnvironment -ScriptBlock $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$layout = Get-UnifiedLayout -ProjectRoot $projectRoot -Profile $Profile
$ports = Get-UnifiedPorts
Set-UnifiedEnvironment -Layout $layout -Ports $ports
$verificationDrive = [IO.Path]::GetPathRoot($projectRoot)
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $pathHashBytes = $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot))
}
finally {
    $sha256.Dispose()
}
$pathHash = ([BitConverter]::ToString($pathHashBytes) -replace "-", "").Substring(0, 10).ToLowerInvariant()
$verificationTemp = Join-Path $verificationDrive "sgfm-tmp\verify-$pathHash"
New-Item -ItemType Directory -Force -Path $verificationTemp | Out-Null
$env:TEMP = $verificationTemp
$env:TMP = $verificationTemp
if (-not (Test-Path -LiteralPath $layout.ApiPython -PathType Leaf)) {
    throw "SocialGraph-FM API environment is missing. Run scripts\setup.ps1 -Profile $Profile first."
}

& (Join-Path $PSScriptRoot "publication-check.ps1") -RepositoryRoot $projectRoot
& (Join-Path $PSScriptRoot "tests\NativeCommand.Tests.ps1")
& (Join-Path $PSScriptRoot "tests\Startup.Tests.ps1")
& (Join-Path $PSScriptRoot "tests\RuntimeBundle.Tests.ps1")

Invoke-VerificationCommand "Repository tests" {
    & $layout.ApiPython -m pytest (Join-Path $projectRoot "tests")
}

Push-Location $layout.Api
try {
    Invoke-VerificationCommand "API Ruff" { & $layout.ApiPython -m ruff check app tests }
    Invoke-VerificationCommand "API mypy" { & $layout.ApiPython -m mypy app }
    Invoke-VerificationCommand "API tests" { & $layout.ApiPython -m pytest }
    & $layout.ApiPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pip_audit') else 1)"
    if ($LASTEXITCODE -eq 0) {
        Invoke-VerificationCommand "API dependency audit" {
            & $layout.ApiPython -m pip_audit --local
        }
    }
}
finally {
    Pop-Location
}

$npm = if ($env:OS -eq "Windows_NT") { "npm.cmd" } else { "npm" }
Invoke-VerificationCommand "Web dependency audit" { & $npm --prefix $layout.GovernanceWeb audit --audit-level=high }
Invoke-VerificationCommand "Web typecheck" { & $npm --prefix $layout.GovernanceWeb run typecheck }
Invoke-VerificationCommand "Web tests" { & $npm --prefix $layout.GovernanceWeb test -- --reporter=dot }
Invoke-VerificationCommand "Web benchmark-runner tests" { & $npm --prefix $layout.GovernanceWeb run test:benchmark-runner }
if (-not $SkipBuild) {
    Invoke-VerificationCommand "Web production build" { & $npm --prefix $layout.GovernanceWeb run build }
}
if (-not $SkipE2E) {
    Invoke-VerificationCommand "Web offline browser tests" { & $npm --prefix $layout.GovernanceWeb run test:e2e:offline }
}

if ($Profile -ne "Offline") {
    if (-not (Test-Path -LiteralPath $layout.GfmPython -PathType Leaf)) {
        throw "GFM environment is missing for profile $Profile."
    }
    Push-Location $layout.GfmPackage
    try {
        $previousPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $layout.GfmPackage "src"
        try {
            Invoke-VerificationCommand "GFM Ruff" { & $layout.GfmPython -m ruff check src tests }
            Invoke-VerificationCommand "GFM mypy" { & $layout.GfmPython -m mypy src }
            Invoke-VerificationCommand "GFM tests" { & $layout.GfmPython -m pytest -ra }
            Invoke-VerificationCommand "GFM doctor" {
                & $layout.GfmPython -m socialgraph_gfm.cli doctor --device cpu --root $layout.GfmHome --json
            }
            Invoke-VerificationCommand "GFM contract smoke" {
                & $layout.GfmPython -m socialgraph_gfm.cli smoke --fixture both --device cpu --root $layout.GfmHome --json
            }
        }
        finally {
            $env:PYTHONPATH = $previousPythonPath
        }
    }
    finally {
        Pop-Location
    }
}

Invoke-VerificationCommand "Git whitespace check" { git -C $projectRoot diff --check }
Write-Host "All requested verification gates passed." -ForegroundColor Green
