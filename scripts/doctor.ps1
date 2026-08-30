param(
    [string]$BootstrapPython,
    [switch]$TestLlm,
    [switch]$Json,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

$arguments = @((Join-Path $PSScriptRoot "socialgraph.py"), "doctor")
if ($TestLlm) { $arguments += "--test-llm" }
if ($Json) { $arguments += "--json" }
if ($Full) { $arguments += "--full" }

& $launcher.FilePath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM doctor failed with exit code $LASTEXITCODE."
}
