param([string]$BootstrapPython)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

& $launcher.FilePath (Join-Path $PSScriptRoot "socialgraph.py") "stop"
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM shutdown failed with exit code $LASTEXITCODE."
}
