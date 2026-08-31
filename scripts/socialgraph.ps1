param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SocialGraphArguments
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher

& $launcher.FilePath (Join-Path $PSScriptRoot "socialgraph.py") @SocialGraphArguments
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM command failed with exit code $LASTEXITCODE."
}
