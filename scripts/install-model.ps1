param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "lib\UnifiedOperations.ps1")

$layout = Get-UnifiedLayout -ProjectRoot $projectRoot
Initialize-UnifiedDirectories -Layout $layout
Invoke-WithProcessEnvironment -Environment (Get-ClearedLlmEnvironment) -ScriptBlock {
    Install-AuthorizedModelPackage -Layout $layout -PackageRoot $PackageRoot | Out-Null
}
