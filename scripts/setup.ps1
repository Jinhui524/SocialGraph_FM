param(
    [ValidateSet("Offline", "Cpu", "Cuda")]
    [string]$Profile,
    [string]$WheelProfile,
    [ValidateSet("Auto", "Cpu", "Cuda-Required")]
    [string]$DevicePolicy = "Auto",
    [ValidateSet("Auto", "Reuse", "Managed")]
    [string]$EnvMode = "Auto",
    [string]$BootstrapPython,
    [string]$ApiPython,
    [string]$GfmPython,
    [switch]$SkipApi,
    [switch]$SkipWeb,
    [switch]$GfmTextProfile
)

$ErrorActionPreference = "Stop"
if ($PSBoundParameters.ContainsKey("Profile") -and $PSBoundParameters.ContainsKey("WheelProfile")) {
    throw "Use either -Profile or -WheelProfile, not both."
}
$launcherLibrary = Join-Path $PSScriptRoot "lib/PythonLauncher.ps1"
. $launcherLibrary
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

$arguments = @(
    (Join-Path $PSScriptRoot "socialgraph.py"),
    "setup"
)
if ($PSBoundParameters.ContainsKey("Profile")) {
    $arguments += @("--profile", $Profile.ToLowerInvariant())
}
if ($PSBoundParameters.ContainsKey("WheelProfile")) {
    $arguments += @("--wheel-profile", $WheelProfile.ToLowerInvariant())
}
$arguments += @("--device-policy", $DevicePolicy.ToLowerInvariant())
$arguments += @("--env-mode", $EnvMode.ToLowerInvariant())
if ($PSBoundParameters.ContainsKey("BootstrapPython")) {
    $arguments += @("--bootstrap-python", $launcher.BootstrapPython)
}
if ($PSBoundParameters.ContainsKey("ApiPython")) {
    $arguments += @("--api-python", $ApiPython)
}
if ($PSBoundParameters.ContainsKey("GfmPython")) {
    $arguments += @("--gfm-python", $GfmPython)
}
if ($SkipApi) { $arguments += "--skip-api" }
if ($SkipWeb) { $arguments += "--skip-web" }
if ($GfmTextProfile) { $arguments += "--gfm-text" }

& $launcher.FilePath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM setup failed with exit code $LASTEXITCODE."
}
