param(
    [string]$WheelProfile,
    [ValidateSet("Auto", "Cpu", "Cuda-Required")]
    [string]$DevicePolicy = "Auto",
    [ValidateSet("Auto", "Reuse", "Managed")]
    [string]$EnvMode = "Auto",
    [string]$BootstrapPython,
    [string]$ApiPython,
    [string]$GfmPython,
    [string]$Preset,
    [string]$ApiBase,
    [string]$Model,
    [ValidateSet("Chat_Completions", "Responses", "Anthropic_Messages")]
    [string]$ApiMode,
    [ValidateSet("Bearer", "X-Api-Key")]
    [string]$AuthScheme,
    [string]$AnthropicVersion,
    [ValidateRange(1, 60)]
    [int]$TimeoutSeconds = 15,
    [switch]$ApiKeyStdin,
    [switch]$SkipWeb,
    [switch]$GfmTextProfile,
    [switch]$AllowInsecureLoopback
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

$arguments = @(
    (Join-Path $PSScriptRoot "socialgraph.py"),
    "onboard",
    "--device-policy", $DevicePolicy.ToLowerInvariant(),
    "--env-mode", $EnvMode.ToLowerInvariant()
)
if ($PSBoundParameters.ContainsKey("WheelProfile")) {
    $arguments += @("--wheel-profile", $WheelProfile.ToLowerInvariant())
}
if ($PSBoundParameters.ContainsKey("BootstrapPython")) {
    $arguments += @("--bootstrap-python", $launcher.BootstrapPython)
}
if ($PSBoundParameters.ContainsKey("ApiPython")) {
    $arguments += @("--api-python", $ApiPython)
}
if ($PSBoundParameters.ContainsKey("GfmPython")) {
    $arguments += @("--gfm-python", $GfmPython)
}
if ($PSBoundParameters.ContainsKey("Preset")) {
    $arguments += @("--preset", $Preset)
}
if ($PSBoundParameters.ContainsKey("ApiBase")) {
    $arguments += @("--api-base", $ApiBase)
}
if ($PSBoundParameters.ContainsKey("Model")) {
    $arguments += @("--model", $Model)
}
if ($PSBoundParameters.ContainsKey("ApiMode")) {
    $arguments += @("--api-mode", $ApiMode.ToLowerInvariant())
}
if ($PSBoundParameters.ContainsKey("AuthScheme")) {
    $arguments += @("--auth-scheme", $AuthScheme.ToLowerInvariant())
}
if ($PSBoundParameters.ContainsKey("AnthropicVersion")) {
    $arguments += @("--anthropic-version", $AnthropicVersion)
}
$arguments += @("--timeout-seconds", "$TimeoutSeconds")
if ($ApiKeyStdin) { $arguments += "--api-key-stdin" }
if ($SkipWeb) { $arguments += "--skip-web" }
if ($GfmTextProfile) { $arguments += "--gfm-text" }
if ($AllowInsecureLoopback) { $arguments += "--allow-insecure-loopback" }

& $launcher.FilePath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM onboarding failed with exit code $LASTEXITCODE."
}
