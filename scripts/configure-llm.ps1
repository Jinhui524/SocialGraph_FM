param(
    [string]$BootstrapPython,
    [ValidateSet("openai_responses", "deepseek", "glm", "anthropic", "custom", "custom_anthropic")]
    [string]$Preset,
    [string]$ApiBase,
    [string]$Model,
    [ValidateSet("responses", "chat_completions", "anthropic_messages")]
    [string]$ApiMode,
    [ValidateSet("bearer", "x-api-key")]
    [string]$AuthScheme,
    [string]$AnthropicVersion,
    [ValidateRange(1, 60)]
    [int]$TimeoutSeconds,
    [Security.SecureString]$ApiKey,
    [switch]$AllowInsecureLoopback,
    [switch]$TestLlm,
    [switch]$SkipLlmTest
)

$ErrorActionPreference = "Stop"
if ($TestLlm -and $SkipLlmTest) {
    throw "-TestLlm and -SkipLlmTest cannot be used together."
}

. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

$arguments = @((Join-Path $PSScriptRoot "socialgraph.py"), "configure-llm")
if ($PSBoundParameters.ContainsKey("Preset")) {
    $arguments += @("--preset", $Preset.ToLowerInvariant())
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
if ($PSBoundParameters.ContainsKey("TimeoutSeconds")) {
    $arguments += @("--timeout-seconds", [string]$TimeoutSeconds)
}
if ($AllowInsecureLoopback) { $arguments += "--allow-insecure-loopback" }
if ($TestLlm) { $arguments += "--test-llm" }
if ($SkipLlmTest) { $arguments += "--skip-llm-test" }

$keyPointer = [IntPtr]::Zero
$plainKey = $null
try {
    if ($PSBoundParameters.ContainsKey("ApiKey")) {
        $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ApiKey)
        $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
        $arguments += "--api-key-stdin"
        $plainKey | & $launcher.FilePath @arguments
    }
    else {
        & $launcher.FilePath @arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "SocialGraph-FM LLM configuration failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Remove-Variable plainKey -ErrorAction SilentlyContinue
    Remove-Variable ApiKey -ErrorAction SilentlyContinue
}
