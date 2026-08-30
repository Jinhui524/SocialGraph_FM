param(
    [ValidateSet("openai_responses", "deepseek", "glm", "anthropic", "custom", "custom_anthropic")]
    [string]$Preset,
    [string]$ApiBase,
    [string]$Model,
    [ValidateSet("responses", "chat_completions", "anthropic_messages")][string]$ApiMode,
    [ValidateSet("bearer", "x-api-key")][string]$AuthScheme,
    [string]$AnthropicVersion,
    [ValidateRange(1, 60)][int]$TimeoutSeconds,
    [Security.SecureString]$ApiKey,
    [switch]$AllowInsecureLoopback,
    [switch]$TestLlm,
    [switch]$SkipLlmTest
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$forward = @{}
if ($PSBoundParameters.ContainsKey("Preset")) { $forward.Preset = $Preset }
if (-not [string]::IsNullOrWhiteSpace($ApiBase)) { $forward.ApiBase = $ApiBase }
if (-not [string]::IsNullOrWhiteSpace($Model)) { $forward.Model = $Model }
if ($PSBoundParameters.ContainsKey("ApiMode")) { $forward.ApiMode = $ApiMode }
if ($PSBoundParameters.ContainsKey("AuthScheme")) { $forward.AuthScheme = $AuthScheme }
if ($PSBoundParameters.ContainsKey("AnthropicVersion")) {
    $forward.AnthropicVersion = $AnthropicVersion
}
if ($PSBoundParameters.ContainsKey("TimeoutSeconds")) { $forward.TimeoutSeconds = $TimeoutSeconds }
if ($null -ne $ApiKey) { $forward.ApiKey = $ApiKey }
if ($AllowInsecureLoopback) { $forward.AllowInsecureLoopback = $true }
if ($TestLlm) { $forward.TestLlm = $true }
if ($SkipLlmTest) { $forward.SkipLlmTest = $true }
& (Join-Path $projectRoot "scripts\configure-llm.ps1") @forward
