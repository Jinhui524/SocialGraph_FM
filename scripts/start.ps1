param(
    [string]$BootstrapPython,
    [ValidateSet("Optional", "Required", "Disabled")]
    [string]$LlmMode = "Optional",
    [switch]$NoLlmPrompt,
    [switch]$ReconfigureLlm,
    [switch]$TestLlm
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython

$arguments = @(
    (Join-Path $PSScriptRoot "socialgraph.py"),
    "start",
    "--llm-mode", $LlmMode.ToLowerInvariant()
)
if ($NoLlmPrompt) { $arguments += "--no-llm-prompt" }
if ($ReconfigureLlm) { $arguments += "--reconfigure-llm" }
if ($TestLlm) { $arguments += "--test-llm" }

& $launcher.FilePath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "SocialGraph-FM startup failed with exit code $LASTEXITCODE."
}
