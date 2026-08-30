$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$scriptsRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $scriptsRoot "lib\UnifiedOperations.ps1")

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Throws {
    param([scriptblock]$Action, [string]$Message)
    try { & $Action }
    catch { return }
    throw $Message
}

$systemTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testRoot = Join-Path $systemTemp "SocialGraph FM startup $PID $([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $offline = Get-UnifiedLayout -ProjectRoot $testRoot -Profile Offline
    $cpu = Get-UnifiedLayout -ProjectRoot $testRoot -Profile Cpu
    $cuda = Get-UnifiedLayout -ProjectRoot $testRoot -Profile Cuda
    Assert-True ($offline.GovernanceWeb -eq (Join-Path $testRoot "apps\web")) "Web path did not use apps/web."
    Assert-True ($offline.Api -eq (Join-Path $testRoot "services\api")) "API path did not use services/api."
    Assert-True ($offline.GfmPackage -eq (Join-Path $testRoot "packages\gfm")) "GFM path did not use packages/gfm."
    Assert-True ($cpu.GfmPython -like "*gfm-cpu*") "CPU and CUDA environments were not isolated."
    Assert-True ($cuda.GfmPython -like "*gfm-cu130-clean*") "CUDA environment path was not preserved."
    Assert-True ($offline.UvExecutable -like "*var*tools*uv-0.12.3*") "Pinned uv is not repository-local."
    Assert-True ($offline.ManagedPythonRoot -like "*var*gfm*python") "Managed Python is not repository-local."
    Assert-True ($offline.ManagedPythonVersion -eq "3.12.13") "Managed Python patch version changed."

    $presetCatalogPath = Join-Path $scriptsRoot "config\llm-presets.json"
    $presetCatalog = Read-LlmPresetCatalog -Path $presetCatalogPath
    Assert-True ($presetCatalog.presets.openai_responses.apiBase -eq "https://api.openai.com/v1") `
        "OpenAI preset API Base changed."
    Assert-True ($presetCatalog.presets.openai_responses.defaultApiMode -eq "responses") `
        "OpenAI preset did not default to Responses."
    Assert-True ($presetCatalog.presets.deepseek.defaultApiMode -eq "chat_completions") `
        "DeepSeek preset did not default to Chat Completions."
    Assert-True ($presetCatalog.presets.glm.apiBase -eq "https://open.bigmodel.cn/api/paas/v4") `
        "GLM preset API Base changed."
    Assert-True ($presetCatalog.presets.anthropic.defaultApiMode -eq "anthropic_messages") `
        "Anthropic preset did not default to Messages."
    Assert-True ($presetCatalog.presets.anthropic.defaultAuthScheme -eq "x-api-key") `
        "Anthropic preset did not default to x-api-key."
    foreach ($presetId in @(
        "openai_responses", "deepseek", "glm", "anthropic", "custom", "custom_anthropic"
    )) {
        $preset = $presetCatalog.presets.PSObject.Properties[$presetId].Value
        Assert-True ($null -eq $preset.PSObject.Properties["model"]) `
            "Preset '$presetId' fixed a model name."
    }

    Initialize-UnifiedDirectories -Layout $offline
    Write-LlmPrivateConfiguration -Path $offline.ApiConfig `
        -ApiBase "https://provider.example/v1/" -ApiKey "private-test-key" `
        -Model "test-model" -ApiMode chat_completions -TimeoutSeconds 15
    $configuration = Read-ApiPrivateEnvironment -Path $offline.ApiConfig
    Assert-True ((Get-LlmConfigurationState -Environment $configuration) -eq "Complete") `
        "Complete LLM configuration was not recognized."
    Assert-True ($configuration.LLM_API_BASE -eq "https://provider.example/v1") `
        "API Base was not normalized."
    Assert-True ($configuration.LLM_VERIFICATION_STATUS -eq "configured_unverified") `
        "New LLM configuration was not marked unverified."
    Set-LlmVerificationStatus -Path $offline.ApiConfig -Status call_succeeded
    $verifiedConfiguration = Read-ApiPrivateEnvironment -Path $offline.ApiConfig
    Assert-True ($verifiedConfiguration.LLM_VERIFICATION_STATUS -eq "call_succeeded") `
        "Successful LLM verification state was not persisted."
    Assert-True ($verifiedConfiguration.LLM_API_KEY -eq "private-test-key") `
        "Verification status update changed the API key."
    Set-LlmVerificationStatus -Path $offline.ApiConfig -Status fallback
    Assert-True (
        (Read-ApiPrivateEnvironment -Path $offline.ApiConfig).LLM_VERIFICATION_STATUS -eq "fallback"
    ) "Fallback LLM verification state was not persisted."
    Assert-True (@(Get-ChildItem -LiteralPath $offline.ConfigRoot -Filter "*.tmp" -Force).Count -eq 0) `
        "Atomic configuration left a temporary file behind."
    Write-LlmPrivateConfiguration -Path $offline.ApiConfig `
        -ApiBase "https://provider.example/v1" -ApiKey "replacement-test-key" `
        -Model "replacement-model" -ApiMode responses -TimeoutSeconds 12
    $replaced = Read-ApiPrivateEnvironment -Path $offline.ApiConfig
    Assert-True ($replaced["LLM_MODEL"] -eq "replacement-model") `
        "Atomic replacement did not publish the new configuration."
    Assert-True (@(Get-ChildItem -LiteralPath $offline.ConfigRoot -Filter "*.tmp" -Force).Count -eq 0) `
        "Atomic replacement left a temporary file behind."

    Assert-Throws {
        Normalize-LlmApiBase -ApiBase "http://provider.example/v1"
    } "Remote HTTP API Base was accepted."
    Assert-Throws {
        Normalize-LlmApiBase -ApiBase "https://user:secret@provider.example/v1"
    } "Embedded API Base credentials were accepted."
    Assert-Throws {
        Normalize-LlmApiBase -ApiBase "https://provider.example/v1?tenant=x"
    } "API Base query string was accepted."
    Assert-Throws {
        Normalize-LlmApiBase -ApiBase "https://https://provider.example/v1"
    } "Repeated protocol prefix was accepted."
    $loopback = Normalize-LlmApiBase -ApiBase "http://127.0.0.1:11434/v1/" `
        -AllowInsecureLoopback
    Assert-True ($loopback -eq "http://127.0.0.1:11434/v1") `
        "Explicit loopback HTTP was not normalized."

    $missing = Get-UnifiedLayout -ProjectRoot (Join-Path $testRoot "missing") -Profile Offline
    Initialize-UnifiedDirectories -Layout $missing
    Assert-True (-not (Resolve-LlmStartup -Layout $missing -LlmMode Optional -NoLlmPrompt)) `
        "Non-interactive Optional mode did not fall back offline."
    Assert-True (-not (Resolve-LlmStartup -Layout $missing -LlmMode Disabled -NoLlmPrompt)) `
        "Disabled mode enabled LLM access."
    Assert-Throws {
        Resolve-LlmStartup -Layout $missing -LlmMode Required -NoLlmPrompt
    } "Required mode accepted missing configuration."
    [IO.File]::WriteAllText(
        $missing.ApiConfig,
        "LLM_API_BASE=https://provider.example/v1`n",
        [Text.UTF8Encoding]::new($false)
    )
    Assert-True (
        (Get-LlmConfigurationState -Environment (
            Read-ApiPrivateEnvironment -Path $missing.ApiConfig
        )) -eq "Partial"
    ) "Partial LLM configuration was not recognized."
    Assert-Throws {
        Resolve-LlmStartup -Layout $missing -LlmMode Optional -NoLlmPrompt
    } "Optional mode silently accepted partial configuration."

    $interactiveRoot = Join-Path $testRoot "interactive"
    $interactive = Get-UnifiedLayout -ProjectRoot $interactiveRoot -Profile Offline
    Initialize-UnifiedDirectories -Layout $interactive
    $savedInteractiveProbe = ${function:Test-UnifiedInteractiveHost}
    try {
        function Test-UnifiedInteractiveHost { return $true }
        function global:Read-Host { return $global:SocialGraphStartupPromptChoice }

        $global:SocialGraphStartupPromptChoice = "O"
        Assert-True (-not (Resolve-LlmStartup -Layout $interactive -LlmMode Optional)) `
            "Interactive O choice did not continue offline."
        Assert-Throws {
            Resolve-LlmStartup -Layout $interactive -LlmMode Required
        } "Interactive O choice bypassed Required mode."

        $global:SocialGraphStartupPromptChoice = "Q"
        Assert-Throws {
            Resolve-LlmStartup -Layout $interactive -LlmMode Optional
        } "Interactive Q choice did not cancel startup."
        $global:SocialGraphStartupPromptChoice = "X"
        Assert-Throws {
            Resolve-LlmStartup -Layout $interactive -LlmMode Optional
        } "Unknown interactive choice was accepted."

        $fixtureScripts = Join-Path $interactiveRoot "scripts"
        New-Item -ItemType Directory -Force -Path $fixtureScripts | Out-Null
        $escapedConfig = $interactive.ApiConfig.Replace("'", "''")
        $escapedOperations = (Join-Path $scriptsRoot "lib\UnifiedOperations.ps1").Replace("'", "''")
        [IO.File]::WriteAllText(
            (Join-Path $fixtureScripts "configure-llm.ps1"),
            ". '$escapedOperations'`nWrite-LlmPrivateConfiguration -Path '$escapedConfig' -ApiBase 'https://provider.example/v1' -ApiKey 'interactive-test-key' -Model 'test-model' -ApiMode chat_completions -TimeoutSeconds 15`n",
            [Text.UTF8Encoding]::new($false)
        )
        $global:SocialGraphStartupPromptChoice = "C"
        Assert-True (Resolve-LlmStartup -Layout $interactive -LlmMode Optional) `
            "Interactive C choice did not complete configuration."
    }
    finally {
        Remove-Item Function:\global:Read-Host -ErrorAction SilentlyContinue
        Remove-Variable SocialGraphStartupPromptChoice -Scope Global -ErrorAction SilentlyContinue
        ${function:Test-UnifiedInteractiveHost} = $savedInteractiveProbe
    }

    $savedKey = [Environment]::GetEnvironmentVariable("LLM_API_KEY", "Process")
    $savedRelay = [Environment]::GetEnvironmentVariable("LLM_API_RELAY_TOKEN", "Process")
    try {
        $env:LLM_API_KEY = "parent-key"
        $env:LLM_API_RELAY_TOKEN = "parent-relay"
        Invoke-WithProcessEnvironment -Environment (Get-ClearedLlmEnvironment) -ScriptBlock {
            Assert-True ([string]::IsNullOrEmpty($env:LLM_API_KEY)) `
                "The key was not cleared for a child process."
            Assert-True ([string]::IsNullOrEmpty($env:LLM_API_RELAY_TOKEN)) `
                "A dynamically named LLM_API_* variable reached a child environment."
        }
        Assert-True ($env:LLM_API_KEY -eq "parent-key") "The launcher did not restore its parent environment."

        $ports = [pscustomobject]@{ GovernanceWeb = 5173; Api = 8000; Gfm = 8766 }
        $offlineServices = @(Get-UnifiedServices -Layout $offline -Ports $ports `
            -Mode Development -EnableLlm:$true)
        Assert-True ($offlineServices.Count -eq 2) "Offline profile unexpectedly started GFM."
        $api = @($offlineServices | Where-Object Name -eq "socialgraph-api")[0]
        $web = @($offlineServices | Where-Object Name -eq "governance-web")[0]
        Assert-True ($api.Environment.LLM_API_KEY -eq "replacement-test-key") `
            "The API child did not receive its private whitelist."
        Assert-True ([string]::IsNullOrEmpty($web.Environment.LLM_API_KEY)) `
            "The Web child received the LLM key."

        $cudaServices = @(Get-UnifiedServices -Layout $cuda -Ports $ports `
            -Mode Development -EnableLlm:$false)
        $gfm = @($cudaServices | Where-Object Name -eq "gfm")[0]
        Assert-True ([string]::IsNullOrEmpty($gfm.Environment.LLM_API_KEY)) `
            "The GFM child received the LLM key."
        Assert-True (($gfm.Arguments -join " ").Contains($cuda.GlobalModelRoot)) `
            "GFM did not use the isolated authorized model root."
    }
    finally {
        [Environment]::SetEnvironmentVariable("LLM_API_KEY", $savedKey, "Process")
        [Environment]::SetEnvironmentVariable("LLM_API_RELAY_TOKEN", $savedRelay, "Process")
    }

    Write-UnifiedRuntimeProfile -Layout $offline -Profile Offline
    Assert-True ((Get-UnifiedRuntimeProfile -ProjectRoot $testRoot) -eq "Offline") `
        "Runtime profile did not pass atomic read-back."
}
finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    if (-not $resolved.StartsWith($systemTemp, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a startup-test path outside the system temporary directory."
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Startup script tests passed."
