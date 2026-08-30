Set-StrictMode -Version Latest

function Assert-UnifiedRuntimeReady {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode
    )

    $required = @(
        $Layout.ApiPython,
        (Join-Path $Layout.GovernanceWeb "node_modules\vite\bin\vite.js")
    )
    if ($Layout.RuntimeProfile -ne "Offline") {
        $required += @($Layout.GfmPython, $Layout.ServingControl)
    }
    if ($Mode -eq "Production") {
        $required += Join-Path $Layout.GovernanceWeb "dist\client\index.html"
    }
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) {
        throw "Unified runtime is incomplete. Run scripts\setup.ps1 first. Missing:`n$($missing -join "`n")"
    }
    Assert-ApiIsTorchFree -PythonExecutable $Layout.ApiPython
}

function Get-UnifiedServices {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Ports,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode,
        [bool]$EnableLlm = $true
    )

    $node = (Get-Command node -ErrorAction Stop).Source
    $governanceVite = Join-Path $Layout.GovernanceWeb "node_modules\vite\bin\vite.js"
    $viteCommand = if ($Mode -eq "Development") { "dev" } else { "preview" }
    $common = @{
        SOCIALGRAPH_FM_HOME = $Layout.GfmHome
        TEMP = $Layout.TempRoot
        TMP = $Layout.TempRoot
        PIP_CACHE_DIR = Join-Path $Layout.CacheRoot "pip"
        UV_CACHE_DIR = Join-Path $Layout.CacheRoot "uv"
        HF_HOME = Join-Path $Layout.CacheRoot "hf"
        TORCH_HOME = Join-Path $Layout.CacheRoot "torch"
        TORCHINDUCTOR_CACHE_DIR = Join-Path $Layout.CacheRoot "torchinductor"
        WANDB_DIR = Join-Path $Layout.CacheRoot "wandb"
    }
    $clearedLlm = Get-ClearedLlmEnvironment

    $gfmEnvironment = @{} + $common + $clearedLlm
    $gfmEnvironment.PYTHONPATH = Join-Path $Layout.GfmPackage "src"
    $gfmEnvironment.PYTHONNOUSERSITE = "1"

    $apiEnvironment = @{} + $common + $clearedLlm
    if ($EnableLlm) {
        $privateEnvironment = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
        if ((Get-LlmConfigurationState -Environment $privateEnvironment) -ne "Complete") {
            throw "The launcher enabled LLM access without a complete private configuration."
        }
        foreach ($name in $privateEnvironment.Keys) {
            $apiEnvironment[$name] = $privateEnvironment[$name]
        }
    }
    $publishedControlProperty = $Layout.PSObject.Properties["PublishedServingControl"]
    $publishedServingControl = if ($null -ne $publishedControlProperty) {
        [string]$publishedControlProperty.Value
    }
    else {
        [string]$Layout.ServingControl
    }
    $gfmEnabled = $Layout.RuntimeProfile -ne "Offline"
    $apiEnvironment.GFM_INFRASTRUCTURE_READY = "false"
    $apiEnvironment.SOCIALGRAPH_CORE_API_PORT = [string]$Ports.Api
    $apiEnvironment.GFM_SERVICE_URL = if ($gfmEnabled) { "http://127.0.0.1:$($Ports.Gfm)" } else { "" }
    $apiEnvironment.GFM_SESSION_TOKEN_FILE = if ($gfmEnabled) { $Layout.ServingToken } else { "" }
    $apiEnvironment.GFM_CORE_SERVING_CONTROL_FILE = if ($gfmEnabled) { $publishedServingControl } else { "" }
    $apiEnvironment.GFM_CORE_RUN_BINDING_ROOT = $Layout.ApiBindings
    $apiEnvironment.GFM_RESEARCH_RUN_BINDING_ROOT = $Layout.ApiResearchBindings
    $apiEnvironment.GFM_GLOBAL_MODEL_RUN_BINDING_ROOT = $Layout.GlobalModelBindings
    $apiEnvironment.GFM_GLOBAL_MODEL_REVIEW_ROOT = $Layout.GlobalModelReviews
    $apiEnvironment.GFM_GOVERNANCE_ROOT = $Layout.GovernanceRoot
    $apiEnvironment.GFM_GOVERNANCE_BUNDLE_MAX_BYTES = "268435456"
    $apiEnvironment.GFM_GOVERNANCE_EXPANDED_MAX_BYTES = "1073741824"
    $apiEnvironment.GFM_CORE_SERVING_HIGH_WATER_ROOT = $Layout.ApiHighWater
    $apiEnvironment.DATASET_STORAGE_ROOT = $Layout.ApiData
    $apiEnvironment.ALLOWED_ORIGINS = "http://127.0.0.1:$($Ports.GovernanceWeb),http://localhost:$($Ports.GovernanceWeb)"
    $apiEnvironment.ENABLE_TRUSTED_LOCAL_CONVERSION = "false"
    $apiEnvironment.TRUSTED_DATA_ROOTS = ""
    $apiEnvironment.TRUSTED_CONVERTER_PYTHON = ""
    $apiEnvironment.LOCAL_DEMO_LOOPBACK_ONLY = "true"
    $apiEnvironment.RUNTIME_BUILD_ID = "unified-local"
    $apiEnvironment.PYTHONPATH = ""
    $apiEnvironment.PYTHONNOUSERSITE = "1"
    $governanceProperty = $Layout.PSObject.Properties["GovernanceVerification"]
    if ($null -ne $governanceProperty -and [bool]$governanceProperty.Value) {
        foreach ($name in Get-LlmSensitiveEnvironmentNames) {
            $apiEnvironment[$name] = $null
        }
    }

    $governanceWebEnvironment = @{} + $common + $clearedLlm
    $governanceWebEnvironment.VITE_SOCIALGRAPH_API_BASE_URL = "http://127.0.0.1:$($Ports.Api)"

    $services = [Collections.Generic.List[object]]::new()
    if ($gfmEnabled) {
        $publishedServingRoot = $Layout.PublishedServingRoot
        $publishedArtifacts = $Layout.PublishedServingArtifacts
        $globalModelDevice = if ($Layout.RuntimeProfile -eq "Cpu") { "cpu" } else { "auto" }
        $services.Add([pscustomobject]@{
            Name = "gfm"
            Port = $Ports.Gfm
            Executable = $Layout.GfmPython
            WorkingDirectory = $Layout.GfmPackage
            Arguments = @(
                "-m", "socialgraph_gfm.core.inference_cli",
                "--runtime-root", $Layout.ServingRoot,
                "--serving-control", $publishedServingControl,
                "--published-serving-root", $publishedServingRoot,
                "--published-artifact-root", $publishedArtifacts,
                "--artifact-root", $Layout.ServingArtifacts,
                "--research-root", $Layout.ResearchRoot,
                "--global-model-root", $Layout.GlobalModelRoot,
                "--governance-root", $Layout.GovernanceRoot,
                "--global-model-device", $globalModelDevice,
                "--dataset-store-root", $Layout.ApiData,
                "--token-file", $Layout.ServingToken,
                "--host", "127.0.0.1", "--port", [string]$Ports.Gfm
            )
            IdentityTokens = @(
                "socialgraph_gfm.core.inference_cli",
                "--runtime-root", $Layout.ServingRoot,
                "--serving-control", $publishedServingControl,
                "--published-serving-root", $publishedServingRoot,
                "--published-artifact-root", $publishedArtifacts,
                "--artifact-root", $Layout.ServingArtifacts,
                "--global-model-root", $Layout.GlobalModelRoot,
                "--governance-root", $Layout.GovernanceRoot,
                "--dataset-store-root", $Layout.ApiData,
                "--token-file", $Layout.ServingToken,
                "--port", [string]$Ports.Gfm
            )
            Environment = $gfmEnvironment
        })
    }
    $services.Add([pscustomobject]@{
        Name = "socialgraph-api"
        Port = $Ports.Api
        Executable = $Layout.ApiPython
        WorkingDirectory = $Layout.Api
        Arguments = @("-m", "app", "--runtime-identity-root", $Layout.GovernanceRoot)
        IdentityTokens = @("-m", "app", "--runtime-identity-root", $Layout.GovernanceRoot)
        Environment = $apiEnvironment
    })
    $services.Add([pscustomobject]@{
        Name = "governance-web"
        Port = $Ports.GovernanceWeb
        Executable = $node
        WorkingDirectory = $Layout.GovernanceWeb
        Arguments = @(
            $governanceVite, $viteCommand, "--host", "127.0.0.1",
            "--port", [string]$Ports.GovernanceWeb, "--strictPort"
        )
        IdentityTokens = @($governanceVite, "--port", [string]$Ports.GovernanceWeb)
        Environment = $governanceWebEnvironment
    })
    return @($services)
}

function Start-UnifiedStack {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Ports,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode,
        [bool]$EnableLlm = $true
    )

    Assert-UnifiedRuntimeReady -Layout $Layout -Mode $Mode
    Initialize-UnifiedDirectories -Layout $Layout
    $started = [Collections.Generic.List[string]]::new()
    try {
        foreach ($service in Get-UnifiedServices `
            -Layout $Layout -Ports $Ports -Mode $Mode -EnableLlm:$EnableLlm) {
            if (Start-ManagedService -Layout $Layout -Service $service) {
                $started.Add($service.Name)
            }
        }
    }
    catch {
        for ($index = $started.Count - 1; $index -ge 0; $index--) {
            Stop-ManagedService -Layout $Layout -Name $started[$index]
        }
        throw
    }
    Write-Host "Governance: http://127.0.0.1:$($Ports.GovernanceWeb)"
    Write-Host "SocialGraph-FM API: http://127.0.0.1:$($Ports.Api)"
    if ($Layout.RuntimeProfile -eq "Offline") {
        Write-Host "GFM: offline profile (model runtime not started)"
    }
    else {
        Write-Host "GFM loopback: http://127.0.0.1:$($Ports.Gfm)"
    }
    Write-Host "LLM: $(if ($EnableLlm) { 'configured for the API process' } else { 'deterministic fallback' })"
    Write-Host "Logs: $($Layout.LogRoot)"
}
