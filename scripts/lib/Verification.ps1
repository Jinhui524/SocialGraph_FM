Set-StrictMode -Version Latest

function Test-LlmConfigurationEnvironment {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][hashtable]$Environment
    )

    $environment = $Environment
    if ((Get-LlmConfigurationState -Environment $environment) -ne "Complete") {
        throw "A complete LLM configuration is required for -TestLlm."
    }
    if (-not (Test-Path -LiteralPath $Layout.ApiPython -PathType Leaf)) {
        throw "The API environment is missing. Run scripts\setup.ps1 before testing LLM access."
    }
    $processEnvironment = @{} + (Get-ClearedLlmEnvironment)
    foreach ($name in @(
        "LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL", "LLM_API_MODE",
        "LLM_AUTH_SCHEME", "LLM_ANTHROPIC_VERSION",
        "LLM_TIMEOUT_SECONDS", "LLM_ALLOW_INSECURE_LOOPBACK",
        "LLM_VERIFICATION_STATUS"
    )) {
        if ($environment.ContainsKey($name)) {
            $processEnvironment[$name] = $environment[$name]
        }
    }
    $processEnvironment["PYTHONNOUSERSITE"] = "1"
    $processEnvironment["PYTHONPATH"] = ""
    $exitCode = Invoke-WithProcessEnvironment -Environment $processEnvironment -ScriptBlock {
        Push-Location $Layout.Api
        try {
            $null = & $Layout.ApiPython -m app.provider_check 2>&1
            return $LASTEXITCODE
        }
        finally {
            Pop-Location
        }
    }
    if ([int]$exitCode -ne 0) {
        throw "LLM connection check failed provider response validation."
    }
    Write-Host "LLM connection check passed for model $($environment['LLM_MODEL'])."
    return $true
}

function Test-LlmPrivateConfiguration {
    param([Parameter(Mandatory = $true)]$Layout)

    $environment = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
    try {
        $passed = Test-LlmConfigurationEnvironment -Layout $Layout -Environment $environment
        Set-LlmVerificationStatus -Path $Layout.ApiConfig -Status call_succeeded
        return $passed
    }
    catch {
        try {
            if ((Get-LlmConfigurationState -Environment $environment) -eq "Complete") {
                Set-LlmVerificationStatus -Path $Layout.ApiConfig -Status fallback
            }
        }
        catch {
            # Preserve the original provider validation failure.
        }
        throw
    }
}

function Invoke-UnifiedDoctor {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [switch]$TestLlm,
        [switch]$AsJson
    )

    $checks = [Collections.Generic.List[object]]::new()
    $add = {
        param([string]$Name, [bool]$Passed, [string]$Detail, [bool]$Required = $true)
        $checks.Add([ordered]@{
            name = $Name
            passed = $Passed
            required = $Required
            detail = $Detail
        })
    }
    & $add "repository-layout" (
        (Test-Path -LiteralPath $Layout.GovernanceWeb -PathType Container) -and
        (Test-Path -LiteralPath $Layout.Api -PathType Container) -and
        (Test-Path -LiteralPath $Layout.GfmPackage -PathType Container)
    ) "apps/web, services/api, packages/gfm"
    $apiExists = Test-Path -LiteralPath $Layout.ApiPython -PathType Leaf
    & $add "api-environment" $apiExists $Layout.ApiPython
    if ($apiExists) {
        & $Layout.ApiPython -m pip check *> $null
        $apiPipReady = $LASTEXITCODE -eq 0
        & $Layout.ApiPython -c `
            "import fastapi,httpx,numpy,pydantic,pydantic_settings,uvicorn; print('ready')" *> $null
        $apiImportsReady = $LASTEXITCODE -eq 0
        $torchFree = $true
        try {
            Assert-ApiIsTorchFree -PythonExecutable $Layout.ApiPython
        }
        catch {
            $torchFree = $false
        }
        & $add "api-dependencies" ($apiPipReady -and $apiImportsReady -and $torchFree) `
            "pip check, import smoke, and Torch-free boundary"
    }
    & $add "web-dependencies" (
        Test-Path -LiteralPath (Join-Path $Layout.GovernanceWeb "node_modules\vite\bin\vite.js") -PathType Leaf
    ) $Layout.GovernanceWeb
    if ($Layout.RuntimeProfile -ne "Offline") {
        $gfmExists = Test-Path -LiteralPath $Layout.GfmPython -PathType Leaf
        & $add "gfm-environment" $gfmExists $Layout.GfmPython
        if ($gfmExists) {
            $cleared = Get-ClearedLlmEnvironment
            $gfmReady = [bool](Invoke-WithProcessEnvironment -Environment $cleared -ScriptBlock {
                & $Layout.GfmPython -m pip check *> $null
                $pipReady = $LASTEXITCODE -eq 0
                & $Layout.GfmPython -c `
                    "import torch,torch_geometric; print(torch.__version__,torch_geometric.__version__)" *> $null
                $importReady = $LASTEXITCODE -eq 0
                $pipReady -and $importReady
            })
            & $add "gfm-dependencies" $gfmReady "pip check plus Torch/PyG import smoke"
        }
    }
    $contractsReady = $true
    foreach ($name in @(
        "core-serving-control.json",
        "core-serving-graph-catalog.json",
        "core-serving-registry.json"
    )) {
        $canonical = Join-Path $Layout.ContractsRoot "core\serving\$name"
        foreach ($mirror in @(
            (Join-Path $Layout.Api "app\contracts\$name"),
            (Join-Path $Layout.GfmPackage "contracts\$name")
        )) {
            if (-not (Test-Path -LiteralPath $canonical -PathType Leaf) -or
                -not (Test-Path -LiteralPath $mirror -PathType Leaf) -or
                (Get-FileHash -LiteralPath $canonical -Algorithm SHA256).Hash -ne
                    (Get-FileHash -LiteralPath $mirror -Algorithm SHA256).Hash) {
                $contractsReady = $false
            }
        }
    }
    $publicReadiness = Join-Path $Layout.ProjectRoot "docs\status\readiness.json"
    $packageReadiness = Join-Path $Layout.GfmPackage "contracts\core-readiness.json"
    $readinessReady = (
        (Test-Path -LiteralPath $publicReadiness -PathType Leaf) -and
        (Test-Path -LiteralPath $packageReadiness -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $publicReadiness -Algorithm SHA256).Hash -eq
            (Get-FileHash -LiteralPath $packageReadiness -Algorithm SHA256).Hash)
    )
    & $add "contracts" $contractsReady "canonical serving contracts match API/GFM mirrors"
    & $add "readiness" $readinessReady "public and packaged SocialGraph-FM Core readiness records match"

    $processesReady = $true
    $recordedCount = 0
    foreach ($name in @("governance-web", "socialgraph-api", "gfm")) {
        $record = Read-ManagedPidRecord -Layout $Layout -Name $name
        if ($null -eq $record) { continue }
        $recordedCount += 1
        $processInfo = Get-ManagedProcessInfo -ProcessId ([int]$record.pid)
        if ($null -eq $processInfo -or
            -not (Test-RecordedProcessIdentity -Record $record -ProcessInfo $processInfo) -or
            -not (Test-LoopbackPort -Port ([int]$record.port))) {
            $processesReady = $false
        }
    }
    & $add "managed-processes" $processesReady `
        $(if ($recordedCount -eq 0) { "stack stopped; no stale PID records" } else { "$recordedCount managed service(s) verified" })
    $llmEnvironment = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
    $llmState = Get-LlmConfigurationState -Environment $llmEnvironment
    & $add "llm-configuration" ($llmState -ne "Partial") $llmState
    if ($TestLlm) {
        $passed = Test-LlmPrivateConfiguration -Layout $Layout
        & $add "llm-connectivity" $passed "provider returned JSON without redirects or proxy inheritance"
    }
    $modelRequired = $Layout.RuntimeProfile -ne "Offline"
    $modelReady = $false
    $modelDetail = "not installed"
    $registry = Join-Path $Layout.GlobalModelRoot "registry\socialgraph-global.json"
    if (Test-Path -LiteralPath $registry -PathType Leaf) {
        try {
            $bundle = Assert-RuntimeBundle -Layout $Layout
            $modelAssets = @($bundle.Assets | Where-Object role -eq "model")
            Assert-RuntimeBundleInstalledAssets -Assets $modelAssets `
                -SourcePrefix "bundles/models/socialgraph-global" `
                -Destination $Layout.GlobalModelRoot
            $published = Get-Content -LiteralPath $registry -Raw | ConvertFrom-Json -ErrorAction Stop
            $modelReady = (
                $published.state -eq "servingReady" -and
                $published.modelVersionHash -ceq [string]$bundle.Document.model.modelVersionHash -and
                $published.artifactHash -ceq [string]$bundle.Document.model.artifactHash -and
                $published.corpusHash -ceq [string]$bundle.Document.model.corpusHash
            )
            $modelDetail = if ($modelReady) {
                "manifest-bound Global model installed"
            }
            else { "installed registry identity differs from the public bundle" }
        }
        catch {
            $modelReady = $false
            $modelDetail = "installed model failed bundle verification"
        }
    }
    & $add "authorized-model" $modelReady $modelDetail $modelRequired

    $fatal = @($checks | Where-Object { $_.required -and -not $_.passed })
    $document = [ordered]@{
        schemaVersion = "socialgraph-fm.doctor/1.0"
        profile = $Layout.RuntimeProfile
        passed = $fatal.Count -eq 0
        checks = @($checks)
    }
    if ($AsJson) {
        $document | ConvertTo-Json -Depth 5
    }
    else {
        foreach ($check in $checks) {
            $marker = if ($check.passed) { "PASS" } elseif ($check.required) { "FAIL" } else { "INFO" }
            Write-Host "[$marker] $($check.name): $($check.detail)"
        }
    }
    # Offline keeps the model informational; CPU/CUDA profiles require the bundled runtime.
    if ($fatal.Count -gt 0) {
        throw "Doctor found $($fatal.Count) blocking environment problem(s)."
    }
}
