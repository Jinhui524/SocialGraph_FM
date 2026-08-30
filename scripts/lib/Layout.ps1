Set-StrictMode -Version Latest

function Get-UnifiedRuntimeProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [ValidateSet("Offline", "Cpu", "Cuda")][string]$Fallback = "Offline"
    )

    $profilePath = Join-Path ([IO.Path]::GetFullPath($ProjectRoot)) "var\config\runtime-profile.json"
    if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
        return $Fallback
    }
    try {
        $document = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "The runtime profile is invalid. Run scripts\setup.ps1 again: $profilePath"
    }
    if ($document.schemaVersion -eq "socialgraph-fm.runtime-profile/1.0" -and
        [string]$document.profile -in @("Offline", "Cpu", "Cuda")) {
        return [string]$document.profile
    }
    if ($document.schemaVersion -in @(
            "socialgraph-fm.runtime-profile/2.0",
            "socialgraph-fm.runtime-profile/3.0"
        ) -and
        [string]$document.profile -in @("offline", "cpu", "cuda")) {
        return (Get-Culture).TextInfo.ToTitleCase([string]$document.profile)
    }
    throw "The runtime profile is unsupported. Run scripts\setup.ps1 again: $profilePath"
}

function Get-UnifiedLayout {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [ValidateSet("Offline", "Cpu", "Cuda")][string]$Profile
    )

    $root = [IO.Path]::GetFullPath($ProjectRoot)
    $selectedProfile = if ($PSBoundParameters.ContainsKey("Profile")) {
        $Profile
    }
    else {
        Get-UnifiedRuntimeProfile -ProjectRoot $root
    }
    $varRoot = Join-Path $root "var"
    $gfmHome = Join-Path $varRoot "gfm"
    $coreRuntime = Join-Path $gfmHome "core-runtime"
    $researchRoot = Join-Path $gfmHome "research"
    $globalModelSourceRoot = Join-Path $gfmHome "socialgraph-global"
    $governanceRoot = Join-Path $gfmHome "governance"
    $authorizedModelRoot = Join-Path $varRoot "models\socialgraph-global"
    $runtimeBundleRoot = Join-Path $root "bundles"
    $servingRoot = Join-Path $coreRuntime "serving"
    $configRoot = Join-Path $varRoot "config"
    $cacheRoot = Join-Path $gfmHome "cache"
    $governanceStateRoot = Join-Path $varRoot "governance"
    $pythonRelative = if ($env:OS -eq "Windows_NT") { "Scripts\python.exe" } else { "bin/python" }
    $profileDirectory = switch ($selectedProfile) {
        "Cpu" { "gfm-cpu" }
        "Cuda" { "gfm-cu130-clean" }
        default { "gfm-offline" }
    }
    $apiPython = Join-Path (Join-Path $varRoot "envs\socialgraph-api") $pythonRelative
    $gfmPython = Join-Path (Join-Path $varRoot "envs\$profileDirectory") $pythonRelative
    $runtimeProfileDocument = $null
    $runtimeProfilePath = Join-Path $configRoot "runtime-profile.json"
    if (Test-Path -LiteralPath $runtimeProfilePath -PathType Leaf) {
        try {
            $candidateDocument = Get-Content -LiteralPath $runtimeProfilePath -Raw |
                ConvertFrom-Json -ErrorAction Stop
            if ($candidateDocument.schemaVersion -in @(
                    "socialgraph-fm.runtime-profile/2.0",
                    "socialgraph-fm.runtime-profile/3.0"
                )) {
                $runtimeProfileDocument = $candidateDocument
            }
        }
        catch {
            throw "The runtime profile is invalid. Run scripts\setup.ps1 again: $runtimeProfilePath"
        }
    }
    if ($null -ne $runtimeProfileDocument) {
        $apiRecord = $runtimeProfileDocument.interpreters.api
        if ($null -ne $apiRecord -and -not [string]::IsNullOrWhiteSpace([string]$apiRecord.path)) {
            $apiPython = [IO.Path]::GetFullPath([string]$apiRecord.path)
        }
        $recordedProfile = (Get-Culture).TextInfo.ToTitleCase(
            [string]$runtimeProfileDocument.profile
        )
        $gfmRecord = $runtimeProfileDocument.interpreters.gfm
        if ($recordedProfile -eq $selectedProfile -and $null -ne $gfmRecord -and
            -not [string]::IsNullOrWhiteSpace([string]$gfmRecord.path)) {
            $gfmPython = [IO.Path]::GetFullPath([string]$gfmRecord.path)
        }
    }
    $apiParent = Split-Path -Parent $apiPython
    $gfmParent = Split-Path -Parent $gfmPython
    $apiEnvironmentRoot = if ((Split-Path -Leaf $apiParent) -in @("Scripts", "bin")) {
        Split-Path -Parent $apiParent
    }
    else { $apiParent }
    $gfmEnvironmentRoot = if ((Split-Path -Leaf $gfmParent) -in @("Scripts", "bin")) {
        Split-Path -Parent $gfmParent
    }
    else { $gfmParent }
    $uvRelative = if ($env:OS -eq "Windows_NT") { "uv.exe" } else { "uv" }

    [pscustomobject]@{
        ProjectRoot = $root
        RuntimeProfile = $selectedProfile
        VarRoot = $varRoot
        GfmHome = $gfmHome
        ManagedPythonRoot = Join-Path $gfmHome "python"
        ManagedPythonVersion = "3.12.13"
        CacheRoot = $cacheRoot
        ConfigRoot = $configRoot
        RuntimeProfileFile = Join-Path $configRoot "runtime-profile.json"
        GovernanceStateRoot = $governanceStateRoot
        GovernanceAdaptationInputs = Join-Path $governanceStateRoot "adaptation-inputs"
        GlobalModelCorpus = Join-Path $globalModelSourceRoot "corpus"
        ApiConfig = Join-Path $configRoot "socialgraph-api.env"
        CoreRuntime = $coreRuntime
        ResearchRoot = $researchRoot
        GlobalModelSourceRoot = $globalModelSourceRoot
        GlobalModelRoot = $authorizedModelRoot
        GovernanceRoot = $governanceRoot
        RuntimeBundleRoot = $runtimeBundleRoot
        RuntimeBundleManifest = Join-Path $runtimeBundleRoot "runtime-manifest.json"
        RuntimeBundleModelRoot = Join-Path $runtimeBundleRoot "models\socialgraph-global"
        RuntimeBundleGovernanceRoot = Join-Path $runtimeBundleRoot "governance"
        RuntimeExamplesRoot = Join-Path $root "examples\governance"
        ServingRoot = $servingRoot
        PublishedServingRoot = $servingRoot
        ServingControl = Join-Path $servingRoot "core-serving-control.json"
        PublishedServingControl = Join-Path $servingRoot "core-serving-control.json"
        ServingToken = Join-Path $servingRoot "session.token"
        ServingArtifacts = Join-Path $coreRuntime "serving-graphs"
        PublishedServingArtifacts = Join-Path $coreRuntime "serving-graphs"
        ApiData = Join-Path $coreRuntime "api\dataset-store"
        ApiBindings = Join-Path $coreRuntime "api\gfm-run-bindings"
        ApiResearchBindings = Join-Path $coreRuntime "api\gfm-research-run-bindings"
        GlobalModelBindings = Join-Path $coreRuntime "api\gfm-global-model-run-bindings"
        GlobalModelReviews = Join-Path $coreRuntime "api\gfm-global-model-reviews"
        ApiHighWater = Join-Path $coreRuntime "api\serving-control"
        DeployRoot = Join-Path $varRoot "deploy"
        LogRoot = Join-Path $varRoot "deploy\logs"
        PidRoot = Join-Path $varRoot "deploy\pids"
        TempRoot = Join-Path $varRoot "tmp"
        ApiEnvironmentRoot = $apiEnvironmentRoot
        ApiPython = $apiPython
        GfmEnvironmentRoot = $gfmEnvironmentRoot
        GfmPython = $gfmPython
        UvRoot = Join-Path $varRoot "tools\uv-0.12.3"
        UvExecutable = Join-Path (Join-Path $varRoot "tools\uv-0.12.3") $uvRelative
        GfmUv = Join-Path (Join-Path $varRoot "tools\uv-0.12.3") $uvRelative
        LegacyGfmUv = Join-Path $gfmHome "tools\uv-0.12.3\Scripts\uv.exe"
        GovernanceWeb = Join-Path $root "apps\web"
        Api = Join-Path $root "services\api"
        GfmPackage = Join-Path $root "packages\gfm"
        ContractsRoot = Join-Path $root "contracts"
        ProductSkillsRoot = Join-Path $root "skills\governance"
        GovernanceVerification = $false
    }
}

function Get-UnifiedPort {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][int]$Default
    )

    $raw = [Environment]::GetEnvironmentVariable($EnvironmentName, "Process")
    $port = $Default
    if (-not [string]::IsNullOrWhiteSpace($raw) -and -not [int]::TryParse($raw, [ref]$port)) {
        throw "$EnvironmentName must be an integer."
    }
    if ($port -lt 1 -or $port -gt 65535) {
        throw "$EnvironmentName must be between 1 and 65535."
    }
    return $port
}

function Get-UnifiedPorts {
    [pscustomobject]@{
        GovernanceWeb = Get-UnifiedPort -EnvironmentName "SOCIALGRAPH_GOVERNANCE_WEB_PORT" -Default 5173
        Api = Get-UnifiedPort -EnvironmentName "SOCIALGRAPH_CORE_API_PORT" -Default 8000
        Gfm = Get-UnifiedPort -EnvironmentName "SOCIALGRAPH_GFM_PORT" -Default 8766
    }
}

function Set-UnifiedEnvironment {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Ports
    )

    $env:SOCIALGRAPH_FM_HOME = $Layout.GfmHome
    $env:SOCIALGRAPH_GFM_PYTHON = $Layout.GfmPython
    $env:PIP_CACHE_DIR = Join-Path $Layout.CacheRoot "pip"
    $env:UV_CACHE_DIR = Join-Path $Layout.CacheRoot "uv"
    $env:UV_PYTHON_INSTALL_DIR = $Layout.ManagedPythonRoot
    $env:HF_HOME = Join-Path $Layout.CacheRoot "hf"
    $env:TORCH_HOME = Join-Path $Layout.CacheRoot "torch"
    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $Layout.CacheRoot "torchinductor"
    $env:WANDB_DIR = Join-Path $Layout.CacheRoot "wandb"
    $env:SOCIALGRAPH_GOVERNANCE_WEB_PORT = [string]$Ports.GovernanceWeb
    $env:SOCIALGRAPH_CORE_API_PORT = [string]$Ports.Api
    $env:SOCIALGRAPH_GFM_PORT = [string]$Ports.Gfm
    $env:TEMP = $Layout.TempRoot
    $env:TMP = $Layout.TempRoot
    $env:VITE_SOCIALGRAPH_API_BASE_URL = "http://127.0.0.1:$($Ports.Api)"
}
