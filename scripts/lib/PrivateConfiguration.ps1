Set-StrictMode -Version Latest

function Read-LlmPresetCatalog {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "LLM preset catalog is missing: $Path"
    }
    try {
        $document = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 |
            ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "LLM preset catalog is not valid JSON: $Path"
    }
    if ($document.schemaVersion -ne "socialgraph-fm.llm-presets/2.0") {
        throw "LLM preset catalog schema is unsupported: $Path"
    }
    $expectedIds = @(
        "openai_responses", "deepseek", "glm", "anthropic", "custom", "custom_anthropic"
    )
    $actualIds = @($document.presets.PSObject.Properties.Name)
    if (@(Compare-Object $expectedIds $actualIds).Count -ne 0) {
        throw "LLM preset catalog must contain exactly: $($expectedIds -join ', ')."
    }
    foreach ($presetId in $expectedIds) {
        $preset = $document.presets.PSObject.Properties[$presetId].Value
        $expectedFields = @(
            "displayName", "connectionKind", "apiBase", "defaultApiMode", "allowedApiModes",
            "defaultAuthScheme", "allowedAuthSchemes", "anthropicVersion"
        )
        if (@(Compare-Object $expectedFields @($preset.PSObject.Properties.Name)).Count -ne 0) {
            throw "LLM preset '$presetId' has an unsupported shape."
        }
        Assert-LlmSingleLineValue -Name "LLM preset display name" -Value ([string]$preset.displayName)
        $allowedModes = @($preset.allowedApiModes)
        if ($allowedModes.Count -eq 0 -or
            @($allowedModes | Where-Object {
                $_ -notin @("chat_completions", "responses", "anthropic_messages")
            }).Count -ne 0 -or
            $preset.defaultApiMode -notin $allowedModes) {
            throw "LLM preset '$presetId' has invalid API modes."
        }
        $allowedAuthSchemes = @($preset.allowedAuthSchemes)
        if ($preset.connectionKind -notin @("direct", "custom_relay") -or
            $allowedAuthSchemes.Count -eq 0 -or
            @($allowedAuthSchemes | Where-Object { $_ -notin @("bearer", "x-api-key") }).Count -ne 0 -or
            $preset.defaultAuthScheme -notin $allowedAuthSchemes) {
            throw "LLM preset '$presetId' has invalid authentication metadata."
        }
        if ($preset.connectionKind -eq "custom_relay") {
            if ($null -ne $preset.apiBase) {
                throw "The custom LLM preset cannot fix an API Base."
            }
        }
        else {
            $normalized = Normalize-LlmApiBase -ApiBase ([string]$preset.apiBase)
            if ($normalized -ne $preset.apiBase) {
                throw "LLM preset '$presetId' API Base is not normalized."
            }
        }
    }
    return $document
}

function Get-LlmPreset {
    param(
        [Parameter(Mandatory = $true)][string]$CatalogPath,
        [Parameter(Mandatory = $true)]
        [ValidateSet("openai_responses", "deepseek", "glm", "anthropic", "custom", "custom_anthropic")]
        [string]$Preset
    )

    $catalog = Read-LlmPresetCatalog -Path $CatalogPath
    return $catalog.presets.PSObject.Properties[$Preset].Value
}

function Protect-UnifiedConfigDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($env:OS -ne "Windows_NT") { return }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The private configuration directory cannot be a reparse point: $Path"
    }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $icacls = Join-Path $env:SystemRoot "System32\icacls.exe"
    & $icacls $Path "/inheritance:r" "/grant:r" `
        "*$($currentSid):(OI)(CI)F" `
        "*S-1-5-18:(OI)(CI)F" `
        "*S-1-5-32-544:(OI)(CI)F" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not protect the private configuration directory: $Path"
    }
    Assert-PrivateConfigurationAcl -Path $Path -RequireProtectedRules
}

function Import-OsOwnedSecurityModule {
    if ($env:OS -ne "Windows_NT") { return }
    if ($PSVersionTable.PSEdition -eq "Desktop") {
        $manifest = Join-Path $env:SystemRoot `
            "System32\WindowsPowerShell\v1.0\Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
        if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
            throw "The OS-owned Microsoft.PowerShell.Security module is missing."
        }
        Remove-Module Microsoft.PowerShell.Security -Force -ErrorAction SilentlyContinue
        Import-Module $manifest -Force -ErrorAction Stop
    }
    elseif ($null -eq (Get-Command Get-Acl -ErrorAction SilentlyContinue)) {
        throw "The local PowerShell Security module is unavailable."
    }
}

function Assert-PrivateConfigurationAcl {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$RequireProtectedRules
    )

    if ($env:OS -ne "Windows_NT") { return }
    Import-OsOwnedSecurityModule
    $acl = Get-Acl -LiteralPath $Path
    if ($RequireProtectedRules -and -not $acl.AreAccessRulesProtected) {
        throw "Private configuration ACL inheritance must be disabled: $Path"
    }
    $currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowedSids = @($currentSid, "S-1-5-18", "S-1-5-32-544")
    $hasCurrentUser = $false
    foreach ($rule in $acl.Access) {
        if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) {
            continue
        }
        try {
            $sid = $rule.IdentityReference.Translate(
                [Security.Principal.SecurityIdentifier]
            ).Value
        }
        catch {
            throw "Could not verify a private configuration ACL identity: $($rule.IdentityReference)"
        }
        if ($allowedSids -notcontains $sid) {
            throw "Private configuration is accessible to an unexpected local principal: $sid"
        }
        if ($sid -eq $currentSid) { $hasCurrentUser = $true }
    }
    if (-not $hasCurrentUser) {
        throw "Private configuration does not grant access to the current user."
    }
}

function Assert-PrivateConfigurationFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "A private configuration file cannot be a reparse point: $Path"
    }
    Assert-PrivateConfigurationAcl -Path $Path
}

function Get-LlmSensitiveEnvironmentNames {
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    @(
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_API_MODE",
        "LLM_AUTH_SCHEME",
        "LLM_ANTHROPIC_VERSION",
        "LLM_TIMEOUT_SECONDS",
        "LLM_ALLOW_INSECURE_LOOPBACK",
        "LLM_VERIFICATION_STATUS",
        "LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "COHERE_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "GEMINI_API_KEY",
        "GLM_API_KEY",
        "GLM_BASE_URL",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MINIMAX_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ZHIPUAI_API_KEY",
        "ZHIPUAI_BASE_URL"
    ) | ForEach-Object { [void]$names.Add($_) }
    Get-ChildItem Env: | Where-Object {
        $_.Name -like "LLM_*" -or $names.Contains($_.Name)
    } | ForEach-Object { [void]$names.Add($_.Name) }
    return @($names | Sort-Object)
}

function Get-ClearedLlmEnvironment {
    $environment = @{}
    foreach ($name in Get-LlmSensitiveEnvironmentNames) {
        # A null process-scope value removes the variable for the child. Empty
        # strings are intentionally avoided because typed settings reject them.
        $environment[$name] = $null
    }
    return $environment
}

function Assert-LlmSingleLineValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value,
        [switch]$AllowEmpty
    )

    if (-not $AllowEmpty -and [string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name cannot be empty."
    }
    if ($Value -match '[\x00-\x1f\x7f]') {
        throw "$Name must be single-line text without control characters."
    }
}

function Test-LlmLoopbackHost {
    param([Parameter(Mandatory = $true)][string]$HostName)

    $normalized = $HostName.Trim('[', ']').ToLowerInvariant()
    if ($normalized -eq "localhost" -or $normalized -eq "127.0.0.1" -or $normalized -eq "::1") {
        return $true
    }
    $address = $null
    if ([Net.IPAddress]::TryParse($normalized, [ref]$address)) {
        return [Net.IPAddress]::IsLoopback($address)
    }
    return $false
}

function Normalize-LlmApiBase {
    param(
        [Parameter(Mandatory = $true)][string]$ApiBase,
        [switch]$AllowInsecureLoopback
    )

    Assert-LlmSingleLineValue -Name "API Base" -Value $ApiBase
    $normalized = $ApiBase.Trim().TrimEnd('/')
    if ($normalized -match '(?i)^https?://https?://' -or $normalized.Contains("\\")) {
        throw "API Base has an invalid or repeated protocol prefix."
    }
    if ($normalized.Contains("?") -or $normalized.Contains("#")) {
        throw "API Base cannot contain a query string or fragment."
    }
    if ($normalized -match '(?i)%0[0ad]') {
        throw "API Base cannot contain encoded control characters."
    }
    $uri = $null
    if (
        -not [Uri]::TryCreate($normalized, [UriKind]::Absolute, [ref]$uri) -or
        [string]::IsNullOrWhiteSpace($uri.Host) -or
        $uri.Scheme -notin @("http", "https")
    ) {
        throw "API Base must be an absolute HTTP(S) URL, for example https://provider.example/v1."
    }
    if (-not [string]::IsNullOrEmpty($uri.UserInfo)) {
        throw "API Base cannot contain embedded credentials."
    }
    if ($uri.Scheme -eq "http" -and (
        -not $AllowInsecureLoopback -or -not (Test-LlmLoopbackHost -HostName $uri.Host)
    )) {
        throw "Remote API Base URLs must use HTTPS. HTTP is allowed only for an explicitly enabled loopback endpoint."
    }
    return $normalized
}

function Read-ApiPrivateEnvironment {
    param([Parameter(Mandatory = $true)][string]$Path)

    $environment = @{}
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $environment
    }
    Assert-PrivateConfigurationFile -Path $Path
    $allowedNames = @(
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_API_MODE",
        "LLM_AUTH_SCHEME",
        "LLM_ANTHROPIC_VERSION",
        "LLM_TIMEOUT_SECONDS",
        "LLM_ALLOW_INSECURE_LOOPBACK",
        "LLM_VERIFICATION_STATUS",
        "LOG_LEVEL"
    )
    $launcherOwnedNames = @(
        "DATASET_UPLOAD_MAX_BYTES", "DATASET_ARCHIVE_MAX_BYTES",
        "DATASET_ARCHIVE_MAX_FILES", "DATASET_STORAGE_ROOT",
        "INSPECTION_CACHE_TTL_SECONDS", "INSPECTION_CACHE_MAX_BYTES",
        "INSPECTION_CACHE_MAX_PROJECT_BYTES", "INSPECTION_CACHE_MAX_ENTRY_BYTES",
        "RUNTIME_BUILD_ID", "LOCAL_DEMO_LOOPBACK_ONLY", "GFM_INFRASTRUCTURE_READY",
        "GFM_SERVICE_URL", "GFM_SESSION_TOKEN_FILE", "GFM_CORE_SERVING_CONTROL_FILE",
        "GFM_CORE_RUN_BINDING_ROOT", "GFM_RESEARCH_RUN_BINDING_ROOT",
        "GFM_GLOBAL_MODEL_RUN_BINDING_ROOT", "GFM_GLOBAL_MODEL_REVIEW_ROOT",
        "GFM_GOVERNANCE_ROOT", "GFM_GOVERNANCE_BUNDLE_MAX_BYTES",
        "GFM_GOVERNANCE_EXPANDED_MAX_BYTES", "GFM_CORE_SERVING_HIGH_WATER_ROOT",
        "GFM_TIMEOUT_SECONDS", "GFM_REQUEST_MAX_BYTES", "GRAPH_HANDOFF_TOKEN_TTL_SECONDS",
        "TRUSTED_ARRAY_MAX_BYTES", "ENABLE_TRUSTED_LOCAL_CONVERSION",
        "TRUSTED_DATA_ROOTS", "TRUSTED_CONVERTER_PYTHON",
        "TRUSTED_CONVERSION_TIMEOUT_SECONDS", "TRUSTED_CONVERSION_MAX_FILES",
        "TRUSTED_CONVERSION_MAX_SOURCE_BYTES", "TRUSTED_CONVERSION_MAX_OUTPUT_BYTES",
        "TRUSTED_CONVERSION_MEMORY_MB", "ALLOWED_ORIGINS"
    )

    foreach ($rawLine in [IO.File]::ReadAllLines($Path, [Text.UTF8Encoding]::new($false, $true))) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) { continue }
        $separator = $line.IndexOf("=")
        if ($separator -le 0) { throw "Invalid private configuration line in $Path." }
        $name = $line.Substring(0, $separator).Trim().ToUpperInvariant()
        if ($name -notmatch '^[A-Z_][A-Z0-9_]*$') {
            throw "Invalid private configuration name in $Path."
        }
        if ($launcherOwnedNames -contains $name) { continue }
        if ($allowedNames -notcontains $name) {
            throw "Unsupported private configuration name: $name"
        }
        if ($environment.ContainsKey($name)) {
            throw "Duplicate private configuration name: $name"
        }
        $value = $line.Substring($separator + 1).Trim()
        if ($value.Length -ge 2 -and (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        )) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        elseif (
            $value.StartsWith('"') -or $value.EndsWith('"') -or
            $value.StartsWith("'") -or $value.EndsWith("'")
        ) {
            throw "Unbalanced quotes for private configuration name: $name"
        }
        Assert-LlmSingleLineValue -Name $name -Value $value -AllowEmpty
        $environment[$name] = $value
    }

    if ($environment.ContainsKey("LLM_API_MODE") -and
        $environment.LLM_API_MODE -notin @("chat_completions", "responses", "anthropic_messages")) {
        throw "LLM_API_MODE must be chat_completions, responses, or anthropic_messages."
    }
    $mode = if ($environment.ContainsKey("LLM_API_MODE") -and $environment.LLM_API_MODE) {
        $environment.LLM_API_MODE
    } else { "chat_completions" }
    if (-not $environment.ContainsKey("LLM_AUTH_SCHEME") -or
        [string]::IsNullOrWhiteSpace($environment.LLM_AUTH_SCHEME)) {
        $environment["LLM_AUTH_SCHEME"] = $(if ($mode -eq "anthropic_messages") { "x-api-key" } else { "bearer" })
    }
    if ($environment.LLM_AUTH_SCHEME -notin @("bearer", "x-api-key")) {
        throw "LLM_AUTH_SCHEME must be bearer or x-api-key."
    }
    if (-not $environment.ContainsKey("LLM_ANTHROPIC_VERSION")) {
        $environment["LLM_ANTHROPIC_VERSION"] = $(if ($mode -eq "anthropic_messages") { "2023-06-01" } else { "" })
    }
    if ($mode -eq "anthropic_messages" -and
        $environment.LLM_ANTHROPIC_VERSION -notmatch '^\d{4}-\d{2}-\d{2}$') {
        throw "LLM_ANTHROPIC_VERSION must use YYYY-MM-DD."
    }
    if ($environment.ContainsKey("LLM_TIMEOUT_SECONDS")) {
        $timeout = 0
        if (-not [int]::TryParse($environment.LLM_TIMEOUT_SECONDS, [ref]$timeout) -or
            $timeout -lt 1 -or $timeout -gt 60) {
            throw "LLM_TIMEOUT_SECONDS must be between 1 and 60."
        }
    }
    if ($environment.ContainsKey("LLM_ALLOW_INSECURE_LOOPBACK") -and
        $environment.LLM_ALLOW_INSECURE_LOOPBACK -notin @("true", "false")) {
        throw "LLM_ALLOW_INSECURE_LOOPBACK must be true or false."
    }
    if ($environment.ContainsKey("LLM_VERIFICATION_STATUS") -and
        $environment.LLM_VERIFICATION_STATUS -notin @(
            "configured_unverified", "call_succeeded", "fallback"
        )) {
        throw "LLM_VERIFICATION_STATUS is invalid."
    }
    if ($environment.ContainsKey("LLM_API_BASE") -and
        -not [string]::IsNullOrWhiteSpace($environment.LLM_API_BASE)) {
        $allowLoopback = $environment.ContainsKey("LLM_ALLOW_INSECURE_LOOPBACK") -and
            $environment["LLM_ALLOW_INSECURE_LOOPBACK"] -eq "true"
        $environment["LLM_API_BASE"] = Normalize-LlmApiBase `
            -ApiBase $environment["LLM_API_BASE"] -AllowInsecureLoopback:$allowLoopback
    }
    return $environment
}

function Get-LlmConfigurationState {
    param([Parameter(Mandatory = $true)][hashtable]$Environment)

    $required = @("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
    $present = @($required | Where-Object {
        $Environment.ContainsKey($_) -and -not [string]::IsNullOrWhiteSpace([string]$Environment[$_])
    })
    $anyLlmValue = @($Environment.Keys | Where-Object {
        $_ -like "LLM_*" -and -not [string]::IsNullOrWhiteSpace([string]$Environment[$_])
    }).Count -gt 0
    if ($present.Count -eq 0 -and -not $anyLlmValue) { return "Missing" }
    if ($present.Count -ne $required.Count) { return "Partial" }
    return "Complete"
}

function Write-LlmPrivateConfiguration {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ApiBase,
        [Parameter(Mandatory = $true)][string]$ApiKey,
        [Parameter(Mandatory = $true)][string]$Model,
        [ValidateSet("responses", "chat_completions", "anthropic_messages")]
        [string]$ApiMode = "chat_completions",
        [ValidateSet("bearer", "x-api-key")][string]$AuthScheme,
        [string]$AnthropicVersion,
        [ValidateRange(1, 60)][int]$TimeoutSeconds = 15,
        [ValidateSet("configured_unverified", "call_succeeded", "fallback")]
        [string]$VerificationStatus = "configured_unverified",
        [switch]$AllowInsecureLoopback
    )

    Assert-LlmSingleLineValue -Name "API Key" -Value $ApiKey
    Assert-LlmSingleLineValue -Name "Model" -Value $Model
    if (-not $PSBoundParameters.ContainsKey("AuthScheme")) {
        $AuthScheme = $(if ($ApiMode -eq "anthropic_messages") { "x-api-key" } else { "bearer" })
    }
    if (-not $PSBoundParameters.ContainsKey("AnthropicVersion")) {
        $AnthropicVersion = $(if ($ApiMode -eq "anthropic_messages") { "2023-06-01" } else { "" })
    }
    if ($ApiMode -eq "anthropic_messages" -and $AnthropicVersion -notmatch '^\d{4}-\d{2}-\d{2}$') {
        throw "AnthropicVersion must use YYYY-MM-DD."
    }
    if ($ApiKey -ne $ApiKey.Trim() -or $Model -ne $Model.Trim()) {
        throw "API Key and model cannot have leading or trailing whitespace."
    }
    $normalizedApiBase = Normalize-LlmApiBase `
        -ApiBase $ApiBase -AllowInsecureLoopback:$AllowInsecureLoopback
    $directory = Split-Path -Parent ([IO.Path]::GetFullPath($Path))
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    Protect-UnifiedConfigDirectory -Path $directory
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        Assert-PrivateConfigurationFile -Path $Path
    }
    $lines = @(
        "LLM_API_BASE=$normalizedApiBase",
        "LLM_API_KEY=$ApiKey",
        "LLM_MODEL=$($Model.Trim())",
        "LLM_API_MODE=$ApiMode",
        "LLM_AUTH_SCHEME=$AuthScheme",
        "LLM_ANTHROPIC_VERSION=$AnthropicVersion",
        "LLM_TIMEOUT_SECONDS=$TimeoutSeconds",
        "LLM_ALLOW_INSECURE_LOOPBACK=$($AllowInsecureLoopback.IsPresent.ToString().ToLowerInvariant())",
        "LLM_VERIFICATION_STATUS=$VerificationStatus",
        "LOG_LEVEL=INFO"
    )
    $temporaryPath = Join-Path $directory ".$([IO.Path]::GetFileName($Path)).$PID.$([Guid]::NewGuid().ToString('N')).tmp"
    $backupPath = Join-Path $directory ".$([IO.Path]::GetFileName($Path)).$PID.$([Guid]::NewGuid().ToString('N')).bak"
    try {
        [IO.File]::WriteAllText(
            $temporaryPath,
            (($lines -join [Environment]::NewLine) + [Environment]::NewLine),
            [Text.UTF8Encoding]::new($false)
        )
        $verified = Read-ApiPrivateEnvironment -Path $temporaryPath
        if ((Get-LlmConfigurationState -Environment $verified) -ne "Complete") {
            throw "The staged LLM configuration did not pass validation."
        }
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            [IO.File]::Replace($temporaryPath, $Path, $backupPath, $true)
        }
        else {
            [IO.File]::Move($temporaryPath, $Path)
        }
        Assert-PrivateConfigurationFile -Path $Path
        $readBack = Read-ApiPrivateEnvironment -Path $Path
        if (
            (Get-LlmConfigurationState -Environment $readBack) -ne "Complete" -or
            $readBack["LLM_API_BASE"] -ne $normalizedApiBase -or
            $readBack["LLM_API_KEY"] -ne $ApiKey -or
            $readBack["LLM_MODEL"] -ne $Model.Trim() -or
            $readBack["LLM_API_MODE"] -ne $ApiMode -or
            $readBack["LLM_AUTH_SCHEME"] -ne $AuthScheme -or
            $readBack["LLM_ANTHROPIC_VERSION"] -ne $AnthropicVersion -or
            [int]$readBack["LLM_TIMEOUT_SECONDS"] -ne $TimeoutSeconds -or
            $readBack["LLM_VERIFICATION_STATUS"] -ne $VerificationStatus
        ) {
            throw "The saved LLM configuration failed read-back validation."
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $backupPath -ErrorAction SilentlyContinue
    }
}

function Convert-ApiPrivateConfiguration {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    $environment = Read-ApiPrivateEnvironment -Path $Path
    if ((Get-LlmConfigurationState -Environment $environment) -eq "Partial") {
        throw "The existing LLM configuration is partial. Run scripts\configure-llm.ps1."
    }
    if ((Get-LlmConfigurationState -Environment $environment) -eq "Complete") {
        Write-LlmPrivateConfiguration -Path $Path `
            -ApiBase $environment["LLM_API_BASE"] `
            -ApiKey $environment["LLM_API_KEY"] `
            -Model $environment["LLM_MODEL"] `
            -ApiMode $(if ($environment.ContainsKey("LLM_API_MODE")) { $environment["LLM_API_MODE"] } else { "chat_completions" }) `
            -AuthScheme $environment["LLM_AUTH_SCHEME"] `
            -AnthropicVersion $environment["LLM_ANTHROPIC_VERSION"] `
            -TimeoutSeconds $(if ($environment.ContainsKey("LLM_TIMEOUT_SECONDS")) { [int]$environment["LLM_TIMEOUT_SECONDS"] } else { 15 }) `
            -VerificationStatus $(if ($environment.ContainsKey("LLM_VERIFICATION_STATUS")) { $environment["LLM_VERIFICATION_STATUS"] } else { "configured_unverified" }) `
            -AllowInsecureLoopback:($environment.ContainsKey("LLM_ALLOW_INSECURE_LOOPBACK") -and $environment["LLM_ALLOW_INSECURE_LOOPBACK"] -eq "true")
    }
}

function Set-LlmVerificationStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]
        [ValidateSet("configured_unverified", "call_succeeded", "fallback")]
        [string]$Status
    )

    $environment = Read-ApiPrivateEnvironment -Path $Path
    if ((Get-LlmConfigurationState -Environment $environment) -ne "Complete") {
        throw "A complete LLM configuration is required to update verification status."
    }
    Write-LlmPrivateConfiguration -Path $Path `
        -ApiBase $environment["LLM_API_BASE"] `
        -ApiKey $environment["LLM_API_KEY"] `
        -Model $environment["LLM_MODEL"] `
        -ApiMode $(if ($environment.ContainsKey("LLM_API_MODE")) { $environment["LLM_API_MODE"] } else { "chat_completions" }) `
        -AuthScheme $environment["LLM_AUTH_SCHEME"] `
        -AnthropicVersion $environment["LLM_ANTHROPIC_VERSION"] `
        -TimeoutSeconds $(if ($environment.ContainsKey("LLM_TIMEOUT_SECONDS")) { [int]$environment["LLM_TIMEOUT_SECONDS"] } else { 15 }) `
        -VerificationStatus $Status `
        -AllowInsecureLoopback:($environment.ContainsKey("LLM_ALLOW_INSECURE_LOOPBACK") -and $environment["LLM_ALLOW_INSECURE_LOOPBACK"] -eq "true")
}

function Test-UnifiedInteractiveHost {
    if (-not [Environment]::UserInteractive) { return $false }
    try {
        if ([Console]::IsInputRedirected) { return $false }
    }
    catch { return $false }
    return $null -ne $Host.UI -and $null -ne $Host.UI.RawUI
}

function Resolve-LlmStartup {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [ValidateSet("Optional", "Required", "Disabled")][string]$LlmMode = "Optional",
        [switch]$NoLlmPrompt,
        [switch]$ReconfigureLlm
    )

    if ($LlmMode -eq "Disabled") { return $false }
    $interactive = -not $NoLlmPrompt -and (Test-UnifiedInteractiveHost)
    if ($ReconfigureLlm) {
        if (-not $interactive) {
            throw "-ReconfigureLlm requires an interactive terminal. Run scripts\configure-llm.ps1 directly for scripted configuration."
        }
        & (Join-Path $Layout.ProjectRoot "scripts\configure-llm.ps1")
    }

    $environment = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
    $state = Get-LlmConfigurationState -Environment $environment
    if ($state -eq "Complete") { return $true }
    if ($state -eq "Partial") {
        throw "LLM configuration is partial. Run scripts\configure-llm.ps1 to replace it safely."
    }
    if (-not $interactive) {
        if ($LlmMode -eq "Required") {
            throw "LLM configuration is required. Run scripts\configure-llm.ps1 first."
        }
        Write-Host "LLM is not configured; continuing in deterministic offline mode."
        return $false
    }

    $choice = (Read-Host "LLM is not configured. [C]onfigure now / continue [O]ffline / [Q]uit (default C)").Trim().ToUpperInvariant()
    if ([string]::IsNullOrEmpty($choice)) { $choice = "C" }
    switch ($choice) {
        "C" {
            & (Join-Path $Layout.ProjectRoot "scripts\configure-llm.ps1")
            $configured = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
            if ((Get-LlmConfigurationState -Environment $configured) -ne "Complete") {
                throw "LLM configuration did not complete."
            }
            return $true
        }
        "O" {
            if ($LlmMode -eq "Required") {
                throw "LLM mode is Required; offline continuation is not allowed."
            }
            return $false
        }
        "Q" { throw "Startup was cancelled by the user." }
        default { throw "Unknown selection. Choose C, O, or Q." }
    }
}
