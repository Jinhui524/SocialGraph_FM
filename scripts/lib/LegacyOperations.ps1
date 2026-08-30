Set-StrictMode -Version Latest

# Transitional implementations retained for command/import compatibility.
# Focused public startup behavior is provided by the modules loaded below.

function Get-GovernanceVerificationLayout {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]
        [ValidatePattern("^[0-9a-f]{32}$")]
        [string]$RunId
    )

    $values = [ordered]@{}
    foreach ($property in $Layout.PSObject.Properties) {
        $values[$property.Name] = $property.Value
    }
    $runtime = Join-Path $Layout.GovernanceStateRoot "verification-runs\$RunId"
    $values["GovernanceVerification"] = $true
    $values["GovernanceRunId"] = $RunId
    $values["VarRoot"] = $runtime
    $values["GfmHome"] = Join-Path $runtime "gfm"
    $values["CacheRoot"] = Join-Path $runtime "gfm\cache"
    $values["ConfigRoot"] = Join-Path $runtime "config"
    $values["TempRoot"] = Join-Path $runtime "tmp"
    $values["PidRoot"] = Join-Path $runtime "deploy\pids"
    $values["LogRoot"] = Join-Path $runtime "deploy\logs"
    $values["ServingRoot"] = Join-Path $runtime "gfm\serving-runtime"
    $values["ServingControl"] = Join-Path $runtime "gfm\serving-runtime\core-serving-control.json"
    $values["ServingToken"] = Join-Path $runtime "gfm\serving-runtime\session.token"
    $values["ServingArtifacts"] = Join-Path $runtime "gfm\serving-graphs"
    $values["PublishedServingArtifacts"] = $Layout.ServingArtifacts
    $values["ApiData"] = Join-Path $runtime "api\dataset-store"
    $values["ApiBindings"] = Join-Path $runtime "api\gfm-run-bindings"
    $values["ApiResearchBindings"] = Join-Path $runtime "api\gfm-research-run-bindings"
    $values["GlobalModelBindings"] = Join-Path $runtime "api\gfm-global-model-run-bindings"
    $values["GlobalModelReviews"] = Join-Path $runtime "api\gfm-global-model-reviews"
    $values["ApiHighWater"] = Join-Path $runtime "api\serving-control"
    $values["ResearchRoot"] = Join-Path $runtime "gfm\research"
    $values["GlobalModelSourceRoot"] = Join-Path $runtime "gfm\socialgraph-global"
    $values["GovernanceRoot"] = Join-Path $runtime "gfm\governance"
    return [pscustomobject]$values
}

function Get-GovernanceVerificationPorts {
    param([Parameter(Mandatory = $true)]$Ports)

    $excluded = [Collections.Generic.HashSet[int]]::new()
    foreach ($port in @($Ports.Gfm, $Ports.Api, $Ports.GovernanceWeb)) {
        [void]$excluded.Add([int]$port)
    }
    $allocate = {
        for ($attempt = 0; $attempt -lt 32; $attempt++) {
            $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
            try {
                $listener.Start()
                $candidate = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
            }
            finally {
                $listener.Stop()
            }
            if ($excluded.Add([int]$candidate) -and -not (Test-LoopbackPort -Port $candidate)) {
                return [int]$candidate
            }
        }
        throw "Unable to allocate an isolated loopback port for governance verification."
    }
    [pscustomobject]@{
        GovernanceWeb = [int]$Ports.GovernanceWeb
        Api = & $allocate
        Gfm = & $allocate
    }
}

function Sync-GovernanceVerificationKnowledge {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$GovernanceLayout
    )

    $expectedNames = @("knowledge.sqlite3", "manifest.json")
    $source = Join-Path $Layout.GovernanceRoot "knowledge"
    $destination = Join-Path $GovernanceLayout.GovernanceRoot "knowledge"
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "The verified SocialGraph-FM Governance knowledge index is unavailable: $source"
    }
    $sourceItem = Get-Item -LiteralPath $source -Force
    if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The verified SocialGraph-FM Governance knowledge index cannot be a reparse point: $source"
    }
    $sourceChildren = @(Get-ChildItem -LiteralPath $source -Force)
    if (
        $sourceChildren.Count -ne $expectedNames.Count -or
        @($sourceChildren | Where-Object {
            $_.PSIsContainer -or
            ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $expectedNames -notcontains $_.Name
        }).Count -ne 0
    ) {
        throw "The verified SocialGraph-FM Governance knowledge index inventory is invalid."
    }

    $sha256 = {
        param([string]$Path)
        $stream = [IO.File]::OpenRead($Path)
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace("-", "")
        }
        finally {
            $algorithm.Dispose()
            $stream.Dispose()
        }
    }
    try {
        $manifest = Get-Content -LiteralPath (Join-Path $source "manifest.json") -Raw |
            ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "The verified SocialGraph-FM Governance knowledge manifest is invalid."
    }
    $databasePath = Join-Path $source "knowledge.sqlite3"
    $databaseItem = Get-Item -LiteralPath $databasePath -Force
    if (
        $manifest.schemaVersion -ne "socialgraph-fm.governance-knowledge-index/1.0" -or
        [int64]$manifest.database.bytes -ne [int64]$databaseItem.Length -or
        -not [string]::Equals(
            [string]$manifest.database.sha256,
            (& $sha256 $databasePath),
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "The verified SocialGraph-FM Governance knowledge database does not match its manifest."
    }
    $databaseHeader = ([IO.File]::ReadAllBytes($databasePath))[0..15]
    if ([Text.Encoding]::ASCII.GetString($databaseHeader) -ne "SQLite format 3`0") {
        throw "The verified SocialGraph-FM Governance knowledge database is not SQLite."
    }

    New-Item -ItemType Directory -Force -Path $GovernanceLayout.GovernanceRoot | Out-Null
    if (Test-Path -LiteralPath $destination) {
        $destinationItem = Get-Item -LiteralPath $destination -Force
        if (
            -not $destinationItem.PSIsContainer -or
            ($destinationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            throw "The governance knowledge destination is unsafe: $destination"
        }
        $unexpected = @(Get-ChildItem -LiteralPath $destination -Force | Where-Object {
            $_.PSIsContainer -or
            ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $expectedNames -notcontains $_.Name
        })
        if ($unexpected.Count -ne 0) {
            throw "The governance knowledge destination contains an unexpected entry."
        }
    }
    else {
        New-Item -ItemType Directory -Path $destination | Out-Null
    }

    foreach ($name in $expectedNames) {
        $sourcePath = Join-Path $source $name
        $destinationPath = Join-Path $destination $name
        $temporaryPath = Join-Path $destination ".$name.$PID.tmp"
        try {
            Copy-Item -LiteralPath $sourcePath -Destination $temporaryPath
            if ((& $sha256 $temporaryPath) -ne (& $sha256 $sourcePath)) {
                throw "The governance knowledge copy failed its hash check: $name"
            }
            Move-Item -LiteralPath $temporaryPath -Destination $destinationPath -Force
        }
        finally {
            Remove-Item -LiteralPath $temporaryPath -ErrorAction SilentlyContinue
        }
    }
}

function Assert-ApiIsTorchFree {
    param([Parameter(Mandatory = $true)][string]$PythonExecutable)

    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        throw "SocialGraph-FM API Python environment is missing: $PythonExecutable"
    }
    $probe = @'
import importlib.util
import sys

forbidden = [name for name in ("torch", "torch_geometric") if importlib.util.find_spec(name) is not None]
if forbidden:
    print("SocialGraph-FM API environment contains forbidden ML packages: " + ", ".join(forbidden), file=sys.stderr)
    raise SystemExit(1)
'@
    $probe | & $PythonExecutable -
    if ($LASTEXITCODE -ne 0) {
        throw "SocialGraph-FM API environment must not contain torch or torch_geometric."
    }
}

function Initialize-UnifiedDirectories {
    param([Parameter(Mandatory = $true)]$Layout)

    @(
        $Layout.LogRoot,
        $Layout.PidRoot,
        $Layout.TempRoot,
        $Layout.ManagedPythonRoot,
        (Join-Path $Layout.CacheRoot "pip"),
        (Join-Path $Layout.CacheRoot "uv"),
        (Join-Path $Layout.CacheRoot "hf"),
        (Join-Path $Layout.CacheRoot "torch"),
        (Join-Path $Layout.CacheRoot "torchinductor"),
        (Join-Path $Layout.CacheRoot "wandb"),
        $Layout.ServingRoot,
        $Layout.ServingArtifacts,
        $Layout.ApiData,
        $Layout.ApiBindings,
        $Layout.ApiResearchBindings,
        $Layout.GlobalModelBindings,
        $Layout.GlobalModelReviews,
        $Layout.ApiHighWater,
        $Layout.ResearchRoot,
        $Layout.GlobalModelSourceRoot,
        $Layout.GovernanceRoot,
        (Join-Path $Layout.GovernanceRoot "incoming"),
        (Join-Path $Layout.GovernanceRoot "artifacts"),
        (Join-Path $Layout.GovernanceRoot "runs"),
        (Join-Path $Layout.GovernanceRoot "samples"),
        $Layout.ConfigRoot,
        (Join-Path $Layout.VarRoot "research\incoming"),
        (Join-Path $Layout.VarRoot "backups\git")
    ) | ForEach-Object {
        New-Item -ItemType Directory -Force -Path $_ | Out-Null
    }
    Protect-UnifiedConfigDirectory -Path $Layout.ConfigRoot
}

function Initialize-UnifiedServingContracts {
    param([Parameter(Mandatory = $true)]$Layout)

    $contractRoot = Join-Path $Layout.GfmPackage "contracts"
    $contractNames = @(
        "core-serving-control.json",
        "core-serving-registry.json",
        "core-serving-graph-catalog.json"
    )
    foreach ($name in $contractNames) {
        $source = Join-Path $contractRoot $name
        $destination = Join-Path $Layout.ServingRoot $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "SocialGraph-FM Core serving contract is missing: $source"
        }
        if (-not (Test-Path -LiteralPath $destination)) {
            Copy-Item -LiteralPath $source -Destination $destination
        }
    }
}

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-WithProcessEnvironment {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Environment,
        [Parameter(Mandatory = $true)][scriptblock]$ScriptBlock
    )

    $previous = @{}
    try {
        foreach ($name in $Environment.Keys) {
            $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
            $value = $Environment[$name]
            [Environment]::SetEnvironmentVariable(
                $name,
                $(if ($null -eq $value) { $null } else { [string]$value }),
                "Process"
            )
        }
        & $ScriptBlock
    }
    finally {
        foreach ($name in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
}

function Get-ManagedProcessInfo {
    param([Parameter(Mandatory = $true)][int]$ProcessId)

    Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function ConvertTo-NormalizedProcessCreationTime {
    param([Parameter(Mandatory = $true)]$Value)

    try {
        if ($Value -is [DateTimeOffset]) {
            $timestamp = [DateTimeOffset]$Value
        }
        elseif ($Value -is [DateTime]) {
            $timestamp = [DateTimeOffset]([DateTime]$Value)
        }
        else {
            $timestamp = [DateTimeOffset]::MinValue
            if (-not [DateTimeOffset]::TryParse(
                [string]$Value,
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal,
                [ref]$timestamp
            )) {
                return $null
            }
        }
        return $timestamp.ToUniversalTime().ToString(
            "yyyy-MM-ddTHH:mm:ss.fff'Z'",
            [Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        return $null
    }
}

function Test-RecordedProcessIdentity {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)]$ProcessInfo
    )

    if ([string]::IsNullOrWhiteSpace([string]$ProcessInfo.ExecutablePath)) {
        return $false
    }
    $recordCreationProperty = $Record.PSObject.Properties["creationTimeUtc"]
    $processCreationProperty = $ProcessInfo.PSObject.Properties["CreationDate"]
    if ($null -eq $recordCreationProperty -or $null -eq $processCreationProperty) {
        return $false
    }
    $expectedCreation = ConvertTo-NormalizedProcessCreationTime -Value $recordCreationProperty.Value
    $actualCreation = ConvertTo-NormalizedProcessCreationTime -Value $processCreationProperty.Value
    if ([string]::IsNullOrWhiteSpace($expectedCreation) -or
        -not [string]::Equals($expectedCreation, $actualCreation, [StringComparison]::Ordinal)) {
        return $false
    }
    try {
        $actualExecutable = [System.IO.Path]::GetFullPath([string]$ProcessInfo.ExecutablePath)
        $expectedExecutable = [System.IO.Path]::GetFullPath([string]$Record.executablePath)
    }
    catch {
        return $false
    }
    if (-not [string]::Equals($actualExecutable, $expectedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }

    $commandLine = [string]$ProcessInfo.CommandLine
    if ([string]::IsNullOrWhiteSpace($commandLine)) {
        return $false
    }
    foreach ($token in @($Record.requiredCommandLineTokens)) {
        if ($commandLine.IndexOf([string]$token, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            return $false
        }
    }
    return $true
}

function Test-LoopbackPort {
    param([Parameter(Mandatory = $true)][int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $result = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne(250)) {
            return $false
        }
        $client.EndConnect($result)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Read-ManagedPidRecord {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $path = Join-Path $Layout.PidRoot "$Name.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Invalid PID record; inspect it before retrying: $path"
    }
}

function Get-ManagedServiceSnapshot {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Service
    )

    $record = Read-ManagedPidRecord -Layout $Layout -Name $Service.Name
    if ($null -eq $record) {
        if (Test-LoopbackPort -Port $Service.Port) {
            throw "Cannot snapshot $($Service.Name): port $($Service.Port) is owned by an unmanaged process."
        }
        return [pscustomobject]@{
            Name = $Service.Name
            Port = $Service.Port
            WasRunning = $false
            ProcessId = $null
            CreationTimeUtc = $null
            ExecutablePath = [System.IO.Path]::GetFullPath($Service.Executable)
            IdentityTokens = @($Service.IdentityTokens)
            PidRecordSha256 = $null
        }
    }
    $processInfo = Get-ManagedProcessInfo -ProcessId ([int]$record.pid)
    if ($null -eq $processInfo -or -not (Test-RecordedProcessIdentity -Record $record -ProcessInfo $processInfo)) {
        throw "Cannot snapshot $($Service.Name): its managed process identity is stale or mismatched."
    }
    if ([int]$record.port -ne [int]$Service.Port -or -not (Test-LoopbackPort -Port $Service.Port)) {
        throw "Cannot snapshot $($Service.Name): its recorded port is inconsistent."
    }
    return [pscustomobject]@{
        Name = $Service.Name
        Port = $Service.Port
        WasRunning = $true
        ProcessId = [int]$record.pid
        CreationTimeUtc = ConvertTo-NormalizedProcessCreationTime -Value $record.creationTimeUtc
        ExecutablePath = [System.IO.Path]::GetFullPath([string]$record.executablePath)
        IdentityTokens = @($record.requiredCommandLineTokens)
        PidRecordSha256 = Get-UnifiedFileSha256 -Path (Join-Path $Layout.PidRoot "$($Service.Name).json")
    }
}

function Assert-ManagedServiceSnapshotUnchanged {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Service,
        [Parameter(Mandatory = $true)]$Snapshot
    )

    $current = Get-ManagedServiceSnapshot -Layout $Layout -Service $Service
    foreach ($name in @(
        "Name", "Port", "WasRunning", "ProcessId", "CreationTimeUtc", "ExecutablePath",
        "PidRecordSha256"
    )) {
        if ([string]$current.$name -ne [string]$Snapshot.$name) {
            throw "Ordinary service $($Service.Name) changed during isolated verification ($name)."
        }
    }
    if ((@($current.IdentityTokens) -join "`0") -ne (@($Snapshot.IdentityTokens) -join "`0")) {
        throw "Ordinary service $($Service.Name) identity changed during isolated verification."
    }
}

function Get-UnifiedFileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function Get-UnifiedConfinedFileSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$InventoryName,
        [string]$ExpectedSha256
    )

    $rootPath = [IO.Path]::GetFullPath($Root)
    if (-not (Test-Path -LiteralPath $rootPath -PathType Container)) {
        throw "Published snapshot root is unavailable: $rootPath"
    }
    $rootItem = Get-Item -LiteralPath $rootPath -Force
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Published snapshot root cannot be a reparse point: $rootPath"
    }
    $normalized = $RelativePath.Replace("\", "/")
    $parts = @($normalized.Split("/", [StringSplitOptions]::RemoveEmptyEntries))
    if (
        $parts.Count -eq 0 -or
        [IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath.Contains(":") -or
        @($parts | Where-Object { $_ -in @(".", "..") }).Count -ne 0
    ) {
        throw "Published snapshot relative path is unsafe: $RelativePath"
    }
    $candidate = $rootPath
    foreach ($part in $parts) {
        $candidate = Join-Path $candidate $part
        if (-not (Test-Path -LiteralPath $candidate)) {
            throw "Published snapshot file is unavailable: $normalized"
        }
        $item = Get-Item -LiteralPath $candidate -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Published snapshot path contains a reparse point: $normalized"
        }
    }
    $file = Get-Item -LiteralPath $candidate -Force
    if (-not $file.PSIsContainer -and $file.Length -ge 0) {
        $sha256 = Get-UnifiedFileSha256 -Path $file.FullName
        if (
            -not [string]::IsNullOrWhiteSpace($ExpectedSha256) -and
            -not [string]::Equals($sha256, $ExpectedSha256, [StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "Published snapshot digest is invalid: $normalized"
        }
        return [pscustomobject]@{
            Root = $InventoryName
            RelativePath = $normalized
            Kind = "file"
            Bytes = [int64]$file.Length
            Sha256 = $sha256
        }
    }
    throw "Published snapshot path is not a regular file: $normalized"
}

function Get-ManagedServingStateSnapshot {
    param([Parameter(Mandatory = $true)]$Layout)

    $publishedRootProperty = $Layout.PSObject.Properties["PublishedServingRoot"]
    $publishedRoot = if ($null -ne $publishedRootProperty) {
        [string]$publishedRootProperty.Value
    }
    else {
        [string]$Layout.ServingRoot
    }
    $roots = @(
        [pscustomobject]@{ Name = "published-serving"; Path = $publishedRoot },
        [pscustomobject]@{ Name = "api-high-water"; Path = [string]$Layout.ApiHighWater }
    )
    $publishedArtifactsProperty = $Layout.PSObject.Properties["PublishedServingArtifacts"]
    if ($null -ne $publishedArtifactsProperty) {
        $roots += [pscustomobject]@{
            Name = "published-serving-artifacts"
            Path = [string]$publishedArtifactsProperty.Value
        }
    }
    $entries = [Collections.Generic.List[object]]::new()
    foreach ($root in $roots) {
        $rootPath = [IO.Path]::GetFullPath($root.Path)
        if (-not (Test-Path -LiteralPath $rootPath)) {
            $entries.Add([pscustomobject]@{
                Root = $root.Name
                RelativePath = ""
                Kind = "missing"
                Bytes = $null
                Sha256 = $null
            })
            continue
        }
        $rootItem = Get-Item -LiteralPath $rootPath -Force
        if (-not $rootItem.PSIsContainer -or ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Ordinary serving snapshot root is unsafe: $rootPath"
        }
        $entries.Add([pscustomobject]@{
            Root = $root.Name
            RelativePath = ""
            Kind = "directory"
            Bytes = $null
            Sha256 = $null
        })
        $rootPrefix = $rootPath.TrimEnd([char[]]@("\", "/")) + [IO.Path]::DirectorySeparatorChar
        foreach ($item in Get-ChildItem -LiteralPath $rootPath -Force -Recurse | Sort-Object FullName) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Ordinary serving snapshot contains a reparse point: $($item.FullName)"
            }
            if (-not $item.FullName.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Ordinary serving snapshot escaped its root: $($item.FullName)"
            }
            $relative = $item.FullName.Substring($rootPrefix.Length).Replace("\", "/")
            if ($item.PSIsContainer) {
                $entries.Add([pscustomobject]@{
                    Root = $root.Name
                    RelativePath = $relative
                    Kind = "directory"
                    Bytes = $null
                    Sha256 = $null
                })
                continue
            }
            $entries.Add([pscustomobject]@{
                Root = $root.Name
                RelativePath = $relative
                Kind = "file"
                Bytes = [int64]$item.Length
                Sha256 = Get-UnifiedFileSha256 -Path $item.FullName
            })
        }
    }
    $modelRootProperty = $Layout.PSObject.Properties["GlobalModelRoot"]
    if ($null -ne $modelRootProperty) {
        $modelRoot = [string]$modelRootProperty.Value
        $registryEntry = Get-UnifiedConfinedFileSnapshot `
            -Root $modelRoot `
            -RelativePath "registry/socialgraph-global.json" `
            -InventoryName "published-global-model"
        $entries.Add($registryEntry)
        try {
            $registry = Get-Content `
                -LiteralPath (Join-Path $modelRoot "registry\socialgraph-global.json") `
                -Raw | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            throw "Published Global model registry is invalid."
        }
        foreach ($descriptor in @(
            @("checkpointPath", "checkpointSha256"),
            @("modelCardPath", "modelCardSha256"),
            @("exportPath", "exportSha256")
        )) {
            $relativePath = [string]$registry.($descriptor[0])
            $expectedHash = [string]$registry.($descriptor[1])
            if (
                [string]::IsNullOrWhiteSpace($relativePath) -or
                $expectedHash -notmatch "^[0-9a-f]{64}$"
            ) {
                throw "Published Global model registry binding is invalid: $($descriptor[0])"
            }
            $entries.Add((Get-UnifiedConfinedFileSnapshot `
                -Root $modelRoot `
                -RelativePath $relativePath `
                -InventoryName "published-global-model" `
                -ExpectedSha256 $expectedHash))
        }
    }
    return [pscustomobject]@{
        SchemaVersion = "socialgraph-fm.ordinary-serving-snapshot/1.0"
        Entries = @($entries)
    }
}

function Assert-ManagedServingStateSnapshotUnchanged {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Snapshot
    )

    $current = Get-ManagedServingStateSnapshot -Layout $Layout
    $before = $Snapshot | ConvertTo-Json -Depth 8 -Compress
    $after = $current | ConvertTo-Json -Depth 8 -Compress
    if ($before -cne $after) {
        throw "Ordinary serving files changed during isolated verification."
    }
}

function Restore-ManagedServiceSnapshot {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Service,
        [Parameter(Mandatory = $true)]$Snapshot
    )

    $expectedExecutable = [System.IO.Path]::GetFullPath($Service.Executable)
    if (
        $Snapshot.Name -ne $Service.Name -or
        [int]$Snapshot.Port -ne [int]$Service.Port -or
        -not [string]::Equals(
            [string]$Snapshot.ExecutablePath,
            $expectedExecutable,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        (@($Snapshot.IdentityTokens) -join "`0") -ne (@($Service.IdentityTokens) -join "`0")
    ) {
        throw "Cannot restore $($Service.Name): the service definition changed after snapshot."
    }
    $record = Read-ManagedPidRecord -Layout $Layout -Name $Service.Name
    if (-not [bool]$Snapshot.WasRunning) {
        if ($null -ne $record -or (Test-LoopbackPort -Port $Service.Port)) {
            throw "Cannot restore $($Service.Name) to stopped: a process now owns its managed identity or port."
        }
        return
    }
    if ($null -ne $record) {
        $processInfo = Get-ManagedProcessInfo -ProcessId ([int]$record.pid)
        if ($null -ne $processInfo -and (Test-RecordedProcessIdentity -Record $record -ProcessInfo $processInfo)) {
            if (
                [int]$record.port -ne [int]$Service.Port -or
                -not (Test-LoopbackPort -Port $Service.Port)
            ) {
                throw "Cannot restore $($Service.Name): its recorded port or listener is inconsistent."
            }
            return
        }
        throw "Cannot restore $($Service.Name): a mismatched managed PID record appeared."
    }
    if (Test-LoopbackPort -Port $Service.Port) {
        throw "Cannot restore $($Service.Name): its port is owned by an unmanaged process."
    }
    [void](Start-ManagedService -Layout $Layout -Service $Service)
}

function Move-StaleManagedPidRecord {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$RecordPath,
        [Parameter(Mandatory = $true)][int]$ProcessId
    )

    if (-not (Test-Path -LiteralPath $RecordPath -PathType Leaf)) {
        return
    }
    $staleRoot = Join-Path $Layout.PidRoot "stale"
    New-Item -ItemType Directory -Force -Path $staleRoot | Out-Null
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $suffix = [guid]::NewGuid().ToString("N")
    $destination = Join-Path $staleRoot "$Name.$timestamp.$suffix.json"
    Move-Item -LiteralPath $RecordPath -Destination $destination
    Write-Warning "$Name had a stale PID record for reused PID $ProcessId; the record was archived at $destination. The unrelated process was not stopped."
}

function Start-ManagedService {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Service
    )

    $recordPath = Join-Path $Layout.PidRoot "$($Service.Name).json"
    $existing = Read-ManagedPidRecord -Layout $Layout -Name $Service.Name
    if ($null -ne $existing) {
        $processInfo = Get-ManagedProcessInfo -ProcessId ([int]$existing.pid)
        if ($null -ne $processInfo) {
            if (Test-RecordedProcessIdentity -Record $existing -ProcessInfo $processInfo) {
                Write-Host "$($Service.Name) is already running (PID $($existing.pid))."
                return $false
            }
            if (Test-LoopbackPort -Port $Service.Port) {
                throw "PID record for $($Service.Name) belongs to a different process and port $($Service.Port) is in use; the unrelated process was not stopped: $recordPath"
            }
            Move-StaleManagedPidRecord -Layout $Layout -Name $Service.Name `
                -RecordPath $recordPath -ProcessId ([int]$existing.pid)
        }
        else {
            Remove-Item -LiteralPath $recordPath
        }
    }

    if (Test-LoopbackPort -Port $Service.Port) {
        throw "Port $($Service.Port) for $($Service.Name) is already in use by an unmanaged process."
    }
    if (-not (Test-Path -LiteralPath $Service.Executable -PathType Leaf)) {
        throw "Executable for $($Service.Name) is missing: $($Service.Executable)"
    }

    $stdoutPath = Join-Path $Layout.LogRoot "$($Service.Name).out.log"
    $stderrPath = Join-Path $Layout.LogRoot "$($Service.Name).err.log"
    $argumentLine = (@($Service.Arguments) | ForEach-Object { ConvertTo-NativeArgument -Value ([string]$_) }) -join " "
    $process = Invoke-WithProcessEnvironment -Environment $Service.Environment -ScriptBlock {
        Start-Process -FilePath $Service.Executable -ArgumentList $argumentLine `
            -WorkingDirectory $Service.WorkingDirectory -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    }

    $creationTimeUtc = ConvertTo-NormalizedProcessCreationTime -Value $process.StartTime
    if ([string]::IsNullOrWhiteSpace($creationTimeUtc)) {
        try { $process.Kill() } catch { }
        throw "Could not bind the creation time for $($Service.Name)."
    }
    $record = [ordered]@{
        schemaVersion = "socialgraph-fm.managed-process/1.1"
        service = $Service.Name
        pid = $process.Id
        executablePath = [System.IO.Path]::GetFullPath($Service.Executable)
        requiredCommandLineTokens = @($Service.IdentityTokens)
        creationTimeUtc = $creationTimeUtc
        port = $Service.Port
        startedAtUtc = [DateTime]::UtcNow.ToString("o")
    }
    $temporaryPath = "$recordPath.tmp"
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $recordPath -Force

    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-LoopbackPort -Port $Service.Port) {
            Write-Host "$($Service.Name) started on port $($Service.Port) (PID $($process.Id))."
            return $true
        }
        if ($process.HasExited) {
            Remove-Item -LiteralPath $recordPath -ErrorAction SilentlyContinue
            throw "$($Service.Name) exited during startup. Inspect $stderrPath"
        }
        Start-Sleep -Milliseconds 250
    }
    Stop-ManagedService -Layout $Layout -Name $Service.Name
    throw "$($Service.Name) did not listen on port $($Service.Port) within 60 seconds. Inspect $stderrPath"
}

function Stop-ManagedService {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $recordPath = Join-Path $Layout.PidRoot "$Name.json"
    $record = Read-ManagedPidRecord -Layout $Layout -Name $Name
    if ($null -eq $record) {
        Write-Host "$Name is not recorded as running."
        return
    }

    $processInfo = Get-ManagedProcessInfo -ProcessId ([int]$record.pid)
    if ($null -eq $processInfo) {
        Remove-Item -LiteralPath $recordPath
        Write-Host "$Name had a stale PID record; the record was removed."
        return
    }
    if (-not (Test-RecordedProcessIdentity -Record $record -ProcessInfo $processInfo)) {
        $recordPort = 0
        $recordPortProperty = $record.PSObject.Properties["port"]
        if ($null -eq $recordPortProperty -or
            -not [int]::TryParse([string]$recordPortProperty.Value, [ref]$recordPort) -or
            $recordPort -lt 1 -or $recordPort -gt 65535) {
            Write-Warning "Refusing to stop PID $($record.pid) or archive its PID record: process identity does not match $Name and the recorded port is invalid."
            return
        }
        if (Test-LoopbackPort -Port $recordPort) {
            Write-Warning "Refusing to stop PID $($record.pid) or archive its PID record: process identity does not match $Name and port $recordPort is still in use."
            return
        }
        Move-StaleManagedPidRecord -Layout $Layout -Name $Name `
            -RecordPath $recordPath -ProcessId ([int]$record.pid)
        return
    }

    $rootProcessId = [int]$record.pid
    $processIds = [System.Collections.Generic.List[int]]::new()
    if ($env:OS -eq "Windows_NT") {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop |
            Select-Object ProcessId, ParentProcessId)
        $frontier = @($rootProcessId)
        while ($frontier.Count -gt 0) {
            $next = [System.Collections.Generic.List[int]]::new()
            foreach ($parentId in $frontier) {
                foreach ($child in @($processes | Where-Object ParentProcessId -eq $parentId)) {
                    $childId = [int]$child.ProcessId
                    if (-not $processIds.Contains($childId)) {
                        $processIds.Add($childId)
                        $next.Add($childId)
                    }
                }
            }
            $frontier = @($next)
        }
    }
    $processIds.Reverse()
    $processIds.Add($rootProcessId)
    foreach ($processId in $processIds) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    $recordPort = [int]$record.port
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($null -eq (Get-ManagedProcessInfo -ProcessId $rootProcessId) -and
            -not (Test-LoopbackPort -Port $recordPort)) {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    if ($null -ne (Get-ManagedProcessInfo -ProcessId $rootProcessId) -or
        (Test-LoopbackPort -Port $recordPort)) {
        Write-Warning "$Name or its recorded port is still active; retaining its PID record."
        return
    }
    Remove-Item -LiteralPath $recordPath
    Write-Host "$Name stopped."
}

function Assert-UnifiedRuntimeReady {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode
    )

    $required = @(
        $Layout.ApiPython,
        $Layout.GfmPython,
        $Layout.ServingControl,
        (Join-Path $Layout.GovernanceWeb "node_modules\vite\bin\vite.js")
    )
    if ($Mode -eq "Production") {
        $required += Join-Path $Layout.GovernanceWeb "dist\client\index.html"
    }
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) {
        throw "Unified runtime is incomplete. Run scripts\bootstrap-all.ps1 first. Missing:`n$($missing -join "`n")"
    }
    Assert-ApiIsTorchFree -PythonExecutable $Layout.ApiPython
}

function Get-UnifiedServices {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Ports,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode
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
        UV_PYTHON_INSTALL_DIR = $Layout.ManagedPythonRoot
        HF_HOME = Join-Path $Layout.CacheRoot "hf"
        TORCH_HOME = Join-Path $Layout.CacheRoot "torch"
        TORCHINDUCTOR_CACHE_DIR = Join-Path $Layout.CacheRoot "torchinductor"
        WANDB_DIR = Join-Path $Layout.CacheRoot "wandb"
    }

    $gfmEnvironment = @{} + $common
    $gfmEnvironment.PYTHONPATH = Join-Path $Layout.GfmPackage "src"
    $gfmEnvironment.PYTHONNOUSERSITE = "1"
    $publishedControlProperty = $Layout.PSObject.Properties["PublishedServingControl"]
    $publishedServingControl = if ($null -ne $publishedControlProperty) {
        [string]$publishedControlProperty.Value
    }
    else {
        [string]$Layout.ServingControl
    }

    $apiEnvironment = Read-ApiPrivateEnvironment -Path $Layout.ApiConfig
    foreach ($name in $common.Keys) {
        $apiEnvironment[$name] = $common[$name]
    }
    $apiEnvironment.GFM_INFRASTRUCTURE_READY = "false"
    $apiEnvironment.SOCIALGRAPH_CORE_API_PORT = [string]$Ports.Api
    $apiEnvironment.GFM_SERVICE_URL = "http://127.0.0.1:$($Ports.Gfm)"
    $apiEnvironment.GFM_SESSION_TOKEN_FILE = $Layout.ServingToken
    $apiEnvironment.GFM_CORE_SERVING_CONTROL_FILE = $publishedServingControl
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
        $apiEnvironment.LLM_API_KEY = ""
    }

    $governanceWebEnvironment = @{} + $common
    $governanceWebEnvironment.VITE_SOCIALGRAPH_API_BASE_URL = "http://127.0.0.1:$($Ports.Api)"

    @(
        [pscustomobject]@{
            Name = "gfm"
            Port = $Ports.Gfm
            Executable = $Layout.GfmPython
            WorkingDirectory = $Layout.GfmPackage
            Arguments = @(
                "-m", "socialgraph_gfm.core.inference_cli",
                "--runtime-root", $Layout.ServingRoot,
                "--serving-control", $publishedServingControl,
                "--published-serving-root", $Layout.PublishedServingRoot,
                "--published-artifact-root", $Layout.PublishedServingArtifacts,
                "--artifact-root", $Layout.ServingArtifacts,
                "--research-root", $Layout.ResearchRoot,
                "--global-model-root", $Layout.GlobalModelRoot,
                "--governance-root", $Layout.GovernanceRoot,
                "--global-model-device", "auto",
                "--dataset-store-root", $Layout.ApiData,
                "--token-file", $Layout.ServingToken,
                "--host", "127.0.0.1", "--port", [string]$Ports.Gfm
            )
            IdentityTokens = @(
                "socialgraph_gfm.core.inference_cli",
                "--runtime-root", $Layout.ServingRoot,
                "--serving-control", $publishedServingControl,
                "--published-serving-root", $Layout.PublishedServingRoot,
                "--published-artifact-root", $Layout.PublishedServingArtifacts,
                "--artifact-root", $Layout.ServingArtifacts,
                "--global-model-root", $Layout.GlobalModelRoot,
                "--governance-root", $Layout.GovernanceRoot,
                "--dataset-store-root", $Layout.ApiData,
                "--token-file", $Layout.ServingToken,
                "--port", [string]$Ports.Gfm
            )
            Environment = $gfmEnvironment
        },
        [pscustomobject]@{
            Name = "socialgraph-api"
            Port = $Ports.Api
            Executable = $Layout.ApiPython
            WorkingDirectory = $Layout.Api
            Arguments = @("-m", "app", "--runtime-identity-root", $Layout.GovernanceRoot)
            IdentityTokens = @("-m", "app", "--runtime-identity-root", $Layout.GovernanceRoot)
            Environment = $apiEnvironment
        },
        [pscustomobject]@{
            Name = "governance-web"
            Port = $Ports.GovernanceWeb
            Executable = $node
            WorkingDirectory = $Layout.GovernanceWeb
            Arguments = @($governanceVite, $viteCommand, "--host", "127.0.0.1", "--port", [string]$Ports.GovernanceWeb, "--strictPort")
            IdentityTokens = @($governanceVite, "--port", [string]$Ports.GovernanceWeb)
            Environment = $governanceWebEnvironment
        }
    )
}

function Start-UnifiedStack {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Ports,
        [Parameter(Mandatory = $true)][ValidateSet("Development", "Production")][string]$Mode
    )

    Assert-UnifiedRuntimeReady -Layout $Layout -Mode $Mode
    Initialize-UnifiedDirectories -Layout $Layout
    $started = [System.Collections.Generic.List[string]]::new()
    try {
        foreach ($service in Get-UnifiedServices -Layout $Layout -Ports $Ports -Mode $Mode) {
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
    Write-Host "GFM loopback: http://127.0.0.1:$($Ports.Gfm)"
    Write-Host "Logs: $($Layout.LogRoot)"
}

function Stop-UnifiedStack {
    param([Parameter(Mandatory = $true)]$Layout)

    @("governance-web", "socialgraph-api", "gfm") | ForEach-Object {
        Stop-ManagedService -Layout $Layout -Name $_
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Layout.PidRoot "gfm.json"))) {
        Remove-Item -LiteralPath $Layout.ServingToken -ErrorAction SilentlyContinue
    }
}

# Focused modules override transitional definitions after the legacy support
# functions have loaded.
. (Join-Path $PSScriptRoot "Layout.ps1")
. (Join-Path $PSScriptRoot "PrivateConfiguration.ps1")
. (Join-Path $PSScriptRoot "ModelRuntime.ps1")
. (Join-Path $PSScriptRoot "RuntimeBundle.ps1")
. (Join-Path $PSScriptRoot "ProcessManager.ps1")
. (Join-Path $PSScriptRoot "Verification.ps1")
