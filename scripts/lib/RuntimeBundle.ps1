Set-StrictMode -Version Latest

$script:RuntimeBundleSchema = "socialgraph-fm.runtime-bundle/1.0"
$script:RuntimeInstallSchema = "socialgraph-fm.runtime-install/1.0"
$script:RuntimeBundleRoots = @(
    "bundles/models/socialgraph-global",
    "bundles/governance",
    "examples/governance"
)

function Get-RuntimeBundleStringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value))
        return ([BitConverter]::ToString($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-RuntimeBundleRole {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path.StartsWith("bundles/models/socialgraph-global/", [StringComparison]::Ordinal)) {
        return "model"
    }
    if ($Path.StartsWith("bundles/governance/knowledge/", [StringComparison]::Ordinal)) {
        return "knowledge"
    }
    if ($Path.StartsWith("bundles/governance/reviewed-cases/", [StringComparison]::Ordinal)) {
        return "reviewed_cases"
    }
    if ($Path.StartsWith("examples/governance/russia/", [StringComparison]::Ordinal)) {
        return "russia_example"
    }
    if ($Path.StartsWith("examples/governance/target-domain/", [StringComparison]::Ordinal)) {
        return "target_domain_example"
    }
    throw "The runtime bundle contains a path outside its allowed content roots: $Path"
}

function Assert-RuntimeBundleRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (
        [string]::IsNullOrWhiteSpace($Path) -or
        $Path.Contains("\") -or
        $Path.Contains(":") -or
        $Path.StartsWith("/", [StringComparison]::Ordinal) -or
        $Path.IndexOfAny([char[]]@(0, 10, 13)) -ge 0
    ) {
        throw "The runtime bundle path is not a portable repository-relative path: $Path"
    }
    $segments = @($Path.Split("/"))
    if ($segments.Count -lt 2 -or $segments -contains "" -or
        $segments -contains "." -or $segments -contains "..") {
        throw "The runtime bundle path contains an unsafe segment: $Path"
    }
}

function Assert-RuntimeBundle {
    param([Parameter(Mandatory = $true)]$Layout)

    $manifestPath = [IO.Path]::GetFullPath($Layout.RuntimeBundleManifest)
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "The public SocialGraph-FM runtime bundle manifest is missing: $manifestPath"
    }
    $manifestItem = Get-Item -LiteralPath $manifestPath -Force
    if (($manifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The public SocialGraph-FM runtime bundle manifest cannot be a link or reparse point."
    }
    try {
        $document = Get-Content -LiteralPath $manifestPath -Raw |
            ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "The public SocialGraph-FM runtime bundle manifest is invalid JSON."
    }
    if ($document.schemaVersion -ne $script:RuntimeBundleSchema -or
        $document.bundleVersion -ne "1.0.0") {
        throw "The public SocialGraph-FM runtime bundle manifest version is unsupported."
    }
    $roots = @($document.contentRoots)
    if ($roots.Count -ne $script:RuntimeBundleRoots.Count) {
        throw "The public SocialGraph-FM runtime bundle content-root inventory is invalid."
    }
    for ($index = 0; $index -lt $roots.Count; $index++) {
        if ([string]$roots[$index] -cne $script:RuntimeBundleRoots[$index]) {
            throw "The public SocialGraph-FM runtime bundle content-root inventory is invalid."
        }
    }

    $assets = @($document.assets)
    if ($assets.Count -eq 0 -or [int64]$document.fileCount -ne $assets.Count) {
        throw "The public SocialGraph-FM runtime bundle file count is invalid."
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    [long]$totalBytes = 0
    $inventoryBuilder = [Text.StringBuilder]::new()
    $manifestPaths = [Collections.Generic.List[string]]::new()
    foreach ($asset in $assets) {
        $relative = [string]$asset.path
        Assert-RuntimeBundleRelativePath -Path $relative
        if (-not $seen.Add($relative)) {
            throw "The public SocialGraph-FM runtime bundle contains a duplicate path: $relative"
        }
        $expectedRole = Get-RuntimeBundleRole -Path $relative
        if ([string]$asset.role -cne $expectedRole) {
            throw "The public SocialGraph-FM runtime bundle role is invalid for $relative."
        }
        $expectedHash = [string]$asset.sha256
        if ($expectedHash -cnotmatch "^[0-9a-f]{64}$" -or [int64]$asset.bytes -lt 0) {
            throw "The public SocialGraph-FM runtime bundle identity is invalid for $relative."
        }
        $absolute = [IO.Path]::GetFullPath((Join-Path $Layout.ProjectRoot ($relative -replace "/", "\")))
        if (-not $absolute.StartsWith(
            ([IO.Path]::GetFullPath($Layout.ProjectRoot).TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar),
            [StringComparison]::OrdinalIgnoreCase
        )) {
            throw "The public SocialGraph-FM runtime bundle path escaped the repository: $relative"
        }
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
            throw "The public SocialGraph-FM runtime bundle file is missing: $relative"
        }
        $item = Get-Item -LiteralPath $absolute -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "The public SocialGraph-FM runtime bundle cannot contain links: $relative"
        }
        if ([int64]$item.Length -ne [int64]$asset.bytes -or
            (Get-UnifiedFileSha256 -Path $absolute) -cne $expectedHash) {
            throw "The public SocialGraph-FM runtime bundle hash or size is invalid: $relative"
        }
        $totalBytes += [int64]$item.Length
        [void]$inventoryBuilder.Append(
            "$relative`t$($asset.bytes)`t$expectedHash`t$expectedRole`n"
        )
        $manifestPaths.Add($relative)
    }
    if ($totalBytes -ne [int64]$document.totalBytes -or
        (Get-RuntimeBundleStringSha256 -Value $inventoryBuilder.ToString()) -cne
            [string]$document.inventoryHash) {
        throw "The public SocialGraph-FM runtime bundle inventory identity is invalid."
    }

    $actualPaths = [Collections.Generic.List[string]]::new()
    foreach ($relativeRoot in $script:RuntimeBundleRoots) {
        $absoluteRoot = Join-Path $Layout.ProjectRoot ($relativeRoot -replace "/", "\")
        if (-not (Test-Path -LiteralPath $absoluteRoot -PathType Container)) {
            throw "The public SocialGraph-FM runtime bundle content root is missing: $relativeRoot"
        }
        $rootItem = Get-Item -LiteralPath $absoluteRoot -Force
        if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "The public SocialGraph-FM runtime bundle content root cannot be a link: $relativeRoot"
        }
        $linkedEntry = Get-ChildItem -LiteralPath $absoluteRoot -Recurse -Force |
            Where-Object {
                ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
            } | Select-Object -First 1
        if ($null -ne $linkedEntry) {
            throw "The public SocialGraph-FM runtime bundle cannot contain links: $($linkedEntry.FullName)"
        }
        foreach ($file in Get-ChildItem -LiteralPath $absoluteRoot -File -Recurse -Force) {
            $actualPaths.Add(
                $file.FullName.Substring($Layout.ProjectRoot.TrimEnd("\", "/").Length + 1).
                    Replace("\", "/")
            )
        }
    }
    [string[]]$sortedActual = @($actualPaths)
    [Array]::Sort($sortedActual, [StringComparer]::Ordinal)
    if ($sortedActual.Count -ne $manifestPaths.Count) {
        throw "The public SocialGraph-FM runtime bundle has an unmanifested or missing file."
    }
    for ($index = 0; $index -lt $sortedActual.Count; $index++) {
        if ($sortedActual[$index] -cne $manifestPaths[$index]) {
            throw "The public SocialGraph-FM runtime bundle inventory differs at $($sortedActual[$index])."
        }
    }

    $exportPath = Join-Path $Layout.RuntimeBundleModelRoot `
        "exports\socialgraph-global\export-manifest.json"
    try {
        $export = Get-Content -LiteralPath $exportPath -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "The bundled SocialGraph-FM Global export manifest is invalid."
    }
    foreach ($field in @(
        "releaseId", "modelVersionId", "modelVersionHash", "artifactHash", "corpusHash"
    )) {
        if ([string]$document.model.$field -cne [string]$export.$field) {
            throw "The public SocialGraph-FM runtime bundle model identity disagrees on $field."
        }
    }
    return [pscustomobject]@{
        Document = $document
        ManifestPath = $manifestPath
        ManifestSha256 = Get-UnifiedFileSha256 -Path $manifestPath
        Assets = $assets
    }
}

function Copy-RuntimeBundleAssets {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][object[]]$Assets,
        [Parameter(Mandatory = $true)][string]$SourcePrefix,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $prefix = $SourcePrefix.TrimEnd("/") + "/"
    foreach ($asset in $Assets) {
        $sourceRelative = [string]$asset.path
        if (-not $sourceRelative.StartsWith($prefix, [StringComparison]::Ordinal)) {
            throw "The runtime bundle copy prefix does not contain $sourceRelative."
        }
        $targetRelative = $sourceRelative.Substring($prefix.Length)
        Assert-RuntimeBundleRelativePath -Path ("runtime/" + $targetRelative)
        $source = Join-Path $Layout.ProjectRoot ($sourceRelative -replace "/", "\")
        $target = Join-Path $Destination ($targetRelative -replace "/", "\")
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
        if ((Get-UnifiedFileSha256 -Path $target) -cne [string]$asset.sha256) {
            throw "The staged runtime bundle copy failed verification: $sourceRelative"
        }
    }
}

function Assert-RuntimeBundleInstalledAssets {
    param(
        [Parameter(Mandatory = $true)][object[]]$Assets,
        [Parameter(Mandatory = $true)][string]$SourcePrefix,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $destinationItem = Get-Item -LiteralPath $Destination -Force
    if (-not $destinationItem.PSIsContainer -or
        ($destinationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The existing runtime bundle destination is unsafe: $Destination"
    }
    $prefix = $SourcePrefix.TrimEnd("/") + "/"
    foreach ($asset in $Assets) {
        $sourceRelative = [string]$asset.path
        if (-not $sourceRelative.StartsWith($prefix, [StringComparison]::Ordinal)) {
            throw "The runtime bundle installed-asset prefix does not contain $sourceRelative."
        }
        $targetRelative = $sourceRelative.Substring($prefix.Length)
        $target = Join-Path $Destination ($targetRelative -replace "/", "\")
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "The existing runtime bundle destination is incomplete: $targetRelative"
        }
        $item = Get-Item -LiteralPath $target -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            [int64]$item.Length -ne [int64]$asset.bytes -or
            (Get-UnifiedFileSha256 -Path $target) -cne [string]$asset.sha256) {
            throw "The existing runtime bundle destination differs from the public bundle: $targetRelative"
        }
    }
}

function Install-RuntimeBundleSeed {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][object[]]$Assets,
        [Parameter(Mandatory = $true)][string]$SourcePrefix,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Label,
        [string]$BundleVersion = "1.0.0",
        [switch]$MutableAfterInstall
    )

    if ($Assets.Count -eq 0) {
        throw "The public runtime bundle contains no $Label assets."
    }
    $identityBuilder = [Text.StringBuilder]::new()
    foreach ($asset in $Assets) {
        [void]$identityBuilder.Append(
            "$($asset.path)`t$($asset.bytes)`t$($asset.sha256)`n"
        )
    }
    $seedIdentity = Get-RuntimeBundleStringSha256 -Value $identityBuilder.ToString()
    $seedMarkerName = ".runtime-bundle-seed.json"
    if (Test-Path -LiteralPath $Destination) {
        if ($MutableAfterInstall) {
            $seedMarkerPath = Join-Path $Destination $seedMarkerName
            if (-not (Test-Path -LiteralPath $seedMarkerPath -PathType Leaf)) {
                throw "The existing mutable $Label state has no managed bundle seed marker."
            }
            try {
                $seedMarker = Get-Content -LiteralPath $seedMarkerPath -Raw |
                    ConvertFrom-Json -ErrorAction Stop
            }
            catch {
                throw "The existing mutable $Label bundle seed marker is invalid."
            }
            if ($seedMarker.schemaVersion -ne "socialgraph-fm.runtime-seed-install/1.0" -or
                [string]$seedMarker.bundleVersion -cne $BundleVersion -or
                [string]$seedMarker.sourcePrefix -cne $SourcePrefix -or
                [string]$seedMarker.seedIdentity -cne $seedIdentity) {
                throw "A different $Label bundle seed is already installed; refusing to overwrite runtime state."
            }
            Write-Host "The same bundled $Label seed is already installed; mutable runtime state was preserved."
            return
        }
        Assert-RuntimeBundleInstalledAssets -Assets $Assets -SourcePrefix $SourcePrefix `
            -Destination $Destination
        Write-Host "The same bundled $Label assets are already installed."
        return
    }
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $staging = Join-Path $parent ".$([IO.Path]::GetFileName($Destination)).$PID.$([Guid]::NewGuid().ToString('N')).stage"
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        Copy-RuntimeBundleAssets -Layout $Layout -Assets $Assets `
            -SourcePrefix $SourcePrefix -Destination $staging
        Assert-RuntimeBundleInstalledAssets -Assets $Assets -SourcePrefix $SourcePrefix `
            -Destination $staging
        if ($MutableAfterInstall) {
            $seedMarker = [ordered]@{
                schemaVersion = "socialgraph-fm.runtime-seed-install/1.0"
                bundleVersion = $BundleVersion
                sourcePrefix = $SourcePrefix
                seedIdentity = $seedIdentity
                sourceFileCount = $Assets.Count
            } | ConvertTo-Json
            [IO.File]::WriteAllText(
                (Join-Path $staging $seedMarkerName), "$seedMarker`n",
                [Text.UTF8Encoding]::new($false)
            )
        }
        Move-Item -LiteralPath $staging -Destination $Destination
        Write-Host "Bundled $Label assets installed at $Destination."
    }
    finally {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-RuntimeBundleModelCli {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][ValidateSet("verify", "smoke", "publish")]
        [string]$Operation
    )

    $arguments = switch ($Operation) {
        "verify" { @("_verify-export", "--root", $Root) }
        "smoke" { @("smoke", "--root", $Root) }
        "publish" { @("publish", "--root", $Root) }
    }
    Push-Location $Layout.GfmPackage
    try {
        & $Layout.GfmPython -m socialgraph_gfm.global_model.cli @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "The bundled SocialGraph-FM Global model $Operation gate failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

function Install-RuntimeBundleModel {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Bundle
    )

    $modelAssets = @($Bundle.Assets | Where-Object role -eq "model")
    $modelRoot = $Layout.GlobalModelRoot
    $markerPath = Join-Path $modelRoot "bundle-install.json"
    if (Test-Path -LiteralPath $modelRoot) {
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "An unmanaged SocialGraph-FM Global model already exists at $modelRoot; refusing to replace it."
        }
        try {
            $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            throw "The existing SocialGraph-FM Global model install marker is invalid."
        }
        if ($marker.schemaVersion -ne $script:RuntimeInstallSchema -or
            $marker.bundleVersion -cne [string]$Bundle.Document.bundleVersion -or
            $marker.modelVersionHash -cne [string]$Bundle.Document.model.modelVersionHash -or
            $marker.artifactHash -cne [string]$Bundle.Document.model.artifactHash -or
            $marker.corpusHash -cne [string]$Bundle.Document.model.corpusHash) {
            throw "A different SocialGraph-FM Global model bundle is already installed; refusing to replace it."
        }
        Assert-RuntimeBundleInstalledAssets -Assets $modelAssets `
            -SourcePrefix "bundles/models/socialgraph-global" -Destination $modelRoot
        Invoke-RuntimeBundleModelCli -Layout $Layout -Root $modelRoot -Operation verify
        Write-Host "The same bundled SocialGraph-FM Global model is already installed."
        return
    }

    $parent = Split-Path -Parent $modelRoot
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $staging = Join-Path $parent ".socialgraph-global.$PID.$([Guid]::NewGuid().ToString('N')).stage"
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        Copy-RuntimeBundleAssets -Layout $Layout -Assets $modelAssets `
            -SourcePrefix "bundles/models/socialgraph-global" -Destination $staging
        Assert-RuntimeBundleInstalledAssets -Assets $modelAssets `
            -SourcePrefix "bundles/models/socialgraph-global" -Destination $staging
        Invoke-RuntimeBundleModelCli -Layout $Layout -Root $staging -Operation verify
        Invoke-RuntimeBundleModelCli -Layout $Layout -Root $staging -Operation smoke
        Invoke-RuntimeBundleModelCli -Layout $Layout -Root $staging -Operation publish
        $registry = Get-Content -LiteralPath (Join-Path $staging "registry\socialgraph-global.json") `
            -Raw | ConvertFrom-Json -ErrorAction Stop
        if ($registry.state -ne "servingReady" -or
            $registry.modelVersionHash -cne [string]$Bundle.Document.model.modelVersionHash -or
            $registry.artifactHash -cne [string]$Bundle.Document.model.artifactHash -or
            $registry.corpusHash -cne [string]$Bundle.Document.model.corpusHash) {
            throw "The published bundled SocialGraph-FM Global registry identity is invalid."
        }
        $marker = [ordered]@{
            schemaVersion = $script:RuntimeInstallSchema
            bundleVersion = [string]$Bundle.Document.bundleVersion
            bundleManifestSha256 = $Bundle.ManifestSha256
            modelVersionId = [string]$Bundle.Document.model.modelVersionId
            modelVersionHash = [string]$Bundle.Document.model.modelVersionHash
            artifactHash = [string]$Bundle.Document.model.artifactHash
            corpusHash = [string]$Bundle.Document.model.corpusHash
            sourceFileCount = $modelAssets.Count
        } | ConvertTo-Json
        [IO.File]::WriteAllText(
            (Join-Path $staging "bundle-install.json"), "$marker`n",
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $staging -Destination $modelRoot
        Write-Host "Bundled SocialGraph-FM Global model installed at $modelRoot."
    }
    finally {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Install-PublicRuntimeBundle {
    param([Parameter(Mandatory = $true)]$Layout)

    if ($Layout.RuntimeProfile -eq "Offline" -or
        -not (Test-Path -LiteralPath $Layout.GfmPython -PathType Leaf)) {
        throw "Install a CPU or CUDA profile before installing the public SocialGraph-FM runtime bundle."
    }
    $bundle = Assert-RuntimeBundle -Layout $Layout
    Install-RuntimeBundleModel -Layout $Layout -Bundle $bundle
    Install-RuntimeBundleSeed -Layout $Layout `
        -Assets @($bundle.Assets | Where-Object role -eq "knowledge") `
        -SourcePrefix "bundles/governance/knowledge" `
        -Destination (Join-Path $Layout.GovernanceRoot "knowledge") -Label "knowledge"
    Install-RuntimeBundleSeed -Layout $Layout `
        -Assets @($bundle.Assets | Where-Object role -eq "reviewed_cases") `
        -SourcePrefix "bundles/governance/reviewed-cases" `
        -Destination (Join-Path $Layout.GovernanceRoot "reviewed-cases") -Label "reviewed-case" `
        -BundleVersion ([string]$bundle.Document.bundleVersion) -MutableAfterInstall
    Install-RuntimeBundleSeed -Layout $Layout `
        -Assets @($bundle.Assets | Where-Object {
            $_.role -eq "russia_example" -and $_.path -notlike "*/russia-full.zip"
        }) `
        -SourcePrefix "examples/governance/russia" `
        -Destination (Join-Path $Layout.GovernanceRoot "answer-packs\russia") `
        -Label "Russia answer-pack"
    Install-RuntimeBundleSeed -Layout $Layout `
        -Assets @($bundle.Assets | Where-Object {
            $_.role -eq "russia_example" -and $_.path -like "*/russia-full.zip"
        }) `
        -SourcePrefix "examples/governance/russia" `
        -Destination (Join-Path $Layout.GovernanceRoot "samples\russia") `
        -Label "Russia full-input"
    Install-RuntimeBundleSeed -Layout $Layout `
        -Assets @($bundle.Assets | Where-Object role -eq "target_domain_example") `
        -SourcePrefix "examples/governance/target-domain" `
        -Destination $Layout.GovernanceAdaptationInputs -Label "target-domain task"
    return $Layout.GlobalModelRoot
}
