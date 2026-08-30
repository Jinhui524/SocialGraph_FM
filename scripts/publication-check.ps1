param([string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")))

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($RepositoryRoot)

& (Join-Path $PSScriptRoot "secret-scan.ps1") -RepositoryRoot $root
& (Join-Path $PSScriptRoot "sync-contracts.ps1") -Check
. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$pythonLauncher = Resolve-SocialGraphPythonLauncher
& $pythonLauncher.FilePath (Join-Path $PSScriptRoot "brand-scan.py") `
    --repository-root $root
if ($LASTEXITCODE -ne 0) { throw "Brand scan rejected the publication candidate." }

$tracked = @(git -C $root ls-files)
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate tracked files." }
$candidates = @(git -C $root ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate publication candidate files." }
$runtimeManifestPath = Join-Path $root "bundles\runtime-manifest.json"
if (-not (Test-Path -LiteralPath $runtimeManifestPath -PathType Leaf)) {
    throw "The public SocialGraph-FM runtime bundle manifest is missing."
}
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding UTF8 |
    ConvertFrom-Json -ErrorAction Stop
if ($runtimeManifest.schemaVersion -ne "socialgraph-fm.runtime-bundle/1.0") {
    throw "The public SocialGraph-FM runtime bundle manifest schema is unsupported."
}
$runtimeAssets = [Collections.Generic.Dictionary[string, object]]::new(
    [StringComparer]::Ordinal
)
foreach ($asset in @($runtimeManifest.assets)) {
    $relative = [string]$asset.path
    if ([string]::IsNullOrWhiteSpace($relative) -or
        $relative.Contains("\") -or $relative.Contains(":") -or
        $relative.StartsWith("/") -or
        @($relative.Split("/") | Where-Object { $_ -in @("", ".", "..") }).Count -gt 0 -or
        [string]$asset.sha256 -cnotmatch '^[0-9a-f]{64}$' -or
        [int64]$asset.bytes -lt 0 -or
        $runtimeAssets.ContainsKey($relative)) {
        throw "The public SocialGraph-FM runtime bundle contains an invalid asset entry: $relative"
    }
    $runtimeAssets.Add($relative, $asset)
    if ($tracked -notcontains $relative) {
        throw "A public SocialGraph-FM runtime bundle asset is not tracked: $relative"
    }
    $path = Join-Path $root ($relative -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Item -LiteralPath $path -Force).Length -ne [int64]$asset.bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne
            [string]$asset.sha256) {
        throw "A public SocialGraph-FM runtime bundle asset failed size/hash verification: $relative"
    }
}
if ($runtimeAssets.Count -ne [int]$runtimeManifest.fileCount) {
    throw "The public SocialGraph-FM runtime bundle file count is invalid."
}
$forbiddenPaths = @(
    $candidates | Where-Object {
        $_ -match '(^|/)\.superpowers/' -or
        $_ -match '^platform/' -or
        ($_ -like 'var/*' -and $_ -ne 'var/.gitkeep') -or
        ($_ -match '(^|/)\.env(?:\.|$)' -and $_ -notmatch '\.env\.example$') -or
        (($_ -match '\.(pt|pth|ckpt|safetensors|onnx|pkl|pickle|npy|npz|joblib|h5|hdf5|pb|parquet|sqlite|sqlite3|tar|tgz|7z)$' -or
          $_ -match '\.tar\.(gz|bz2|xz)$') -and -not $runtimeAssets.ContainsKey($_)) -or
        (($_ -match '^bundles/(models|governance)/' -or $_ -match '^examples/governance/') -and
          $_ -ne 'bundles/runtime-manifest.json' -and -not $runtimeAssets.ContainsKey($_))
    }
)
if ($forbiddenPaths.Count -gt 0) {
    throw "Forbidden public paths are tracked: $($forbiddenPaths -join ', ')"
}

$allowlistPath = Join-Path $root "scripts\publication-allowlist.json"
if (-not (Test-Path -LiteralPath $allowlistPath -PathType Leaf)) {
    throw "Publication allowlist is missing."
}
$allowlist = Get-Content -LiteralPath $allowlistPath -Raw -Encoding UTF8 |
    ConvertFrom-Json -ErrorAction Stop

function Get-JsonPropertyValue {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [System.Collections.IDictionary]) { return $Object[$Name] }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}
if ($allowlist.schemaVersion -ne "socialgraph-fm.publication-allowlist/1.0") {
    throw "Publication allowlist schema is unsupported."
}

$binaryMetadataViolations = [System.Collections.Generic.List[string]]::new()
foreach ($relative in @($candidates | Where-Object { $_ -match '\.(png|jpe?g|webp|woff2)$' })) {
    $path = Join-Path $root $relative
    $bytes = [IO.File]::ReadAllBytes($path)
    $extension = [IO.Path]::GetExtension($relative).ToLowerInvariant()
    $magicMatches = switch ($extension) {
        ".png" { $bytes.Length -ge 8 -and [BitConverter]::ToString($bytes[0..7]) -eq "89-50-4E-47-0D-0A-1A-0A" }
        ".jpg" { $bytes.Length -ge 3 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xD8 -and $bytes[2] -eq 0xFF }
        ".jpeg" { $bytes.Length -ge 3 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xD8 -and $bytes[2] -eq 0xFF }
        ".webp" {
            $bytes.Length -ge 12 -and
            [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -eq "RIFF" -and
            [Text.Encoding]::ASCII.GetString($bytes, 8, 4) -eq "WEBP"
        }
        default { $true }
    }
    if (-not $magicMatches) {
        $binaryMetadataViolations.Add("extension/magic mismatch: $relative")
        continue
    }
    $decoded = [Text.Encoding]::UTF8.GetString($bytes)
    if ($decoded -match '(?i)C:[/\\]Users[/\\]|E:[/\\]project[/\\]SocialGraph_FM') {
        $binaryMetadataViolations.Add("machine path metadata: $relative")
    }
    if ($decoded -match '(?i)c2pa|caBX') {
        $entry = Get-JsonPropertyValue -Object $allowlist.binaryMetadata -Name $relative
        if ($null -eq $entry -or $entry.allowC2pa -ne $true) {
            $binaryMetadataViolations.Add("unapproved C2PA metadata: $relative")
        }
    }
}
$binaryTextPatterns = @(
    '(?i)C:[/\\]Users[/\\]',
    '(?i)E:[/\\]project[/\\]SocialGraph_FM',
    '(?i)C:\\\\Users\\\\[A-Za-z0-9]',
    '(?i)E:\\\\project\\\\SocialGraph_FM',
    '(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}',
    '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
)
foreach ($relative in @($candidates | Where-Object {
    $_ -match '\.(pt|pth|ckpt|npy|npz|sqlite|sqlite3)$'
})) {
    $path = Join-Path $root $relative
    $bytes = [IO.File]::ReadAllBytes($path)
    $decodedUtf8 = [Text.Encoding]::UTF8.GetString($bytes)
    $decodedUtf16 = [Text.Encoding]::Unicode.GetString($bytes)
    foreach ($pattern in $binaryTextPatterns) {
        if ($decodedUtf8 -match $pattern -or $decodedUtf16 -match $pattern) {
            $binaryMetadataViolations.Add("sensitive binary text: $relative")
            break
        }
    }
}
if ($binaryMetadataViolations.Count -gt 0) {
    throw "Binary metadata policy violations:`n$($binaryMetadataViolations -join "`n")"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archiveSecretPatterns = @(
    '(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}',
    '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
    '(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{20,}'
)
foreach ($relative in @($candidates | Where-Object { $_ -match '\.(zip|npz|pt)$' })) {
    $runtimePolicy = if ($runtimeAssets.ContainsKey($relative)) {
        $runtimeAssets[$relative]
    }
    else { $null }
    $entryPolicy = if ($null -eq $runtimePolicy) {
        Get-JsonPropertyValue -Object $allowlist.archives -Name $relative
    }
    else { $null }
    if ($null -eq $runtimePolicy -and
        ($null -eq $entryPolicy -or $entryPolicy.synthetic -ne $true)) {
        throw "Tracked archive is neither a manifest-bound runtime asset nor an approved synthetic fixture: $relative"
    }
    $path = Join-Path $root $relative
    $sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedSha256 = if ($null -ne $runtimePolicy) {
        [string]$runtimePolicy.sha256
    }
    else { [string]$entryPolicy.sha256 }
    if ($sha256 -cne $expectedSha256) {
        throw "Approved archive hash changed; review its members and update the allowlist: $relative"
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($path)
    try {
        if ($archive.Entries.Count -gt 100) { throw "Archive contains too many members: $relative" }
        if ($null -ne $entryPolicy) {
            $expectedMembers = @($entryPolicy.members | Sort-Object)
            $actualMembers = @($archive.Entries | ForEach-Object { $_.FullName } | Sort-Object)
            if (($expectedMembers -join "`0") -ne ($actualMembers -join "`0")) {
                throw "Archive member inventory changed: $relative"
            }
        }
        $expandedBytes = [int64]0
        foreach ($member in $archive.Entries) {
            $name = $member.FullName
            if ($name.Contains("\\") -or $name.StartsWith("/") -or $name.Contains(":") -or
                @($name.Split("/") | Where-Object { $_ -eq ".." }).Count -gt 0) {
                throw "Archive contains an unsafe member path: $relative -> $name"
            }
            $expandedBytes += [int64]$member.Length
            if ($expandedBytes -gt 20971520) { throw "Archive exceeds expanded byte budget: $relative" }
            $stream = $member.Open()
            $memory = [IO.MemoryStream]::new()
            try {
                $stream.CopyTo($memory)
                $content = [Text.Encoding]::UTF8.GetString($memory.ToArray())
                foreach ($pattern in $archiveSecretPatterns) {
                    if ($content -match $pattern) {
                        throw "Potential secret in archive member: $relative -> $name"
                    }
                }
                if ($content -match '(?i)C:[/\\]Users[/\\]|E:[/\\]project[/\\]SocialGraph_FM') {
                    throw "Machine-specific path in archive member: $relative -> $name"
                }
            }
            finally {
                $memory.Dispose()
                $stream.Dispose()
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}

$textExtensions = @('.md', '.ps1', '.py', '.ts', '.tsx', '.js', '.mjs', '.json', '.toml', '.yml', '.yaml')
$machinePathViolations = [System.Collections.Generic.List[string]]::new()
foreach ($relative in $candidates) {
    $extension = [IO.Path]::GetExtension($relative).ToLowerInvariant()
    if ($textExtensions -notcontains $extension) { continue }
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $content = [IO.File]::ReadAllText($path)
    if (
        $content -match '(?i)C:[/\\]Users[/\\][^<\s]' -or
        $content -match '(?i)E:[/\\]project[/\\]SocialGraph_FM' -or
        $content -match '(?i)C:\\\\Users\\\\[A-Za-z0-9]' -or
        $content -match '(?i)E:\\\\project\\\\SocialGraph_FM'
    ) {
        $machinePathViolations.Add($relative)
    }
}
if ($machinePathViolations.Count -gt 0) {
    throw "Machine-specific paths remain in public text: $($machinePathViolations -join ', ')"
}

$brokenLinks = [System.Collections.Generic.List[string]]::new()
foreach ($relative in @($candidates | Where-Object { $_.EndsWith('.md') })) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $content = [IO.File]::ReadAllText($path)
    foreach ($match in [regex]::Matches($content, '\[[^\]]*\]\(([^)]+)\)')) {
        $target = $match.Groups[1].Value.Trim().Trim('<', '>')
        if ($target -match '^(?i:https?://|mailto:|#)') { continue }
        $target = ($target -split '#', 2)[0]
        if ([string]::IsNullOrWhiteSpace($target)) { continue }
        $decoded = [Uri]::UnescapeDataString($target)
        $candidate = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $path) $decoded))
        if (-not (Test-Path -LiteralPath $candidate)) {
            $brokenLinks.Add("$relative -> $target")
        }
    }
}
if ($brokenLinks.Count -gt 0) {
    throw "Broken local Markdown links:`n$($brokenLinks -join "`n")"
}

$lock = Join-Path $root "apps\web\package-lock.json"
if ((Test-Path -LiteralPath $lock) -and [IO.File]::ReadAllText($lock).Contains('registry.npmmirror.com')) {
    throw "apps/web/package-lock.json still contains registry.npmmirror.com URLs."
}

Write-Output "Publication policy checks passed for $($candidates.Count) candidate files."
