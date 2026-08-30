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

$projectRoot = Split-Path -Parent $scriptsRoot
$layout = Get-UnifiedLayout -ProjectRoot $projectRoot -Profile Cpu
$bundle = Assert-RuntimeBundle -Layout $layout
$assets = @($bundle.Assets)

Assert-True ($bundle.Document.schemaVersion -eq "socialgraph-fm.runtime-bundle/1.0") `
    "The runtime bundle schema changed."
Assert-True ($bundle.Document.model.modelVersionId -cmatch '^socialgraph-fm-global/[0-9a-f]{16}$') `
    "The bundled Global model identity changed."
Assert-True (@($assets | Where-Object role -eq "model").Count -eq 165) `
    "The minimal model/Russia serving inventory changed."
Assert-True (@($assets | Where-Object { $_.path -like "*/checkpoints/*.pt" }).Count -eq 4) `
    "The runtime bundle does not contain the four frozen model checkpoints."
Assert-True (@($assets | Where-Object role -eq "russia_example").Count -eq 6) `
    "Russia 1-4, catalog, and the deduplicated full input are not exact."
Assert-True (@($assets | Where-Object role -eq "target_domain_example").Count -eq 3) `
    "The target-domain catalog and two task archives are not exact."
Assert-True (-not ($assets.path -match "(^|/)(smoke-report|registry|registry-candidate)\.json$")) `
    "A machine-derived smoke or registry file entered the tracked bundle."
Assert-True (-not ($assets.path -match "research|/runs/|corpus/countries/(?!russia/)")) `
    "A training run, SocialGraph-FM Research asset, or non-Russia corpus entered the bundle."

$testDrive = [IO.Path]::GetPathRoot($projectRoot)
$testParent = Join-Path $testDrive "sgfm-tmp"
New-Item -ItemType Directory -Force -Path $testParent | Out-Null
$testRoot = Join-Path $testParent "bundle-$PID-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    foreach ($relativeRoot in @("bundles\models", "bundles\governance", "examples\governance")) {
        $source = Join-Path $projectRoot $relativeRoot
        $destination = Join-Path $testRoot $relativeRoot
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Recurse
    }
    Copy-Item -LiteralPath $layout.RuntimeBundleManifest `
        -Destination (Join-Path $testRoot "bundles\runtime-manifest.json")
    $copyLayout = Get-UnifiedLayout -ProjectRoot $testRoot -Profile Cpu
    Assert-RuntimeBundle -Layout $copyLayout | Out-Null
    [IO.File]::AppendAllText(
        (Join-Path $copyLayout.RuntimeBundleModelRoot `
            "exports\socialgraph-global\model-card.json"),
        " ", [Text.UTF8Encoding]::new($false)
    )
    Assert-Throws {
        Assert-RuntimeBundle -Layout $copyLayout | Out-Null
    } "A modified bundled model document was accepted."

    $targetAssets = @($assets | Where-Object role -eq "target_domain_example")
    $targetSeed = Join-Path $testRoot "seed\adaptation-inputs"
    Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
        -SourcePrefix "examples/governance/target-domain" `
        -Destination $targetSeed -Label "test target-domain" | Out-Null
    Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
        -SourcePrefix "examples/governance/target-domain" `
        -Destination $targetSeed -Label "test target-domain" | Out-Null
    [IO.File]::AppendAllText(
        (Join-Path $targetSeed "governance-target-tasks.catalog.json"),
        " ", [Text.UTF8Encoding]::new($false)
    )
    Assert-Throws {
        Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
            -SourcePrefix "examples/governance/target-domain" `
            -Destination $targetSeed -Label "test target-domain" | Out-Null
    } "A different existing target-domain seed was silently overwritten."

    $mutableSeed = Join-Path $testRoot "seed\mutable-reviewed-cases"
    Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
        -SourcePrefix "examples/governance/target-domain" `
        -Destination $mutableSeed -Label "test mutable" -MutableAfterInstall | Out-Null
    [IO.File]::AppendAllText(
        (Join-Path $mutableSeed "governance-target-tasks.catalog.json"),
        " runtime change", [Text.UTF8Encoding]::new($false)
    )
    Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
        -SourcePrefix "examples/governance/target-domain" `
        -Destination $mutableSeed -Label "test mutable" -MutableAfterInstall | Out-Null
    Assert-Throws {
        Install-RuntimeBundleSeed -Layout $layout -Assets $targetAssets `
            -SourcePrefix "examples/governance/target-domain" `
            -Destination $mutableSeed -Label "test mutable" `
            -BundleVersion "2.0.0" -MutableAfterInstall | Out-Null
    } "A different mutable bundle seed version was silently accepted."

    $differentModel = Join-Path $testRoot "different-model"
    New-Item -ItemType Directory -Path $differentModel | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $differentModel "bundle-install.json"),
        '{"schemaVersion":"socialgraph-fm.runtime-install/1.0","bundleManifestSha256":"' +
            ('0' * 64) + '"}',
        [Text.UTF8Encoding]::new($false)
    )
    $differentLayout = [pscustomobject]@{ GlobalModelRoot = $differentModel }
    Assert-Throws {
        Install-RuntimeBundleModel -Layout $differentLayout -Bundle $bundle | Out-Null
    } "A different existing model bundle was silently replaced."
}
finally {
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $expectedPrefix = [IO.Path]::GetFullPath($testParent).TrimEnd("\") + "\bundle-"
    if (-not $resolved.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a runtime-bundle test path outside the verified short temp root."
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Runtime bundle tests passed."
