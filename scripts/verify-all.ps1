param(
    [switch]$SkipBuild,
    [switch]$SkipGfm,
    [switch]$SkipE2E
)

$profile = if ($SkipGfm) { "Offline" } else { "Cuda" }
& (Join-Path $PSScriptRoot "verify.ps1") `
    -Profile $profile `
    -SkipBuild:$SkipBuild `
    -SkipE2E:$SkipE2E
