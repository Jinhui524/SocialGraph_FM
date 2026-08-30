param([switch]$Check)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "contracts\core\serving"
$destinations = @(
    (Join-Path $projectRoot "services\api\app\contracts"),
    (Join-Path $projectRoot "packages\gfm\contracts")
)
$names = @(
    "core-serving-control.json",
    "core-serving-graph-catalog.json",
    "core-serving-registry.json"
)

foreach ($name in $names) {
    $source = Join-Path $sourceRoot $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Canonical serving contract is missing: $source"
    }
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    foreach ($destinationRoot in $destinations) {
        $destination = Join-Path $destinationRoot $name
        if ($Check) {
            if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
                throw "Generated serving contract is missing: $destination"
            }
            $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
            if ($destinationHash -ne $sourceHash) {
                throw "Generated serving contract is stale: $destination"
            }
            continue
        }
        New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
        $temporary = "$destination.tmp-$PID"
        try {
            Copy-Item -LiteralPath $source -Destination $temporary
            if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -ne $sourceHash) {
                throw "Serving contract copy failed verification: $name"
            }
            Move-Item -LiteralPath $temporary -Destination $destination -Force
        }
        finally {
            Remove-Item -LiteralPath $temporary -ErrorAction SilentlyContinue
        }
    }
}

Write-Output $(if ($Check) { "Serving contract mirrors are current." } else { "Serving contract mirrors updated." })
