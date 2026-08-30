param(
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [string]$ZipDestination,
    [string]$BootstrapPython,
    [string]$InitialCommitMessage = "Initial public SocialGraph-FM complete runtime snapshot",
    [string]$AuthorName = "SocialGraph-FM Contributors",
    [string]$AuthorEmail = "socialgraph-fm@users.noreply.github.com"
)

$ErrorActionPreference = "Stop"
$neutralName = "SocialGraph-FM Contributors"
$neutralEmail = "socialgraph-fm@users.noreply.github.com"
if ($AuthorName -cne $neutralName -or $AuthorEmail -cne $neutralEmail) {
    throw "Public exports require the neutral SocialGraph-FM contributor identity."
}

$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$destinationPath = [IO.Path]::GetFullPath($Destination)
if ([string]::IsNullOrWhiteSpace($ZipDestination)) {
    $zipPath = $destinationPath + ".zip"
}
else {
    $zipPath = [IO.Path]::GetFullPath($ZipDestination)
}

# Keep the established PowerShell publication policy as an additional gate.
$scanOutput = @(
    & (Join-Path $PSScriptRoot "publication-check.ps1") -RepositoryRoot $projectRoot
)
foreach ($line in $scanOutput) {
    [Console]::Error.WriteLine([string]$line)
}

. (Join-Path $PSScriptRoot "lib/PythonLauncher.ps1")
$launcher = Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython
$runtimeSource = Join-Path $projectRoot "packages/runtime/src"
$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
try {
    if ([string]::IsNullOrEmpty($previousPythonPath)) {
        $env:PYTHONPATH = $runtimeSource
    }
    else {
        $env:PYTHONPATH = $runtimeSource + [IO.Path]::PathSeparator + $previousPythonPath
    }
    & $launcher.FilePath -m socialgraph_fm_runtime.exporter `
        --source $projectRoot `
        --repository $destinationPath `
        --zip $zipPath `
        --message $InitialCommitMessage
    if ($LASTEXITCODE -ne 0) {
        throw "SocialGraph-FM public export failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
