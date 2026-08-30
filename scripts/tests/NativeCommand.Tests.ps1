$ErrorActionPreference = "Stop"
$scriptsRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $scriptsRoot "lib\NativeCommand.ps1")
. (Join-Path $scriptsRoot "lib\PythonLauncher.ps1")

$nativeShell = if ($env:OS -eq "Windows_NT") { $env:ComSpec } else { "/bin/sh" }
$successArguments = if ($env:OS -eq "Windows_NT") {
  @("/d", "/c", "exit 0")
} else {
  @("-c", "exit 0")
}
$failureArguments = if ($env:OS -eq "Windows_NT") {
  @("/d", "/c", "exit 7")
} else {
  @("-c", "exit 7")
}

Invoke-CheckedNative -FilePath $nativeShell -ArgumentList $successArguments

$failureObserved = $false
try {
  Invoke-CheckedNative -FilePath $nativeShell -ArgumentList $failureArguments
} catch {
  if ($_.Exception.Message -notmatch "exit code 7") {
    throw
  }
  $failureObserved = $true
}

if (-not $failureObserved) {
  throw "Invoke-CheckedNative did not reject a controlled non-zero exit."
}

# Leave callers with a successful native status after the deliberately caught failure.
Invoke-CheckedNative -FilePath $nativeShell -ArgumentList $successArguments

$python = Resolve-SocialGraphPythonLauncher
if ($python.Version -notlike "3.12.*") {
  throw "Python launcher selected an unexpected version: $($python.Version)"
}
if (-not (Test-Path -LiteralPath $python.FilePath -PathType Leaf)) {
  throw "Python launcher did not resolve to a real executable: $($python.FilePath)"
}
$explicitPython = Resolve-SocialGraphPythonLauncher -BootstrapPython $python.FilePath
if ($explicitPython.FilePath -ne $python.FilePath) {
  throw "Explicit Python probing did not preserve the resolved executable."
}

Write-Output "Native command exit-code checks passed."
