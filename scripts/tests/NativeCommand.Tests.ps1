$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSScriptRoot) "lib\NativeCommand.ps1")

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
Write-Output "Native command exit-code checks passed."
