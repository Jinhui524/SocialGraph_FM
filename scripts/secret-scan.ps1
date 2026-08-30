param(
  [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")),
  [int64]$MaximumFileBytes = 52428800
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "lib\NativeCommand.ps1")
$patterns = @(
  '(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}',
  '(?<![A-Za-z0-9_])gh[oprsu]_[A-Za-z0-9]{30,}',
  '(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{40,}',
  '(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])',
  '(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])',
  '(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{20,}',
  '(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
  '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
  '(?i)(api[_-]?key|secret|authorization|password|token)\s*[:=]\s*["''](?![^"'']*(test|fixture|example|with-at-least))[A-Za-z0-9_\-]{24,}["'']',
  '(?m)^\s*[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*(?!test|fixture|example|replace|dummy|hidden)[A-Za-z0-9_./+=-]{16,}\s*$',
  '(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{20,}'
)

$tracked = @(
  Invoke-CheckedNative -FilePath "git" -ArgumentList @(
    "-C", [string]$RepositoryRoot, "ls-files", "--cached", "--others", "--exclude-standard"
  )
)
$violations = @()
foreach ($relativePath in $tracked) {
  if ($relativePath -match '(^|/)\.env(?:\.|$)' -and $relativePath -notmatch '\.env\.example$') {
    $violations += $relativePath
    continue
  }
  $path = Join-Path $RepositoryRoot $relativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    continue
  }
  $item = Get-Item -LiteralPath $path -Force
  if ($item.Length -gt $MaximumFileBytes) {
    throw "Tracked file exceeds the secret-scan byte budget: $relativePath"
  }
  $content = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($path))
  foreach ($pattern in $patterns) {
    if ($content -cmatch $pattern) {
      $violations += $relativePath
      break
    }
  }
}

if ($violations.Count -gt 0) {
  $summary = ($violations | Sort-Object -Unique) -join ", "
  throw "Potential secret in tracked file(s): $summary"
}

Write-Output "Secret scan passed for $($tracked.Count) candidate files."
