[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON,
  [int]$EmbeddingProcessId = 0,
  [ValidateRange(10, 600)]
  [int]$PollSeconds = 30,
  [ValidateRange(8, 64)]
  [int]$MinimumFreeMemoryGiB = 8,
  [ValidateRange(1, 10080)]
  [int]$MemoryWaitTimeoutMinutes = 1440
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
  throw "Set SOCIALGRAPH_FM_HOME or pass -RuntimeRoot explicitly."
}
if ([string]::IsNullOrWhiteSpace($GfmPython)) {
  throw "Set SOCIALGRAPH_GFM_PYTHON or pass -GfmPython explicitly."
}
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$automationRoot = Join-Path $RuntimeRoot "reports\gfm\automation"
$statePath = Join-Path $automationRoot "dev-after-wikimedia-embedding.json"
$ownerPath = Join-Path $automationRoot "dev-after-wikimedia-embedding.owner.json"
$ownerLockPath = Join-Path $automationRoot "dev-after-wikimedia-embedding.owner.lock"
$finalEmbeddingManifest = Join-Path $RuntimeRoot `
  "embeddings\wikimedia-talk-article-2011-2015-bge-m3-v1\manifest.json"
[void](New-Item -ItemType Directory -Path $automationRoot -Force)
$automationScriptPath = [IO.Path]::GetFullPath($PSCommandPath)
$script:attemptId = "{0}-{1}-{2}" -f `
  [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ"), `
  $PID, `
  [Guid]::NewGuid().ToString("N").Substring(0, 8)
$script:startedAt = [DateTimeOffset]::UtcNow.ToString("O")
$script:expectedCodeHash = $null
$script:memoryWaitDeadline = $null
$script:recoveredStaleOwnerAttemptId = $null

function Write-AutomationState {
  param(
    [Parameter(Mandatory = $true)][string]$Status,
    [string]$Detail = ""
  )
  $payload = [ordered]@{
    schemaVersion = "gfm.dev-after-embedding-automation/1.0"
    status = $Status
    detail = $Detail
    embeddingProcessId = $EmbeddingProcessId
    automationPid = $PID
    attemptId = $script:attemptId
    startedAt = $script:startedAt
    memoryWaitDeadline = $script:memoryWaitDeadline
    codeHash = $script:expectedCodeHash
    recoveredStaleOwnerAttemptId = $script:recoveredStaleOwnerAttemptId
    runtimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
    updatedAt = [DateTimeOffset]::UtcNow.ToString("O")
  }
  $temporary = "$statePath.$PID.tmp"
  $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
  Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function Test-AutomationProcess {
  param([Parameter(Mandatory = $true)][int]$ProcessId)
  if ($ProcessId -le 0 -or $ProcessId -eq $PID) { return $false }
  $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" `
    -ErrorAction SilentlyContinue
  if ($null -eq $process -or [string]::IsNullOrWhiteSpace($process.CommandLine)) {
    return $false
  }
  return (
    $process.CommandLine.IndexOf("-File", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
    $process.CommandLine.IndexOf(
      $automationScriptPath,
      [StringComparison]::OrdinalIgnoreCase
    ) -ge 0
  )
}

function Get-OtherAutomationProcesses {
  return @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        $_.ProcessId -ne $PID -and
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine.IndexOf("-File", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $_.CommandLine.IndexOf(
          $automationScriptPath,
          [StringComparison]::OrdinalIgnoreCase
        ) -ge 0
      }
  )
}

function Get-CodeIdentity {
  $output = & $GfmPython -c `
    "from socialgraph_gfm.identity import code_identity_hash; print(code_identity_hash())"
  if ($LASTEXITCODE -ne 0) {
    throw "Unable to calculate the SocialGraphFM source identity."
  }
  $lines = @($output | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
  if ($lines.Count -lt 1) {
    throw "The SocialGraphFM source identity command returned no value."
  }
  $identity = $lines[-1].ToLowerInvariant()
  if ($identity -notmatch "^[0-9a-f]{64}$") {
    throw "The SocialGraphFM source identity is malformed."
  }
  return $identity
}

function Assert-CodeIdentity {
  param([Parameter(Mandatory = $true)][string]$Boundary)
  if ([string]::IsNullOrWhiteSpace($script:expectedCodeHash)) {
    throw "The automation has no startup source identity."
  }
  $currentCodeHash = Get-CodeIdentity
  if ($currentCodeHash -ne $script:expectedCodeHash) {
    throw "SocialGraphFM source changed $Boundary; expected $($script:expectedCodeHash), observed $currentCodeHash. Restart the automation explicitly."
  }
}

function Write-OwnerRecord {
  $payload = [ordered]@{
    schemaVersion = "gfm.dev-after-embedding-owner/1.0"
    automationPid = $PID
    attemptId = $script:attemptId
    startedAt = $script:startedAt
    codeHash = $script:expectedCodeHash
    scriptPath = $automationScriptPath
    runtimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
  }
  $temporary = "$ownerPath.$PID.tmp"
  $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
  Move-Item -LiteralPath $temporary -Destination $ownerPath -Force
}

function Remove-OwnerRecordIfOwned {
  if (-not (Test-Path -LiteralPath $ownerPath -PathType Leaf)) { return }
  try {
    $owner = Get-Content -Raw -LiteralPath $ownerPath | ConvertFrom-Json
    if (
      [int]$owner.automationPid -eq $PID -and
      [string]$owner.attemptId -eq $script:attemptId
    ) {
      Remove-Item -LiteralPath $ownerPath -Force
    }
  } catch {
    # Preserve an unreadable owner record for the next fail-closed stale-owner audit.
  }
}

function Test-ExpectedEmbeddingProcess {
  if ($EmbeddingProcessId -le 0) { return $false }
  $process = Get-CimInstance Win32_Process -Filter "ProcessId=$EmbeddingProcessId" `
    -ErrorAction SilentlyContinue
  if ($null -eq $process) { return $false }
  return (
    $process.CommandLine -like "*socialgraph_gfm.cli*gfm-text-embed*" -and
    $process.CommandLine -like "*wikimedia-talk*"
  )
}

function Get-FreePhysicalMemoryGiB {
  $operatingSystem = Get-CimInstance Win32_OperatingSystem
  return [math]::Round(
    ([int64]$operatingSystem.FreePhysicalMemory * 1KB) / 1GB,
    3
  )
}

function Wait-FreePhysicalMemory {
  param(
    [Parameter(Mandatory = $true)][string]$Status,
    [Parameter(Mandatory = $true)][string]$Workload
  )
  $deadline = [DateTimeOffset]::UtcNow.AddMinutes($MemoryWaitTimeoutMinutes)
  $script:memoryWaitDeadline = $deadline.ToString("O")
  Assert-CodeIdentity -Boundary "before the $Workload memory wait"
  while ($true) {
    $freeMemoryGiB = Get-FreePhysicalMemoryGiB
    if ($freeMemoryGiB -ge $MinimumFreeMemoryGiB) {
      Assert-CodeIdentity -Boundary "after the $Workload memory wait"
      $script:memoryWaitDeadline = $null
      return $freeMemoryGiB
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
      throw "$Workload waited $MemoryWaitTimeoutMinutes minutes for at least $MinimumFreeMemoryGiB GiB free physical memory; only $freeMemoryGiB GiB is available."
    }
    Assert-CodeIdentity -Boundary "during the $Workload memory wait"
    Write-AutomationState `
      -Status $Status `
      -Detail "$Workload is waiting for at least $MinimumFreeMemoryGiB GiB free physical memory; $freeMemoryGiB GiB is currently available."
    Start-Sleep -Seconds $PollSeconds
  }
}

function Invoke-GfmCli {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
  $commandName = if ($Arguments.Count -gt 0) { $Arguments[0] } else { "unknown command" }
  Assert-CodeIdentity -Boundary "before $commandName"
  & $GfmPython -m socialgraph_gfm.cli @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "GFM command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
  }
  Assert-CodeIdentity -Boundary "after $commandName"
}

$otherAutomation = @(Get-OtherAutomationProcesses)
if ($otherAutomation.Count -gt 0) {
  $otherPids = ($otherAutomation | ForEach-Object { [string]$_.ProcessId }) -join ","
  throw "Another Invoke-GfmDevAfterEmbedding automation is active (PID $otherPids); refusing to overwrite its state."
}

$ownerLock = $null
try {
  $ownerLock = [IO.File]::Open(
    $ownerLockPath,
    [IO.FileMode]::OpenOrCreate,
    [IO.FileAccess]::ReadWrite,
    [IO.FileShare]::None
  )
} catch {
  throw "Another Invoke-GfmDevAfterEmbedding automation owns the exclusive lock."
}

if (Test-Path -LiteralPath $ownerPath -PathType Leaf) {
  try {
    $previousOwner = Get-Content -Raw -LiteralPath $ownerPath | ConvertFrom-Json
    $previousOwnerPid = [int]$previousOwner.automationPid
    if (Test-AutomationProcess -ProcessId $previousOwnerPid) {
      throw "The recorded automation owner PID $previousOwnerPid is still active."
    }
    $script:recoveredStaleOwnerAttemptId = [string]$previousOwner.attemptId
  } catch {
    if ($_.Exception.Message -like "The recorded automation owner PID*") {
      $ownerLock.Dispose()
      throw
    }
    $script:recoveredStaleOwnerAttemptId = "unreadable-owner-record"
  }
}

$previousPythonPath = $env:PYTHONPATH
try {
  . (Join-Path $PSScriptRoot "Enter-GfmRuntime.ps1") `
    -RuntimeRoot $RuntimeRoot `
    -GfmPython $GfmPython `
    -Operation run `
    -DependencyProfile text | Out-Null
  $env:PYTHONPATH = Join-Path $repositoryRoot "src"
  $script:expectedCodeHash = Get-CodeIdentity
  Write-OwnerRecord
  $startupDetail = if ($null -eq $script:recoveredStaleOwnerAttemptId) {
    "Exclusive automation owner and source identity established."
  } else {
    "Recovered stale owner attempt $($script:recoveredStaleOwnerAttemptId); exclusive owner and source identity established."
  }
  Write-AutomationState -Status "starting" -Detail $startupDetail

  while (
    -not (Test-Path -LiteralPath $finalEmbeddingManifest -PathType Leaf) -and
    (Test-ExpectedEmbeddingProcess)
  ) {
    Assert-CodeIdentity -Boundary "while waiting for the Wikimedia embedding"
    Write-AutomationState `
      -Status "waiting-for-wikimedia-embedding" `
      -Detail "The existing formal embedding process still owns the artifact."
    Start-Sleep -Seconds $PollSeconds
  }

  # This command is deliberately repeated. It either resumes an interrupted
  # hash-bound shard set or fully revalidates an already published artifact.
  Write-AutomationState -Status "verifying-wikimedia-embedding"
  Invoke-GfmCli `
    "gfm-text-embed" `
    "--encoder" "BAAI/bge-m3" `
    "--domain" "wikimedia-talk" `
    "--root" $RuntimeRoot `
    "--json"

  Write-AutomationState -Status "verifying-collaboration-assets"
  Invoke-GfmCli `
    "gfm-task-assets" `
    "--task" "collaboration" `
    "--root" $RuntimeRoot `
    "--json"

  $null = Wait-FreePhysicalMemory `
    -Status "waiting-for-dev-memory" `
    -Workload "Dev pretraining"

  Write-AutomationState -Status "running-dev-core-base"
  Invoke-GfmCli `
    "gfm-pretrain" `
    "--phase" "dev" `
    "--config" "socialgraph-core.json" `
    "--variant" "core-base" `
    "--device" "cuda" `
    "--root" $RuntimeRoot `
    "--json"

  $null = Wait-FreePhysicalMemory `
    -Status "waiting-for-core-moe-memory" `
    -Workload "Core-MoE dev"

  Write-AutomationState -Status "running-dev-core-moe"
  Invoke-GfmCli `
    "gfm-pretrain" `
    "--phase" "dev" `
    "--config" "socialgraph-core.json" `
    "--variant" "core-moe" `
    "--device" "cuda" `
    "--root" $RuntimeRoot `
    "--json"

  Write-AutomationState `
    -Status "succeeded" `
    -Detail "Wikimedia embedding and both fixed dev variants completed."
} catch {
  Write-AutomationState -Status "failed" -Detail $_.Exception.Message
  throw
} finally {
  $env:PYTHONPATH = $previousPythonPath
  Remove-OwnerRecordIfOwned
  if ($null -ne $ownerLock) { $ownerLock.Dispose() }
}
