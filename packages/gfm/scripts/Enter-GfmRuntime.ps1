[CmdletBinding()]
param(
  [string]$RuntimeRoot = $env:SOCIALGRAPH_FM_HOME,
  [string]$GfmPython = $env:SOCIALGRAPH_GFM_PYTHON,
  [ValidateSet("fetch", "run")]
  [string]$Operation = "run",
  [ValidateSet("base", "text")]
  [string]$DependencyProfile = "base",
  [switch]$PromptForSecrets,
  [switch]$PromptForWikimediaSalt,
  [scriptblock]$SecretAction
)

# Codex and other host runtimes may prepend shadow copies of the Windows
# security module. Secret prompts must resolve the OS-owned implementation.
if ($PSVersionTable.PSEdition -eq "Desktop") {
  $systemModuleRoot = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\Modules"
  if (-not (Test-Path -LiteralPath $systemModuleRoot -PathType Container)) {
    throw "The OS-owned Windows PowerShell module directory is unavailable."
  }
  $modulePaths = @($env:PSModulePath -split [IO.Path]::PathSeparator) |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -ne $systemModuleRoot }
  $env:PSModulePath = (@($systemModuleRoot) + $modulePaths) -join [IO.Path]::PathSeparator
}

# Reuse the audited process-scoped launcher. Dot-source this script when the caller
# needs its environment values; neither script changes user or machine variables.
$runtimeContext = . (Join-Path $PSScriptRoot "Enter-BaselineRuntime.ps1") `
  -RuntimeRoot $RuntimeRoot `
  -GfmPython $GfmPython `
  -Operation $Operation

# Corpus fetch/prepare and non-text tests need only the base CUDA profile. Request the
# optional profile before BGE-M3 embedding so a partial environment fails before a heavy
# job starts. Distribution metadata is checked without importing the model stack or
# downloading model weights.
if ($DependencyProfile -eq "text") {
  try {
    $flagEmbeddingVersion = (
      & $runtimeContext.GfmPython -c `
        "import importlib.metadata as m; print(m.version('FlagEmbedding'))" 2>$null |
        Out-String
    ).Trim()
    $flagEmbeddingExitCode = $LASTEXITCODE
    $transformersVersion = (
      & $runtimeContext.GfmPython -c `
        "import importlib.metadata as m; print(m.version('transformers'))" 2>$null |
        Out-String
    ).Trim()
    $transformersExitCode = $LASTEXITCODE
    if (
      $flagEmbeddingExitCode -ne 0 -or $flagEmbeddingVersion -ne "1.4.0" -or
      $transformersExitCode -ne 0 -or $transformersVersion -ne "5.14.1"
    ) {
      throw "optional dependency version mismatch"
    }
  } catch {
    throw "The windows-cu130-gfm optional runtime is not installed exactly. Sync locks\windows-cu130-gfm.requirements.txt with --require-hashes --no-build."
  }
}

$profileName = if ($DependencyProfile -eq "text") { "windows-cu130-gfm" } else { "windows-cu130" }
$runtimeContext | Add-Member -NotePropertyName RuntimeLockProfile -NotePropertyValue $profileName

# An already-running Codex process cannot inherit variables added later in another
# terminal. This explicit interactive boundary avoids copying either secret into a
# command line, transcript, repository file, runtime artifact, or manifest. The action
# runs while the values exist only in this PowerShell process; the previous process
# values are restored even when the action fails.
if ($PromptForSecrets -and $PromptForWikimediaSalt) {
  throw "Choose either -PromptForSecrets or -PromptForWikimediaSalt, not both."
}
$promptForAnySecret = $PromptForSecrets -or $PromptForWikimediaSalt
if ($promptForAnySecret -and $null -eq $SecretAction) {
  throw "A secret prompt requires -SecretAction so secrets can be cleared deterministically."
}
if (-not $promptForAnySecret -and $null -ne $SecretAction) {
  throw "-SecretAction is only valid with a secret prompt."
}

if ($promptForAnySecret) {
  $openAlexVariable = "OPENALEX_API_KEY"
  $saltVariable = "SOCIALGRAPH_GFM_PSEUDONYM_SALT"
  $previousOpenAlex = $null
  $previousSalt = [Environment]::GetEnvironmentVariable(
    $saltVariable,
    [EnvironmentVariableTarget]::Process
  )
  $openAlexSecure = $null
  $saltSecure = $null
  $openAlexPlain = $null
  $saltPlain = $null

  try {
    if ($PromptForSecrets) {
      $previousOpenAlex = [Environment]::GetEnvironmentVariable(
        $openAlexVariable,
        [EnvironmentVariableTarget]::Process
      )
      $openAlexSecure = Read-Host -Prompt "OpenAlex API key" -AsSecureString
    }
    $saltSecure = Read-Host -Prompt "Stable Wikimedia pseudonym salt (at least 32 UTF-8 bytes)" -AsSecureString
    if (($PromptForSecrets -and $null -eq $openAlexSecure) -or $null -eq $saltSecure) {
      throw "Every requested secret prompt is required."
    }

    if ($PromptForSecrets) {
      $openAlexPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($openAlexSecure)
      try {
        $openAlexPlain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($openAlexPointer).Trim()
      } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($openAlexPointer)
      }
    }
    $saltPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($saltSecure)
    try {
      $saltPlain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($saltPointer).Trim()
    } finally {
      [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($saltPointer)
    }

    if ($PromptForSecrets -and [string]::IsNullOrWhiteSpace($openAlexPlain)) {
      throw "The OpenAlex API key cannot be empty."
    }
    if ([Text.Encoding]::UTF8.GetByteCount($saltPlain) -lt 32) {
      throw "The Wikimedia pseudonym salt must contain at least 32 UTF-8 bytes."
    }

    if ($PromptForSecrets) {
      [Environment]::SetEnvironmentVariable(
        $openAlexVariable,
        $openAlexPlain,
        [EnvironmentVariableTarget]::Process
      )
    }
    [Environment]::SetEnvironmentVariable(
      $saltVariable,
      $saltPlain,
      [EnvironmentVariableTarget]::Process
    )
    & $SecretAction $runtimeContext
  } finally {
    if ($PromptForSecrets) {
      [Environment]::SetEnvironmentVariable(
        $openAlexVariable,
        $previousOpenAlex,
        [EnvironmentVariableTarget]::Process
      )
    }
    [Environment]::SetEnvironmentVariable(
      $saltVariable,
      $previousSalt,
      [EnvironmentVariableTarget]::Process
    )
    if ($null -ne $openAlexSecure) { $openAlexSecure.Dispose() }
    if ($null -ne $saltSecure) { $saltSecure.Dispose() }
    $openAlexPlain = $null
    $saltPlain = $null
    $previousOpenAlex = $null
    $previousSalt = $null
  }
  return
}

$runtimeContext
