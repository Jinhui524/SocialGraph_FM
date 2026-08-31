param(
  [string]$UvExecutable = "uv"
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Invoke-CheckedUv {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  & $UvExecutable @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "uv failed with exit code $LASTEXITCODE"
  }
}

$uvVersion = ((& $UvExecutable --version) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
  throw "Unable to execute uv: $UvExecutable"
}
if ($uvVersion -notmatch '^uv 0\.12\.3(?:\s|$)') {
  throw "Lock regeneration requires uv 0.12.3; found: $uvVersion"
}

$profiles = @(
  @(
    "constraints\windows-cpu.txt", "x86_64-pc-windows-msvc", "cpu",
    "locks\windows-cpu.requirements.txt", "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
  ),
  @(
    "constraints\cpu-ci.txt", "x86_64-unknown-linux-gnu", "cpu",
    "locks\cpu-ci.requirements.txt", "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
  ),
  @(
    "constraints\public-runtime.in", "x86_64-pc-windows-msvc", "cpu",
    "locks\install-windows-x86_64-cpu-pt28.requirements.txt",
    "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
  ),
  @(
    "constraints\public-runtime.in", "x86_64-unknown-linux-gnu", "cpu",
    "locks\install-linux-x86_64-cpu-pt28.requirements.txt",
    "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
  )
)

Push-Location $projectRoot
try {
  foreach ($profile in $profiles) {
    $arguments = @(
      "pip", "compile", $profile[0],
      "--generate-hashes",
      "--python-platform", $profile[1],
      "--python-version", "3.12",
      "--index-url", "https://pypi.org/simple",
      "--torch-backend", $profile[2]
    )
    if ($profile[4]) {
      $arguments += @("--find-links", $profile[4])
    }
    $arguments += @("--no-build", "--output-file", $profile[3])
    Invoke-CheckedUv -Arguments $arguments
  }
} finally {
  Pop-Location
}
