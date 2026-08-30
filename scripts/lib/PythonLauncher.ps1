Set-StrictMode -Version Latest

function Test-SocialGraphPythonLauncher {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$PrefixArguments = @()
    )

    try {
        # Windows PowerShell 5.1 serializes multiline and quote-heavy native
        # arguments differently from PowerShell 7.  Keep the -c bootstrap free
        # of quotes and carry the real probe as an opaque Base64 argv value so
        # both hosts execute exactly the same Python source.
        $probeSource = 'import json,sys;print(json.dumps({"executable":sys.executable,"version":list(sys.version_info[:3])}))'
        $probePayload = [Convert]::ToBase64String(
            [Text.Encoding]::UTF8.GetBytes($probeSource)
        )
        $probeBootstrap = 'import base64,sys;exec(base64.b64decode(sys.argv[1]))'
        $probeOutput = @(
            & $Command @PrefixArguments -I -c $probeBootstrap $probePayload 2>$null
        )
        if ($LASTEXITCODE -ne 0 -or $probeOutput.Count -ne 1) {
            return $null
        }
        $probe = $probeOutput[0] | ConvertFrom-Json -ErrorAction Stop
        if ($probe.version.Count -ne 3 -or $probe.version[0] -ne 3 -or $probe.version[1] -ne 12) {
            return $null
        }
        $executable = [System.IO.Path]::GetFullPath([string]$probe.executable)
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
            return $null
        }
        return [pscustomobject]@{
            FilePath = $executable
            BootstrapPython = $executable
            Version = ($probe.version -join ".")
        }
    }
    catch {
        return $null
    }
}

function Resolve-SocialGraphPythonLauncher {
    [CmdletBinding()]
    param([string]$BootstrapPython)

    if (-not [string]::IsNullOrWhiteSpace($BootstrapPython)) {
        $explicit = Get-Command $BootstrapPython -ErrorAction SilentlyContinue
        if ($null -eq $explicit) {
            throw "Bootstrap Python does not exist: $BootstrapPython"
        }
        $prefix = if ([System.IO.Path]::GetFileNameWithoutExtension($explicit.Source) -eq "py") {
            @("-3.12")
        }
        else {
            @()
        }
        $selected = Test-SocialGraphPythonLauncher -Command $explicit.Source -PrefixArguments $prefix
        if ($null -eq $selected) {
            throw "Bootstrap Python must be CPython 3.12: $BootstrapPython"
        }
        return $selected
    }

    $candidates = @()
    if ($env:OS -eq "Windows_NT") {
        $candidates += [pscustomobject]@{ Name = "py"; Prefix = @("-3.12") }
    }
    $candidates += [pscustomobject]@{ Name = "python"; Prefix = @() }
    $candidates += [pscustomobject]@{ Name = "python3"; Prefix = @() }
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Name -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            continue
        }
        $selected = Test-SocialGraphPythonLauncher `
            -Command $command.Source `
            -PrefixArguments $candidate.Prefix
        if ($null -ne $selected) {
            return $selected
        }
    }
    throw "CPython 3.12 is required. Tried: py -3.12, python, python3."
}
