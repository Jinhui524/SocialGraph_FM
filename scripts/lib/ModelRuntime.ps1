Set-StrictMode -Version Latest

$script:UnifiedUvVersion = "0.12.3"
$script:UnifiedUvArchiveHashes = @{
    Windows = "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"
    Linux = "600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"
}

function Assert-PinnedUv {
    param([Parameter(Mandatory = $true)][string]$Executable)

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    $version = ((& $Executable --version 2>$null) | Out-String).Trim()
    return $LASTEXITCODE -eq 0 -and $version -match '^uv 0\.12\.3(?:\s|$)'
}

function Install-LocalUv {
    param([Parameter(Mandatory = $true)]$Layout)

    if (Assert-PinnedUv -Executable $Layout.UvExecutable) {
        return $Layout.UvExecutable
    }
    if (Assert-PinnedUv -Executable $Layout.LegacyGfmUv) {
        New-Item -ItemType Directory -Force -Path $Layout.UvRoot | Out-Null
        Copy-Item -LiteralPath $Layout.LegacyGfmUv -Destination $Layout.UvExecutable -Force
        if (Assert-PinnedUv -Executable $Layout.UvExecutable) {
            return $Layout.UvExecutable
        }
    }

    New-Item -ItemType Directory -Force -Path $Layout.UvRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $Layout.TempRoot | Out-Null
    $downloadRoot = Join-Path $Layout.TempRoot "uv-bootstrap-$PID-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    try {
        if ($env:OS -eq "Windows_NT") {
            $archive = Join-Path $downloadRoot "uv.zip"
            $url = "https://github.com/astral-sh/uv/releases/download/$script:UnifiedUvVersion/uv-x86_64-pc-windows-msvc.zip"
            Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
            $actualArchiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualArchiveHash -ne $script:UnifiedUvArchiveHashes.Windows) {
                throw "The downloaded uv archive failed SHA-256 verification."
            }
            Expand-Archive -LiteralPath $archive -DestinationPath $downloadRoot -Force
            $candidate = Get-ChildItem -LiteralPath $downloadRoot -Recurse -Filter "uv.exe" -File |
                Select-Object -First 1
        }
        else {
            $archive = Join-Path $downloadRoot "uv.tar.gz"
            $url = "https://github.com/astral-sh/uv/releases/download/$script:UnifiedUvVersion/uv-x86_64-unknown-linux-gnu.tar.gz"
            Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
            $actualArchiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualArchiveHash -ne $script:UnifiedUvArchiveHashes.Linux) {
                throw "The downloaded uv archive failed SHA-256 verification."
            }
            & tar -xzf $archive -C $downloadRoot
            if ($LASTEXITCODE -ne 0) { throw "Could not extract the pinned uv archive." }
            $candidate = Get-ChildItem -LiteralPath $downloadRoot -Recurse -Filter "uv" -File |
                Select-Object -First 1
        }
        if ($null -eq $candidate) { throw "The pinned uv archive did not contain uv." }
        Copy-Item -LiteralPath $candidate.FullName -Destination $Layout.UvExecutable -Force
        if (-not (Assert-PinnedUv -Executable $Layout.UvExecutable)) {
            throw "The downloaded uv executable did not report version $script:UnifiedUvVersion."
        }
        return $Layout.UvExecutable
    }
    finally {
        Remove-Item -LiteralPath $downloadRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-UvChecked {
    param(
        [Parameter(Mandatory = $true)][string]$UvExecutable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )

    & $UvExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)."
    }
}

function Initialize-ApiEnvironment {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$UvExecutable
    )

    $apiVenvArguments = @(
        "venv", "--python", $Layout.ManagedPythonVersion, "--managed-python", "--seed"
    )
    if (Test-Path -LiteralPath $Layout.ApiPython -PathType Leaf) {
        $apiVenvArguments += "--allow-existing"
    }
    $apiVenvArguments += $Layout.ApiEnvironmentRoot
    Invoke-UvChecked -UvExecutable $UvExecutable `
        -Arguments $apiVenvArguments `
        -FailureMessage "Could not create or seed the isolated API environment"
    Invoke-UvChecked -UvExecutable $UvExecutable `
        -Arguments @(
            "pip", "sync", (Join-Path $Layout.Api "requirements.lock"),
            "--python", $Layout.ApiPython, "--require-hashes", "--no-build",
            "--index-url", "https://pypi.org/simple"
        ) `
        -FailureMessage "API dependency installation failed"
    Assert-ApiIsTorchFree -PythonExecutable $Layout.ApiPython
}

function Initialize-GfmEnvironment {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$UvExecutable,
        [ValidateSet("Cpu", "Cuda")][string]$Profile,
        [switch]$GfmTextProfile
    )

    $gfmVenvArguments = @(
        "venv", "--python", $Layout.ManagedPythonVersion, "--managed-python", "--seed"
    )
    if (Test-Path -LiteralPath $Layout.GfmPython -PathType Leaf) {
        $gfmVenvArguments += "--allow-existing"
    }
    $gfmVenvArguments += $Layout.GfmEnvironmentRoot
    Invoke-UvChecked -UvExecutable $UvExecutable `
        -Arguments $gfmVenvArguments `
        -FailureMessage "Could not create or seed the isolated GFM environment"
    if ($Profile -eq "Cpu") {
        if ($GfmTextProfile) {
            throw "The frozen text-embedding dependency profile is currently available only for CUDA."
        }
        $cpuLockName = if ($env:OS -eq "Windows_NT") {
            "windows-cpu.requirements.txt"
        }
        else {
            "cpu-ci.requirements.txt"
        }
        $lockPath = Join-Path $Layout.GfmPackage "locks\$cpuLockName"
        $arguments = @(
            "pip", "sync", $lockPath, "--python", $Layout.GfmPython,
            "--require-hashes", "--no-build", "--torch-backend", "cpu",
            "--index-url", "https://pypi.org/simple", "--find-links",
            "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
        )
    }
    else {
        $lockName = if ($GfmTextProfile) {
            "windows-cu130-gfm.requirements.txt"
        }
        else {
            "windows-cu130.requirements.txt"
        }
        $lockPath = Join-Path $Layout.GfmPackage "locks\$lockName"
        $arguments = @(
            "pip", "sync", $lockPath, "--python", $Layout.GfmPython,
            "--require-hashes", "--no-build", "--torch-backend", "cu130",
            "--index-url", "https://pypi.org/simple", "--find-links",
            "https://data.pyg.org/whl/torch-2.12.0+cu130.html"
        )
    }
    Invoke-UvChecked -UvExecutable $UvExecutable -Arguments $arguments `
        -FailureMessage "GFM hash-locked dependency installation failed"
    Invoke-UvChecked -UvExecutable $UvExecutable `
        -Arguments @(
            "pip", "install", "--python", $Layout.GfmPython,
            "--no-deps", "--editable", $Layout.GfmPackage
        ) `
        -FailureMessage "GFM editable package installation failed"
}

function Write-UnifiedRuntimeProfile {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [ValidateSet("Offline", "Cpu", "Cuda")][string]$Profile
    )

    New-Item -ItemType Directory -Force -Path $Layout.ConfigRoot | Out-Null
    Protect-UnifiedConfigDirectory -Path $Layout.ConfigRoot
    $payload = [ordered]@{
        schemaVersion = "socialgraph-fm.runtime-profile/1.0"
        profile = $Profile
        updatedAtUtc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json
    $temporaryPath = Join-Path $Layout.ConfigRoot ".runtime-profile.$PID.$([Guid]::NewGuid().ToString('N')).tmp"
    $backupPath = Join-Path $Layout.ConfigRoot ".runtime-profile.$PID.$([Guid]::NewGuid().ToString('N')).bak"
    try {
        [IO.File]::WriteAllText($temporaryPath, "$payload`n", [Text.UTF8Encoding]::new($false))
        $verified = Get-Content -LiteralPath $temporaryPath -Raw | ConvertFrom-Json -ErrorAction Stop
        if ($verified.schemaVersion -ne "socialgraph-fm.runtime-profile/1.0" -or
            $verified.profile -ne $Profile) {
            throw "The staged runtime profile failed validation."
        }
        if (Test-Path -LiteralPath $Layout.RuntimeProfileFile -PathType Leaf) {
            [IO.File]::Replace($temporaryPath, $Layout.RuntimeProfileFile, $backupPath, $true)
        }
        else {
            [IO.File]::Move($temporaryPath, $Layout.RuntimeProfileFile)
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $backupPath -ErrorAction SilentlyContinue
    }
}

function Install-AuthorizedModelPackage {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )

    if ($Layout.RuntimeProfile -eq "Offline" -or
        -not (Test-Path -LiteralPath $Layout.GfmPython -PathType Leaf)) {
        throw "Install a CPU or CUDA profile before installing an authorized model package."
    }
    $source = Get-Item -LiteralPath ([IO.Path]::GetFullPath($PackageRoot)) -Force
    if (-not $source.PSIsContainer -or
        ($source.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The authorized model package must be a local, non-reparse directory."
    }
    $unsafe = Get-ChildItem -LiteralPath $source.FullName -Recurse -Force | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    } | Select-Object -First 1
    if ($null -ne $unsafe) {
        throw "The authorized model package cannot contain links or reparse points: $($unsafe.FullName)"
    }
    $required = @(
        "exports\socialgraph-global\export-manifest.json",
        "exports\socialgraph-global\smoke-report.json",
        "exports\socialgraph-global\registry.json",
        "registry\socialgraph-global.json"
    )
    foreach ($relative in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $source.FullName $relative) -PathType Leaf)) {
            throw "The authorized model package is missing $relative."
        }
    }

    $modelParent = Split-Path -Parent $Layout.GlobalModelRoot
    New-Item -ItemType Directory -Force -Path $modelParent | Out-Null
    $staging = Join-Path $modelParent ".socialgraph-global.$PID.$([Guid]::NewGuid().ToString('N')).stage"
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        foreach ($entry in Get-ChildItem -LiteralPath $source.FullName -Force) {
            Copy-Item -LiteralPath $entry.FullName -Destination $staging -Recurse
        }
        Push-Location $Layout.GfmPackage
        try {
            & $Layout.GfmPython -m socialgraph_gfm.global_model.cli `
                _verify-export --root $staging
            if ($LASTEXITCODE -ne 0) {
                throw "The authorized SocialGraph-FM Global export failed manifest and artifact hash verification."
            }
            & $Layout.GfmPython -m socialgraph_gfm.global_model.cli publish --root $staging
            if ($LASTEXITCODE -ne 0) {
                throw "The authorized SocialGraph-FM Global registry or smoke result failed verification."
            }
        }
        finally {
            Pop-Location
        }
        if (Test-Path -LiteralPath $Layout.GlobalModelRoot) {
            $existingRegistry = Join-Path $Layout.GlobalModelRoot "registry\socialgraph-global.json"
            $stagedRegistry = Join-Path $staging "registry\socialgraph-global.json"
            if ((Test-Path -LiteralPath $existingRegistry -PathType Leaf) -and
                (Get-UnifiedFileSha256 -Path $existingRegistry) -eq
                    (Get-UnifiedFileSha256 -Path $stagedRegistry)) {
                Write-Host "The same authorized SocialGraph-FM Global model package is already installed."
                return $Layout.GlobalModelRoot
            }
            throw "A different SocialGraph-FM Global model package is already installed. Remove it through an explicit maintenance workflow before replacing it."
        }
        Move-Item -LiteralPath $staging -Destination $Layout.GlobalModelRoot
        Write-Host "Authorized SocialGraph-FM Global model package installed at $($Layout.GlobalModelRoot)."
        return $Layout.GlobalModelRoot
    }
    finally {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}
