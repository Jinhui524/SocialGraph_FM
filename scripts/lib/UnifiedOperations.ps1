Set-StrictMode -Version Latest

# Thin compatibility facade retained for established scripts and test fixtures.
# New code should dot-source only the focused module it needs.
. (Join-Path $PSScriptRoot "LegacyOperations.ps1")
