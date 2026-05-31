<#
.SYNOPSIS
    Fix Windows WDDM TDR (Timeout Detection and Recovery) for long GPU training runs.

.DESCRIPTION
    Raises the GPU timeout from the default 2 seconds to 60 seconds and relaxes the
    TDR recovery level so the driver does not abort QLoRA training jobs that hold
    the GPU for extended periods on a 6 GB card.

    Registry values written under HKLM\System\CurrentControlSet\Control\GraphicsDrivers:
      TdrLevel    = 3   (recover process / desktop, do not bug-check)
      TdrDelay    = 60  (seconds — total timeout)
      TdrDdiDelay = 60  (seconds — DDI call timeout)

    Reference: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys

.NOTES
    Must run as Administrator.
    Restart is REQUIRED for changes to take effect.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Write-Section([string]$Title) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan
}

# ── Admin guard ──────────────────────────────────────────────────────────────
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal   = New-Object Security.Principal.WindowsPrincipal($currentUser)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'ERROR: This script must be run as Administrator.' -ForegroundColor Red
    Write-Host 'Right-click PowerShell -> "Run as Administrator", then re-run this script.' -ForegroundColor Yellow
    exit 1
}

Write-Section 'WDDM TDR Fix — QLoRA Training Stability'

$regPath = 'HKLM:\System\CurrentControlSet\Control\GraphicsDrivers'
$values = @(
    @{ Name = 'TdrLevel';    Value = 3;  Description = 'Recover desktop without bug-check' },
    @{ Name = 'TdrDelay';    Value = 60; Description = 'Total GPU timeout (seconds)' },
    @{ Name = 'TdrDdiDelay'; Value = 60; Description = 'DDI call timeout (seconds)' }
)

if (-not (Test-Path $regPath)) {
    Write-Host "ERROR: Registry path not found: $regPath" -ForegroundColor Red
    exit 1
}

foreach ($entry in $values) {
    try {
        New-ItemProperty -Path $regPath `
                         -Name $entry.Name `
                         -Value $entry.Value `
                         -PropertyType DWord `
                         -Force | Out-Null
        Write-Host ("  [OK] {0,-12} = {1,-3} ({2})" -f $entry.Name, $entry.Value, $entry.Description) `
                   -ForegroundColor Green
    } catch {
        Write-Host ("  [FAIL] {0}: {1}" -f $entry.Name, $_.Exception.Message) -ForegroundColor Red
        exit 1
    }
}

# ── Verify by reading back ───────────────────────────────────────────────────
Write-Host ''
Write-Host 'Verifying written values…' -ForegroundColor Cyan
foreach ($entry in $values) {
    $actual = (Get-ItemProperty -Path $regPath -Name $entry.Name -ErrorAction SilentlyContinue).$($entry.Name)
    if ($actual -eq $entry.Value) {
        Write-Host ("  [VERIFIED] {0} = {1}" -f $entry.Name, $actual) -ForegroundColor Green
    } else {
        Write-Host ("  [MISMATCH] {0} expected {1}, got {2}" -f $entry.Name, $entry.Value, $actual) -ForegroundColor Yellow
    }
}

# ── Warning + reboot prompt ──────────────────────────────────────────────────
Write-Host ''
Write-Host 'RESTART REQUIRED: Changes take effect after reboot.' -ForegroundColor Yellow
Write-Host 'Without this fix, GPU driver may timeout during training' -ForegroundColor Yellow
Write-Host '(training will crash after ~2 min).' -ForegroundColor Yellow
Write-Host ''

$answer = Read-Host 'Restart now? (y/n)'
if ($answer -match '^[Yy]') {
    Write-Host 'Restarting in 5 seconds…' -ForegroundColor Cyan
    Start-Sleep -Seconds 5
    Restart-Computer -Force
} else {
    Write-Host 'Reboot skipped. Please restart manually before running training.' -ForegroundColor Yellow
}
