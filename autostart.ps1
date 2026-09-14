<#
.SYNOPSIS
    Autostart control for Govorun PC.

.DESCRIPTION
    Creates a shortcut to govorun.vbs in the current user Startup folder.
    No administrator rights needed: user-level autostart, not registry,
    not a service.

.EXAMPLE
    .\autostart.ps1
    .\autostart.ps1 -Enable
    .\autostart.ps1 -Disable
    .\autostart.ps1 -Log
#>

[CmdletBinding()]
param(
    [switch]$Enable,
    [switch]$Disable,
    [switch]$Log
)

$ErrorActionPreference = 'Stop'

$here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$target   = Join-Path $here 'govorun.vbs'
$logFile  = Join-Path $here 'govorun.log'
$startup  = [Environment]::GetFolderPath('Startup')
$shortcut = Join-Path $startup 'Govorun PC.lnk'

function Show-Status {
    Write-Host ''
    if (Test-Path $shortcut) {
        Write-Host 'Autostart: ON' -ForegroundColor Green
        $ws  = New-Object -ComObject WScript.Shell
        $lnk = $ws.CreateShortcut($shortcut)
        Write-Host ('  shortcut: ' + $shortcut)
        Write-Host ('  points to: ' + $lnk.TargetPath)
        if ($lnk.TargetPath -ne $target) {
            Write-Host '  WARNING: shortcut points somewhere else.' -ForegroundColor Yellow
            Write-Host '  Looks like the folder moved. Re-run with -Enable' -ForegroundColor Yellow
        }
    }
    else {
        Write-Host 'Autostart: OFF' -ForegroundColor Yellow
        Write-Host '  turn on: .\autostart.ps1 -Enable'
    }

    if (Test-Path $logFile) {
        $age = (Get-Date) - (Get-Item $logFile).LastWriteTime
        $min = [int]$age.TotalMinutes
        Write-Host ''
        Write-Host ('Last run: ' + $min + ' min ago')
        Write-Host ('  log: ' + $logFile)
    }
    Write-Host ''
}

if ($Enable -and $Disable) {
    throw 'Pick one: -Enable or -Disable'
}

if ($Log) {
    if (-not (Test-Path $logFile)) {
        Write-Host 'No log yet: the program has not been started silently.' -ForegroundColor Yellow
        Write-Host 'Silent start: govorun.vbs   Visible window: govorun.bat'
        exit 0
    }
    # Read as UTF-8 explicitly: PowerShell 5.1 defaults to the system
    # codepage and turns Russian into mojibake
    Get-Content $logFile -Encoding UTF8
    exit 0
}

if ($Disable) {
    if (Test-Path $shortcut) {
        Remove-Item $shortcut -Force
        Write-Host 'Autostart disabled.' -ForegroundColor Green
        Write-Host 'The running program keeps running.'
    }
    else {
        Write-Host 'Autostart was already off.'
    }
    exit 0
}

if ($Enable) {
    if (-not (Test-Path $target)) {
        throw ('govorun.vbs not found next to the script: ' + $target)
    }

    $ws  = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($shortcut)
    $lnk.TargetPath       = $target
    $lnk.WorkingDirectory = $here
    $lnk.Description      = 'Offline Russian voice input'
    $lnk.Save()

    Write-Host 'Autostart enabled.' -ForegroundColor Green
    Write-Host ('  shortcut: ' + $shortcut)
    Write-Host ''
    Write-Host 'It will start at your next sign-in. No window, tray icon only.'
    Write-Host 'First run after switching models takes longer: weights download.'
    Write-Host ''
    Write-Host 'No tray icon? Check govorun.log in this folder.'
    exit 0
}

Show-Status