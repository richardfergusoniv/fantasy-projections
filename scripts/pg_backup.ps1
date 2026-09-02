# PostgreSQL backup and restore for the fantasy decision app.
# Works with Docker Compose (`db` service) or a local PostgreSQL install.
param(
    [ValidateSet("backup", "restore")]
    [string]$Action = "backup",
    [string]$OutputPath = "",
    [string]$InputPath = "",
    [string]$DbHost = "localhost",
    [int]$Port = 5432,
    [string]$DbUser = "fantasy",
    [string]$DbName = "fantasy_app",
    [string]$PgBin = "",
    [switch]$UseDocker
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$defaultBackup = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "output/backups/fantasy_app-$timestamp.sql"

if (-not $OutputPath) { $OutputPath = $defaultBackup }
if (-not $InputPath -and $Action -eq "restore") {
    throw "Restore requires -InputPath"
}

function Invoke-PgDump {
    param([string]$Dest)
    $parent = Split-Path $Dest -Parent
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    if ($UseDocker) {
        docker compose exec -T db pg_dump -U $DbUser $DbName | Set-Content -Path $Dest -Encoding utf8
    } else {
        $pgDump = if ($PgBin) { Join-Path $PgBin "pg_dump.exe" } else { "pg_dump" }
        & $pgDump -h $DbHost -p $Port -U $DbUser -d $DbName -f $Dest
    }
}

function Invoke-PgRestore {
    param([string]$Source)
    if (-not (Test-Path $Source)) { throw "Backup file not found: $Source" }
    if ($UseDocker) {
        Get-Content $Source | docker compose exec -T db psql -U $DbUser $DbName
    } else {
        $psql = if ($PgBin) { Join-Path $PgBin "psql.exe" } else { "psql" }
        & $psql -h $DbHost -p $Port -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $Source
    }
}

switch ($Action) {
    "backup" {
        Invoke-PgDump -Dest $OutputPath
        $size = (Get-Item $OutputPath).Length
        Write-Host "OK   backup written to $OutputPath ($size bytes)"
    }
    "restore" {
        Invoke-PgRestore -Source $InputPath
        Write-Host "OK   restored from $InputPath"
    }
}
