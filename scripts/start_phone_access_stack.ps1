# Start API + web + Cloudflare quick tunnel for phone access.
param(
    [string]$AllowedEmail = "",
    [string]$ResendApiKey = "",
    [switch]$SkipBootstrap,
    [switch]$SkipGate,
    [switch]$NonInteractive,
    [switch]$RegisterScheduledTasks
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

function Import-DotEnv {
    param([string]$Path)
    Get-Content $Path | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Stop-IfRunning {
    param([int]$Pid)
    if ($Pid -gt 0) {
        Stop-Process -Id $Pid -Force -ErrorAction SilentlyContinue
    }
}

function Stop-PortListeners {
    param([int]$Port)
    $attempts = 0
    while ($attempts -lt 5) {
        $attempts++
        $owners = @(
            Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique
        )
        if (-not $owners -or $owners.Count -eq 0) {
            return
        }
        foreach ($owner in $owners) {
            if ($owner -gt 0) {
                Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Seconds 1
    }
}

if (-not $AllowedEmail) {
    $AllowedEmail = $env:APP_ALLOWED_EMAIL
}
if (-not $ResendApiKey) {
    $ResendApiKey = $env:RESEND_API_KEY
}

$secretsPath = Join-Path $root "config/phone_access.secrets.json"
if ((-not $AllowedEmail -or -not $ResendApiKey) -and (Test-Path $secretsPath)) {
    $secrets = Get-Content $secretsPath | ConvertFrom-Json
    if (-not $AllowedEmail) { $AllowedEmail = $secrets.allowed_email }
    if (-not $ResendApiKey) { $ResendApiKey = $secrets.resend_api_key }
}

function Find-Npm {
    $cmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = "${env:ProgramFiles}\nodejs\npm.cmd"
    if (Test-Path $default) { return $default }
    throw "npm not found; install Node.js"
}

function Ensure-NodePath {
    $nodeDir = "${env:ProgramFiles}\nodejs"
    if (Test-Path (Join-Path $nodeDir "node.exe")) {
        if ($env:Path -notlike "*$nodeDir*") {
            $env:Path = "$nodeDir;$env:Path"
        }
    } elseif (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "node not found; install Node.js or add it to PATH"
    }
}

$npm = Find-Npm
Ensure-NodePath

Stop-PortListeners -Port 8000
Stop-PortListeners -Port 5173

$apiPort = 8000
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    $apiPort = 8002
    Write-Host "WARN port 8000 still in use; using API on :$apiPort"
}

$api = $null
$webDir = Join-Path $root "web"
$webPort = 5173
$webInfo = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\serve_production_web.ps1") -Port $webPort -ApiPort $apiPort -Root $root
if ($webInfo.mode -eq "vite_preview") {
    $web = Get-Process -Id $webInfo.pid
} else {
    $web = [pscustomobject]@{ Id = 0; mode = $webInfo.mode; container = $webInfo.container }
}
Write-Host "OK   Production web ($($webInfo.mode)) on :$webPort"

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\cloudflare_tunnel.ps1") -WebPort $webPort | Out-Null
$tunnelPath = Join-Path $root "output/live_pg/cloudflare_tunnel.json"
if (-not (Test-Path $tunnelPath)) { throw "Tunnel report missing: $tunnelPath" }
$tunnel = Get-Content $tunnelPath | ConvertFrom-Json
$publicUrl = $tunnel.public_url
$publicHost = $tunnel.public_host

if (-not $SkipBootstrap) {
    if (-not $AllowedEmail -or -not $ResendApiKey) {
        $msg = @"
Missing phone-access credentials.
Create config/phone_access.secrets.json from config/phone_access.secrets.example.json
or pass -AllowedEmail and -ResendApiKey.
"@
        if ($NonInteractive) { throw $msg }
    }
    if (-not $AllowedEmail) {
        $AllowedEmail = Read-Host "Allowed email for magic-link sign-in"
    }
    if (-not $ResendApiKey) {
        $ResendApiKey = Read-Host "Resend API key (re_...)"
    }
    $bootstrapArgs = @(
        "-AllowedEmail", $AllowedEmail,
        "-PublicUrl", $publicUrl,
        "-PublicHost", $publicHost,
        "-ResendApiKey", $ResendApiKey,
        "-TunnelMode"
    )
    if ($RegisterScheduledTasks) { $bootstrapArgs += "-RegisterScheduledTasks" }
    # Infra validation needs a live API; run it after the production server starts.
    $bootstrapArgs += "-SkipValidation"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\bootstrap_phone_access.ps1") @bootstrapArgs
}

Stop-PortListeners -Port 8000
Import-DotEnv (Join-Path $root ".env")

function Start-ProductionApi {
    param([int]$Port)
    return Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList @(
        "-m", "src.app.cli", "api", "--host", "127.0.0.1", "--port", "$Port"
    ) -PassThru -WorkingDirectory $root -WindowStyle Hidden
}

$api = Start-ProductionApi -Port $apiPort
Start-Sleep -Seconds 4
Write-Host "OK   API (production) pid $($api.Id) on :$apiPort"
if ($apiPort -ne 8000 -and $webInfo.mode -eq "vite_preview") {
    Stop-IfRunning $web.Id
    $webInfo = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\serve_production_web.ps1") -Port $webPort -ApiPort $apiPort -Root $root
    $web = Get-Process -Id $webInfo.pid
    Start-Sleep -Seconds 5
}

$stackReport = [ordered]@{
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    public_url = $publicUrl
    public_host = $publicHost
    api_pid = $api.Id
    api_port = $apiPort
    web_mode = $webInfo.mode
    web_pid = if ($webInfo.mode -eq "vite_preview") { $web.Id } else { $null }
    web_container = if ($webInfo.mode -eq "nginx") { $webInfo.container } else { $null }
    web_port = $webPort
    tunnel_pid = $tunnel.pid
    bootstrap = -not $SkipBootstrap
}
$stackPath = Join-Path $root "output/live_pg/phone_access_stack.json"
$stackReport | ConvertTo-Json | Set-Content -Path $stackPath -Encoding utf8

if (-not $SkipGate -and (Test-Path (Join-Path $root ".env"))) {
    & .venv\Scripts\python.exe scripts/production_infrastructure_check.py `
        --env-file .env `
        --api-base-url $publicUrl `
        --database-url $env:DATABASE_URL `
        --report output/live_pg/production_infrastructure_audit.json
    & .venv\Scripts\python.exe scripts/live_go_live_gate.py --api-base-url $publicUrl
}

Write-Host @"

Phone access stack running:
  Public URL:  $publicUrl
  API pid:     $($api.Id)
  Web mode:    $($webInfo.mode) on :$webPort
  Tunnel pid:  $($tunnel.pid)
  Report:      output/live_pg/phone_access_stack.json

Open $publicUrl on your phone and sign in with $AllowedEmail

Stop:
  Stop-Process -Id $($api.Id),$($tunnel.pid) -Force -ErrorAction SilentlyContinue
  if ($webInfo.mode -eq "vite_preview") { Stop-Process -Id $($web.Id) -Force -ErrorAction SilentlyContinue }
  if ($webInfo.mode -eq "nginx") { docker rm -f $($webInfo.container) 2>`$null }
"@
