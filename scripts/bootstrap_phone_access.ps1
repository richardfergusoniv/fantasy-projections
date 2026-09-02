# Bootstrap production .env for phone access and validate infrastructure.
param(
    [Parameter(Mandatory = $true)]
    [string]$AllowedEmail,
    [Parameter(Mandatory = $true)]
    [string]$PublicUrl,
    [string]$PublicHost = "",
    [ValidateSet("resend", "smtp")]
    [string]$EmailProvider = "resend",
    [string]$ResendApiKey = "",
    [string]$SmtpHost = "",
    [string]$SmtpUser = "",
    [string]$SmtpPassword = "",
    [string]$EmailFrom = "",
    [string]$DbPassword = "fantasy",
    [string]$EnvPath = ".env",
    [switch]$TunnelMode,
    [switch]$RegisterScheduledTasks,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not $PublicHost) {
    $uri = [Uri]$PublicUrl
    $PublicHost = $uri.Host
}
if ($TunnelMode) {
    $trusted = "$PublicHost,127.0.0.1,localhost"
} else {
    $trusted = $PublicHost
}
if (-not $EmailFrom) {
    if ($TunnelMode) {
        $EmailFrom = "onboarding@resend.dev"
    } else {
        $EmailFrom = "noreply@$PublicHost"
    }
}

$secret = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object { [char]$_ })

$lines = @(
    "APP_ENV=production",
    "APP_SECRET_KEY=$secret",
    "APP_ALLOWED_EMAIL=$AllowedEmail",
    "APP_ENABLE_DEV_AUTH=false",
    "APP_PUBLIC_URL=$PublicUrl",
    "APP_CORS_ORIGINS=$PublicUrl",
    "TRUSTED_HOSTS=$trusted",
    "DATABASE_URL=postgresql+psycopg://fantasy:${DbPassword}@localhost:5432/fantasy_app",
    "POSTGRES_PASSWORD=$DbPassword",
    "ARTIFACT_BACKEND=local",
    "ARTIFACT_LOCAL_ROOT=output/app_artifacts",
    "SLEEPER_USE_FIXTURES=false",
    "SLEEPER_OWNER_CONFIG=config/sleeper_owner.json",
    "INJURY_RESEARCH_MODE=sleeper",
    "EMAIL_PROVIDER=$EmailProvider",
    "EMAIL_FROM=$EmailFrom",
    "APP_PROJECTION_SOURCE=sealed_release",
    "WEEKLY_RND_ENABLED=false",
    "STATUS_OVERLAY_AUTO_PUBLISH=true",
    "LOG_LEVEL=INFO",
    "LOG_JSON=true"
)

if ($EmailProvider -eq "resend") {
    if (-not $ResendApiKey) { throw "ResendApiKey is required when EmailProvider=resend" }
    $lines += "RESEND_API_KEY=$ResendApiKey"
} else {
    if (-not $SmtpHost -or -not $SmtpUser -or -not $SmtpPassword) {
        throw "SmtpHost, SmtpUser, and SmtpPassword are required when EmailProvider=smtp"
    }
    $lines += @(
        "SMTP_HOST=$SmtpHost",
        "SMTP_PORT=587",
        "SMTP_USER=$SmtpUser",
        "SMTP_PASSWORD=$SmtpPassword"
    )
}

$lines | Set-Content -Path $EnvPath -Encoding utf8
Write-Host "OK   wrote $EnvPath"

if ($RegisterScheduledTasks) {
    $backupScript = Join-Path $root "scripts\pg_backup.ps1"
    $monitorScript = Join-Path $root "scripts\monitor_health.ps1"
    $pgBin = "C:\Program Files\PostgreSQL\16\bin"
    $backupArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$backupScript`" -PgBin `"$pgBin`""
    $monitorArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$monitorScript`" -BaseUrl $PublicUrl -ReportPath output/monitoring/health.json"

    $backupAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $backupArgs -WorkingDirectory $root
    $backupTrigger = New-ScheduledTaskTrigger -Daily -At 3:00AM
    Register-ScheduledTask -TaskName "FantasyApp-PgBackup" -Action $backupAction -Trigger $backupTrigger -Force | Out-Null
    Write-Host "OK   scheduled task FantasyApp-PgBackup (daily 3:00 AM)"

    $monitorAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $monitorArgs -WorkingDirectory $root
    $monitorTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)
    Register-ScheduledTask -TaskName "FantasyApp-HealthMonitor" -Action $monitorAction -Trigger $monitorTrigger -Force | Out-Null
    Write-Host "OK   scheduled task FantasyApp-HealthMonitor (every 5 min)"
}

if (-not $SkipValidation) {
    $py = Join-Path $root ".venv\Scripts\python.exe"
    & $py scripts/production_infrastructure_check.py --env-file $EnvPath --api-base-url $PublicUrl --database-url "postgresql+psycopg://fantasy:${DbPassword}@localhost:5432/fantasy_app" --report output/live_pg/production_infrastructure_audit.json
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN infrastructure check reported blockers (see audit JSON)"
        exit 1
    }
    Write-Host "OK   production infrastructure check passed"
}

Write-Host @"

Next steps:
  1. Build and serve web: cd web; npm ci; npm run build
  2. Terminate TLS at nginx (see docker/nginx.tls.conf.example) or your host
  3. Start API with production env loaded:
       Get-Content .env | ForEach-Object { if (`$_ -match '^([^#=]+)=(.*)$') { Set-Item -Path env:`$matches[1] -Value `$matches[2] } }
       .venv\Scripts\python.exe -m src.app.cli api --host 0.0.0.0 --port 8000
  4. Re-run go-live gate:
       .venv\Scripts\python.exe scripts/live_go_live_gate.py --api-base-url $PublicUrl
"@
