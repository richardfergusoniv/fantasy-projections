# Build and serve the production PWA (nginx when Docker is available, else vite preview).
param(
    [int]$Port = 5173,
    [int]$ApiPort = 8000,
    [string]$Root = ""
)

$ErrorActionPreference = "Stop"
if (-not $Root) {
    $Root = Resolve-Path (Join-Path $PSScriptRoot "..")
}
$webDir = Join-Path $Root "web"
$distDir = Join-Path $webDir "dist"

function Find-Npm {
    $cmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = "${env:ProgramFiles}\nodejs\npm.cmd"
    if (Test-Path $default) { return $default }
    throw "npm not found; install Node.js"
}

$npm = Find-Npm
$nodeDir = Split-Path $npm -Parent
if ($env:Path -notlike "*$nodeDir*") {
    $env:Path = "$nodeDir;$env:Path"
}

Write-Host "Building production web bundle..."
Push-Location $webDir
if (-not (Test-Path (Join-Path $webDir "node_modules\vite"))) {
    & $npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
}
& $npm run build
if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
Pop-Location

if (-not (Test-Path (Join-Path $distDir "index.html"))) {
    throw "Production build missing: $distDir\index.html"
}

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    $nginxTemplate = Join-Path $Root "docker\nginx.phone_access.conf"
    $nginxRuntime = Join-Path $Root "output\live_pg\nginx.phone_access.runtime.conf"
    $nginxDir = Split-Path $nginxRuntime -Parent
    if (-not (Test-Path $nginxDir)) {
        New-Item -ItemType Directory -Path $nginxDir -Force | Out-Null
    }
    (Get-Content $nginxTemplate -Raw).Replace("__API_PORT__", "$ApiPort") |
        Set-Content -Path $nginxRuntime -Encoding utf8

    $containerName = "fantasy-phone-web"
    docker rm -f $containerName 2>$null | Out-Null
    docker run -d --name $containerName `
        -p "${Port}:80" `
        -v "${distDir}:/usr/share/nginx/html:ro" `
        -v "${nginxRuntime}:/etc/nginx/conf.d/default.conf:ro" `
        --add-host=host.docker.internal:host-gateway `
        nginx:1.27-alpine | Out-Null
    Write-Host "OK   nginx container $containerName on :$Port (production PWA)"
    return @{
        mode = "nginx"
        port = $Port
        container = $containerName
    }
}

Write-Host "Docker not found; serving production build via vite preview on :$Port"
try {
    $existing = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -TimeoutSec 3 -UseBasicParsing
    if ($existing.StatusCode -eq 200) {
        Write-Host "OK   vite preview already serving on :$Port (production PWA)"
        return @{
            mode = "vite_preview"
            port = $Port
            pid = 0
        }
    }
} catch {
    # Not running yet — start below.
}

$preview = Start-Process -FilePath "cmd.exe" -ArgumentList @(
    "/c", "set API_PROXY_PORT=$ApiPort&& `"$npm`" run preview -- --host 127.0.0.1 --port $Port"
) -PassThru -WorkingDirectory $webDir -WindowStyle Hidden

for ($i = 0; $i -lt 90; $i++) {
    try {
        $probe = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -TimeoutSec 3 -UseBasicParsing
        if ($probe.StatusCode -eq 200) {
            Write-Host "OK   vite preview pid $($preview.Id) on :$Port (production PWA)"
            return @{
                mode = "vite_preview"
                port = $Port
                pid = $preview.Id
            }
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}
throw "Production web server did not start on :$Port within 180s"
