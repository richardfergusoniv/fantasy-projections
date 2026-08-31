Add-Type -AssemblyName System.Drawing

function Write-PwaIcon {
    param(
        [int]$Size,
        [string]$Path
    )

    $bmp = New-Object System.Drawing.Bitmap $Size, $Size
    $graphics = [System.Drawing.Graphics]::FromImage($bmp)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::FromArgb(255, 15, 20, 25))

    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 59, 130, 246))
    $fontSize = [float]($Size * 0.42)
    $font = New-Object System.Drawing.Font "Segoe UI", $fontSize, [System.Drawing.FontStyle]::Bold
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $rect = New-Object System.Drawing.RectangleF 0, 0, $Size, $Size
    $graphics.DrawString("F", $font, $brush, $rect, $format)

    $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bmp.Dispose()
}

$publicDir = Join-Path (Join-Path $PSScriptRoot "..") "public"
New-Item -ItemType Directory -Force -Path $publicDir | Out-Null
Write-PwaIcon -Size 192 -Path (Join-Path $publicDir "pwa-192x192.png")
Write-PwaIcon -Size 512 -Path (Join-Path $publicDir "pwa-512x512.png")
