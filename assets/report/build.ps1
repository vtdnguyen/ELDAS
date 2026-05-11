# build.ps1 — Compile main.tex → out/main.pdf
# Chạy từ thư mục assets/report/: .\build.ps1
# Hoặc từ project root:           .\assets\report\build.ps1

$ErrorActionPreference = 'Stop'
$ReportDir = $PSScriptRoot
$OutDir = Join-Path $ReportDir 'out'

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Push-Location $ReportDir

try {
    Write-Host '[1/4] pdflatex (pass 1)...' -ForegroundColor Cyan
    pdflatex -synctex=1 -interaction=nonstopmode -output-directory=out main.tex

    Write-Host '[2/4] biber...' -ForegroundColor Cyan
    biber --output-directory out out/main

    Write-Host '[3/4] pdflatex (pass 2)...' -ForegroundColor Cyan
    pdflatex -synctex=1 -interaction=nonstopmode -output-directory=out main.tex

    Write-Host '[4/4] pdflatex (pass 3 — resolve cross-refs)...' -ForegroundColor Cyan
    pdflatex -synctex=1 -interaction=nonstopmode -output-directory=out main.tex

    Write-Host "`nDone! PDF: $OutDir\main.pdf" -ForegroundColor Green
} finally {
    Pop-Location
}
