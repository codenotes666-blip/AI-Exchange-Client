# Downloads and extracts Everything SDK DLLs into this folder.
# Run from PowerShell in AI-Exchange-Client\filesearch-app

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$zipUrl = 'https://www.voidtools.com/Everything-SDK.zip'
$zipPath = Join-Path $here 'Everything-SDK.zip'

Write-Host "Downloading Everything SDK..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -LiteralPath $zipPath -DestinationPath $here -Force

Remove-Item -LiteralPath $zipPath -Force

# The zip typically contains Everything64.dll / Everything32.dll in the root.
$dll64 = Join-Path $here 'Everything64.dll'
$dll32 = Join-Path $here 'Everything32.dll'

if (!(Test-Path $dll64) -and !(Test-Path $dll32)) {
    Write-Warning "SDK zip extracted but DLLs were not found next to this script. Look for Everything64.dll/Everything32.dll in extracted folders."
} else {
    Write-Host "OK: SDK DLL(s) are present." -ForegroundColor Green
}
