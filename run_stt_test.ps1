# End-to-end STT test: start server, send PCM, show transcript lines from log.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"
Set-Location $root

# Ensure PCM exists (convert from testing.wav if needed)
if (-not (Test-Path "testing_5s.pcm") -or ((Get-Item "testing_5s.pcm").Length -lt 50000)) {
    Write-Host "Converting testing.wav -> testing_5s.pcm ..."
    & $python wav_to_pcm.py testing.wav testing_5s.pcm
}

$log = Join-Path $root "stt_test_log.txt"
Remove-Item $log -ErrorAction SilentlyContinue

Write-Host "Starting server..."
$proc = Start-Process -FilePath $python -ArgumentList "-u", "voxgraph.py" -RedirectStandardOutput $log -RedirectStandardError "${log}.err" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 4

try {
    Write-Host "Sending audio..."
    & $python send_test_pcm.py testing_5s.pcm
    Start-Sleep -Seconds 3
    Write-Host "`n--- STT results ---"
    Select-String -Path $log -Pattern "STT \[Results\]|Utterance complete" | ForEach-Object { $_.Line }
}
finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
