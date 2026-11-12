# Launch SOFA v26.06.00 3D Multi-Region Arena + Companion Click-to-Add-Waypoints UI
$env:SOFA_ROOT = "C:\Users\sanch\SOFA\v26.06.00"
$env:SOFAPYTHON3_ROOT = "$env:SOFA_ROOT\plugins\SofaPython3"
$env:PATH = "C:\Users\sanch\anaconda3\envs\sofa_py312;$env:SOFA_ROOT\bin;$env:SOFAPYTHON3_ROOT\bin;$env:PATH"
$env:PYTHONPATH = "$env:SOFAPYTHON3_ROOT\lib\python3\site-packages;$PSScriptRoot\.."

Write-Host "Starting Interactive Soft Snake Arena (SOFA 3D GUI + Click-to-Target Radar)..." -ForegroundColor Cyan
Start-Process -FilePath "C:\Users\sanch\anaconda3\envs\soft_snake\python.exe" `
    -ArgumentList "$PSScriptRoot\interactive_sofa_arena.py --host 127.0.0.1 --port 5555" `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent)

& "$env:SOFA_ROOT\bin\runSofa.exe" -g imgui -a -l "$env:SOFAPYTHON3_ROOT\bin\SofaPython3.dll" "$PSScriptRoot\sofa_scene_server.py"

