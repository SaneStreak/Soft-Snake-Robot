# Launch SOFA v26.06.00 3D GUI with SofaPython3 and our Soft Snake FEM IPC Scene
$env:SOFA_ROOT = "C:\Users\sanch\SOFA\v26.06.00"
$env:SOFAPYTHON3_ROOT = "$env:SOFA_ROOT\plugins\SofaPython3"
$env:PATH = "C:\Users\sanch\anaconda3\envs\sofa_py312;$env:SOFA_ROOT\bin;$env:SOFAPYTHON3_ROOT\bin;$env:PATH"
$env:PYTHONPATH = "$env:SOFAPYTHON3_ROOT\lib\python3\site-packages;$PSScriptRoot\.."

Write-Host "Starting SOFA v26.06.00 3D GUI (Soft Snake FEM Scene)..." -ForegroundColor Cyan
Write-Host "Once the window opens, run in another terminal:" -ForegroundColor Yellow
Write-Host "  conda activate soft_snake"
Write-Host "  python scripts/run_sofa_eval.py --checkpoint checkpoints/ppoc_best.pt --external --host 127.0.0.1 --port 5555 --no-animate"
& "$env:SOFA_ROOT\bin\runSofa.exe" -g imgui -a -l "$env:SOFAPYTHON3_ROOT\bin\SofaPython3.dll" "$PSScriptRoot\sofa_scene_server.py"

