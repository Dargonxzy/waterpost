# 用 Python 版 water_postprocessing 跑 detect -> reconcile -> apply（精细层），作为 C++ 版的对比基线。
#   powershell -ExecutionPolicy Bypass -File scripts\run_python.ps1 -Project D:\data\滑梯 -Out D:\data\滑梯\waterpost_py
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Python = "D:\workspacex\anaconda3\envs\block_gs\python.exe",
    [string]$Code = "D:\workspacex\01code\blockgs"
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$entry = Join-Path $Code "waterpost.py"
$times = @{}
foreach ($stage in @("detect", "reconcile", "apply")) {
    $start = Get-Date
    if ($stage -eq "detect") {
        & $Python $entry --stage detect --project $Project --output (Join-Path $Out "detect") --debug
    } elseif ($stage -eq "reconcile") {
        & $Python $entry --stage reconcile --input (Join-Path $Out "detect") --output (Join-Path $Out "reconcile")
    } else {
        & $Python $entry --stage apply --input (Join-Path $Out "reconcile") --output (Join-Path $Out "apply")
    }
    if ($LASTEXITCODE -ne 0) { throw "python $stage failed ($LASTEXITCODE)" }
    $times[$stage] = [Math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    Write-Host ("[PYTIME] {0} {1}s" -f $stage, $times[$stage])
}
