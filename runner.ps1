# runner.ps1 —— 让 Claude 在这台 Windows 机器上执行构建 / 测试命令。
#
# 启动（保持窗口开着；Ctrl+C 停止）：
#   powershell -ExecutionPolicy Bypass -File D:\user\Desktop\code2\ai3d-waterpost\runner.ps1
#
# 协议（目录 .runner\ 在本脚本旁边）：
#   inbox\<id>.json   任务：{"cwd": "相对 code2 的目录", "program": "...", "args": ["...", ...]}
#   outbox\<id>.out / <id>.err   实时写入的标准输出 / 标准错误
#   outbox\<id>.status           {"state": "running|done|rejected|cancelled", "exit": n, ...}
#   inbox\<id>.cancel             存在时终止该任务（连同子进程）
#   heartbeat                     runner 在线时每 2 秒刷新
#
# 安全限制：工作目录必须在 code2 下；program 只允许 cmake / ctest / git / where / nvidia-smi、
# 名为 python.exe 的解释器、ai3d-waterpost\build 下的 exe，以及 ai3d-waterpost\scripts 下的 .ps1。
# 任务逐个顺序执行。
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$code2 = [System.IO.Path]::GetFullPath((Join-Path $here ".."))
$state = Join-Path $here ".runner"
$inbox = Join-Path $state "inbox"
$outbox = Join-Path $state "outbox"
New-Item -ItemType Directory -Force -Path $inbox, $outbox | Out-Null
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-Status([string]$id, [hashtable]$value) {
    $value["time"] = (Get-Date).ToString("s")
    [System.IO.File]::WriteAllText((Join-Path $outbox "$id.status"), ($value | ConvertTo-Json -Compress), $utf8)
}

function Quote([string]$text) {
    if ($text -eq "") { return '""' }
    if ($text -notmatch '[\s"]') { return $text }
    return '"' + ($text -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Test-Allowed([string]$program, [string]$cwd) {
    $leaf = [System.IO.Path]::GetFileName($program).ToLower()
    if (@("cmake", "cmake.exe", "ctest", "ctest.exe", "git", "git.exe", "where", "where.exe", "nvidia-smi", "nvidia-smi.exe") -contains $leaf) { return $true }
    if ($leaf -eq "python.exe" -or $leaf -eq "python") { return $true }
    $full = $program
    if (-not [System.IO.Path]::IsPathRooted($full)) { $full = Join-Path $cwd $program }
    $full = [System.IO.Path]::GetFullPath($full)
    $build = [System.IO.Path]::GetFullPath((Join-Path $here "build")) + "\"
    $scripts = [System.IO.Path]::GetFullPath((Join-Path $here "scripts")) + "\"
    if ($full.StartsWith($build, "OrdinalIgnoreCase") -and $full.ToLower().EndsWith(".exe")) { return $true }
    if ($full.StartsWith($scripts, "OrdinalIgnoreCase") -and $full.ToLower().EndsWith(".ps1")) { return $true }
    return $false
}

Write-Host "Runner started. code2=$code2  queue=$inbox  (Ctrl+C to stop)" -ForegroundColor Green
while ($true) {
    [System.IO.File]::WriteAllText((Join-Path $state "heartbeat"), (Get-Date).ToString("s"), $utf8)
    $jobs = Get-ChildItem $inbox -Filter *.json -ErrorAction SilentlyContinue | Sort-Object Name
    foreach ($file in $jobs) {
        $id = $file.BaseName
        try {
            $job = [System.IO.File]::ReadAllText($file.FullName, $utf8) | ConvertFrom-Json
        } catch {
            Start-Sleep -Milliseconds 500
            try { $job = [System.IO.File]::ReadAllText($file.FullName, $utf8) | ConvertFrom-Json } catch {
                Remove-Item $file.FullName -Force
                Write-Status $id @{ state = "rejected"; reason = "invalid JSON: $_" }
                continue
            }
        }
        Remove-Item $file.FullName -Force
        $cwd = [System.IO.Path]::GetFullPath((Join-Path $code2 ([string]$job.cwd)))
        if (-not $cwd.StartsWith($code2, "OrdinalIgnoreCase") -or -not (Test-Path $cwd -PathType Container)) {
            Write-Status $id @{ state = "rejected"; reason = "cwd must be an existing directory under code2: $cwd" }
            continue
        }
        $program = [string]$job.program
        if (-not (Test-Allowed $program $cwd)) {
            Write-Status $id @{ state = "rejected"; reason = "program not allowed: $program" }
            continue
        }
        $arguments = @()
        if ($job.args) { $arguments = @($job.args | ForEach-Object { [string]$_ }) }
        if ($program.ToLower().EndsWith(".ps1")) {
            $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $program) + $arguments
            $program = "powershell.exe"
        }
        $argumentText = ($arguments | ForEach-Object { Quote $_ }) -join " "
        $out = Join-Path $outbox "$id.out"
        $err = Join-Path $outbox "$id.err"
        Write-Host "[$id] ($cwd) $program $argumentText" -ForegroundColor Cyan
        $started = Get-Date
        try {
            $startArgs = @{ FilePath = $program; WorkingDirectory = $cwd; RedirectStandardOutput = $out; RedirectStandardError = $err; NoNewWindow = $true; PassThru = $true }
            if ($argumentText) { $startArgs["ArgumentList"] = $argumentText }
            $process = Start-Process @startArgs
            $null = $process.Handle
        } catch {
            Write-Status $id @{ state = "rejected"; reason = "start failed: $_" }
            continue
        }
        Write-Status $id @{ state = "running"; pid = $process.Id; start = $started.ToString("s") }
        $cancelled = $false
        while (-not $process.HasExited) {
            [System.IO.File]::WriteAllText((Join-Path $state "heartbeat"), (Get-Date).ToString("s"), $utf8)
            if (Test-Path (Join-Path $inbox "$id.cancel")) {
                & taskkill.exe /T /F /PID $process.Id | Out-Null
                Remove-Item (Join-Path $inbox "$id.cancel") -Force
                $cancelled = $true
                break
            }
            Start-Sleep -Seconds 2
        }
        $process.WaitForExit()
        $seconds = [Math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        if ($cancelled) {
            Write-Status $id @{ state = "cancelled"; seconds = $seconds }
        } else {
            Write-Status $id @{ state = "done"; exit = $process.ExitCode; seconds = $seconds }
        }
        Write-Host "[$id] finished exit=$($process.ExitCode) in ${seconds}s" -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 2
}
