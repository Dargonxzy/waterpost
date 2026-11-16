# 在 Windows 上构建 ai3d-waterpost（VS2022），并把运行所需的 DLL 复制到 exe 旁边。
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1                 # 默认 windows-siblings
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 -Preset windows # 只用已安装的 ai3d 包
#   ... -RunTests                                                                      # 构建后跑单元测试
#
# 依赖默认取 ..\ai3dgs\build\vs2022-cuda\vcpkg_installed\x64-windows（ai3dgs 已用 vcpkg 装好的目录），
# 可用环境变量 AI3D_WATERPOST_PREFIX 指向别的 vcpkg_installed\x64-windows。
param(
    [string]$Preset = "windows-siblings",
    [switch]$RunTests
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if ($env:AI3D_WATERPOST_PREFIX) {
        $prefix = $env:AI3D_WATERPOST_PREFIX
    } else {
        $prefix = Join-Path $root "..\ai3dgs\build\vs2022-cuda\vcpkg_installed\x64-windows"
    }
    if (-not (Test-Path (Join-Path $prefix "share\opencv4"))) {
        throw "找不到依赖目录 $prefix（需要 opencv4 / tbb / eigen3 / nlohmann-json / zstd / zlib，以及 ai3d_core / ai3d_io 或其源码）。"
    }
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    cmake --build --preset $Preset --parallel
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
    $bin = Join-Path $root "build\$Preset\bin\Release"
    Copy-Item (Join-Path $prefix "bin\*.dll") $bin -Force
    # CUDA 构建（AI3D_WATERPOST_ENABLE_CUDA=ON）动态链接 CUDA 运行时：把 cudart64_*.dll 放到 exe 旁边。
    $cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { $null }
    if ($cudaBin -and (Test-Path $cudaBin)) {
        Get-ChildItem (Join-Path $cudaBin "cudart64_*.dll") -ErrorAction SilentlyContinue | Copy-Item -Destination $bin -Force
    }
    Write-Host "构建完成：$bin\ai3d_waterpost.exe" -ForegroundColor Green
    if ($RunTests) {
        ctest --preset $Preset
        if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    }
} finally {
    Pop-Location
}
