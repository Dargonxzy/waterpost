#!/usr/bin/env bash
# Build with a separate Conda C++ toolchain and an existing CUDA 12.8 toolkit.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${AI3D_WATERPOST_BUILD_PREFIX:-${CONDA_PREFIX:-}}
cuda=${CUDA_HOME:-${CUDA_PATH:-}}
build="$root/build/linux-cu128"
archs='75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual'
jobs=8
while (($#)); do
    case "$1" in
        --prefix) prefix=$2; shift 2 ;;
        --cuda-root) cuda=$2; shift 2 ;;
        --build-dir) build=$2; shift 2 ;;
        --archs) archs=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $(uname -m) == x86_64 ]] || { echo 'This build script targets Linux x86_64.' >&2; exit 2; }
[[ -x "$prefix/bin/x86_64-conda-linux-gnu-c++" && -x "$cuda/bin/nvcc" ]] || {
    echo 'Pass --prefix <C++ build environment> --cuda-root <CUDA 12.8 toolkit>.' >&2; exit 2;
}
"$cuda/bin/nvcc" --version | grep -q 'release 12\.8,' || { echo 'CUDA 12.8 is required.' >&2; exit 2; }
export PATH="$prefix/bin:$cuda/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$prefix/lib:$cuda/targets/x86_64-linux/lib:$cuda/lib"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
cmake -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-cc" \
    -DCMAKE_CXX_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$prefix/bin/x86_64-conda-linux-gnu-c++" \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCUDAToolkit_ROOT="$cuda" \
    -DCMAKE_PREFIX_PATH="$prefix" \
    "-DCMAKE_EXE_LINKER_FLAGS=-L$prefix/lib -Wl,-rpath-link,$prefix/lib" \
    '-DCMAKE_CXX_FLAGS=-march=x86-64 -mtune=generic' \
    -DAI3D_WATERPOST_USE_SIBLINGS=ON \
    -DAI3D_CUDA_ALLOW_UNSUPPORTED_COMPILER=OFF \
    -DAI3D_CUDA_ARCHITECTURES="$archs" \
    -DAI3D_WATERPOST_GPU_SMOKE_TEST=ON
cmake --build "$build" --parallel "$jobs" --target ai3d_waterpost ai3d_waterpost_tests ai3d_waterpost_gpu_smoke
ctest --test-dir "$build" --output-on-failure
"$build/bin/ai3d_waterpost" --help
