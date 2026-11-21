# ai3d-waterpost Linux / CUDA 12.8 运行包

将整个压缩包复制到目标电脑并解压，保留 `bin/`、`lib/`，不要只复制 ELF 文件。
这是原生 C++ 程序，无需安装 Python、Conda、OpenCV 或 CUDA Toolkit；CUDA 运行库已随包提供。
目标电脑仍需可用的 NVIDIA 驱动。

## 目标环境

- Linux x86_64；面向 Ubuntu 20.04、22.04、24.04。构建使用 glibc 2.28 sysroot，打包时检查所有 ELF 的 glibc 要求不高于 2.31。
- GPU 原生代码覆盖 SM 75、80、86、89、90、100、120，包括 RTX 40 系列（SM 89）；另含 compute_120 PTX。
- 建议 NVIDIA Linux 驱动 **570.124.06 或更新**，与 CUDA 12.8.1 完整兼容。CUDA 12.x 的较旧驱动兼容模式有功能限制，详见 [NVIDIA 发布说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)。
- 驱动库 `libcuda.so.1` 和系统 glibc 由目标电脑提供，没有打入包内。

## 使用

```bash
tar -xzf ai3d-waterpost-linux-x86_64-cu128.tar.gz
cd ai3d-waterpost-linux-x86_64-cu128
./ai3d_waterpost --help

# 检查目标电脑能否实际执行本程序的 CUDA 内核（可选参数为 GPU 编号）
./bin/ai3d_waterpost_gpu_smoke 0

# 有分块成果的工程：已有 LOD 和语义 mask
./ai3d_waterpost all \
  --project /data/project \
  --work /data/waterpost-work \
  --output /data/waterpost-lod \
  --gpu-id 0 --threads 8

# 只有 LOD、位姿 JSON 和 mask
./ai3d_waterpost all --mode poses \
  --lod /data/lod/bin --poses /data/poses.json --mask-dir /data/masks \
  --work /data/waterpost-work --output /data/waterpost-lod \
  --gpu-id 0 --threads 8
```

工作目录、输出目录和输入 LOD 请使用三个不同的目录。迁移 Windows 数据后，在 Linux 上使用新的 `--work`，
避免旧 `config.json` 中保存的 Windows 绝对路径。`write` 默认还会对整块最细层去噪，`--denoise 0` 可关闭。
详细参数见 `CLI-README.md`。

`./bin/ai3d_waterpost_tests` 可运行 CPU 单元测试。`build-info.json` 记录 CUDA 架构、ELF 系统库要求和源码/二进制校验值；
`validation.txt` 记录打包后脱离 Conda 环境的测试结果。测试用小场景验证 CUDA 各计算路径，不替代真实工程的质量检查。
