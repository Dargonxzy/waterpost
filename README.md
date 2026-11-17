# ai3d-waterpost：水域后处理（C++ / CUDA 12.8）

`blockgs/water_postprocessing` 精细层流程（detect → reconcile → apply）的 C++ 版。输入是 gs-exporter 导出的
BIN LOD（新的 fix 布局或旧的逐节点 JSON 布局都可以），输出一份补过水的 LOD；只改各块最细层叶节点的 bin，
粗层与跨块代理原样保留。

```bat
ai3d_waterpost.exe all --project D:\data\滑梯_20261025183928 --work D:\data\滑梯\waterpost_cpp\work --output D:\data\滑梯\waterpost_cpp\lod
```

- `--lod` 默认 `<project>\results\3dgs\lod\bin`，`--mask-dir` 默认 `<project>\.temp\semantic\masks`；
- `--mode blocks`（默认）：块清单、块范围、相机取自 `artifacts\blocks\<块>`，高斯取自 LOD 中该块的最细层；
- `--mode poses`：没有分块成果，只要 LOD（新旧布局都可以）+ 位姿 JSON + 掩码，见下文；
- blocks 模式可以像以前一样直接从 `detect` 开始（`detect --project ... --work W`），不跑 split / assign / scene：块、范围、视图
  直接取自 `artifacts\blocks`，每块在 detect 里自己解码掩码；跑了 scene 则改用 `work\scene`（每张掩码全局只解码一次）。
  poses 模式必须先跑 split / assign / scene；
- 阶段顺序 `split → assign → scene → detect → reconcile → candidates → active → samples → model → write → finalize`，`all` 依次全跑；
- 分阶段 / 分机：`split` 写图像分工清单，各机器 `assign --list <清单>` 处理自己的图像；`detect`、`candidates`、`samples`、`write`
  用 `--tile-name` 按块分发；`split`、`scene`、`reconcile`、`active`、`model`、`finalize` 是全局步骤；所有阶段共用 `--work`
  （第一次 split / assign / all 写入 `config.json`，后续阶段直接读）；
- `--link-output`：未改写的文件用硬链接代替复制（同一磁盘时）；`--debug` 额外保存逐点投票数组。
- 输入（`--project`、`--lod`、`--mask-dir`、`--mode`、`--poses`、`--bbox`、`--roi`）在第一次运行时写进 `work\config.json`，
  之后的阶段直接读；再传不同的值会报错（换一个新的 `--work`）。分机时各机器必须用**同一绝对路径**访问 work、LOD、掩码与位姿
  （视图以掩码的绝对路径为键），盘符不同就用统一的 UNC 路径；
- 可以往同一个 `--output` 重跑：finalize 会把上次改写、这次没改写的文件换回输入内容（按大小与修改时间判断），并删掉
  上次留下的 `_water<k>` 溢出节点；detect 重跑时先清掉本块上一次的交接文件。

## 场景准备（split / assign / scene）与 poses 模式

`split --parts N` 把全部视图（按图像名排序）分成 N 份，写 `work\assign\lists\part_XX.list`（每行一个图像名；默认第 k 张归
第 k mod N 份，`--split contiguous` 改为连续切段）。清单也可以手写，只要所有清单合起来恰好覆盖每张图一次。
`assign --list <清单>` 只处理清单里的图像（不给 `--list` 就处理全部），结果写 `work\assign\<清单名>.npz/.json`。
`assign` 对每张视图只解码一次掩码，得到 32 px 占据格网（ROI 预检与反票用）；poses 模式下还把 LOD 每块最细层叶节点的包围盒
投到这张视图里，记下它看到哪些块。`scene` 合并 `work\assign` 里的全部结果（检查各结果用的是同一组输入，且每张图恰好被处理一次：
漏处理的清单、重复处理的残留结果都会报错），写出
`work\scene\scene.json`（每块的范围与视图、补水范围）、`views.json`、`occupancy.npz`。之后的阶段只读 `work\scene`。

| | blocks | poses |
|---|---|---|
| 块 | 工程块清单 ∩ LOD 块 | LOD 中的块（路径 `L<层>/<块>/` 的第二段） |
| 块范围 | `block.json` 的 core_bounds | 该块最细层叶节点包围盒的并集 |
| 每块视图 | 块的 `sparse\0` | 相机中心 XY 在块范围外扩 `--view-margin`（默认 0.5）× 块边长之内，且看到至少 `--min-view-leaves`（默认 1）个叶节点 |
| 补水范围 | `bbox.json` ∩ `roi.json` | `--bbox`（缺省为 LOD 最细层范围）∩ `--roi`（需要 `--bbox` 里的 Normalizer） |

```bat
:: 单机
ai3d_waterpost.exe all --mode poses --poses D:\d\poses.json --lod D:\d\lod\bin --mask-dir D:\d\masks ^
    [--bbox D:\d\bbox.json --roi D:\d\roi.json] --work W --output OUT
:: 分机：先写分工清单，每台机器处理自己的清单，再在一台机器上 scene，然后按原方式分发 detect / candidates / samples / write
ai3d_waterpost.exe split --parts 3 --mode poses --poses ... --lod ... --mask-dir ... --work W
ai3d_waterpost.exe assign --list W\assign\lists\part_00.list --work W     &:: 机器 0（机器 1、2 换成 part_01 / part_02）
ai3d_waterpost.exe scene --work W
```

位姿有两种给法（都在 poses 模式下，二选一）：

- `--poses FILE`：一个 JSON 同时含 `cameras` 与 `images`（`scripts\colmap_to_poses.py <sparse> <out.json>` 可以从 COLMAP sparse 生成）；
- `--cameras FILE --images FILE`：分开的两个 JSON。两种写法都接受数组或以 id 为键的对象：

```json
cameras.json  {"1": {"width": 8192, "height": 5468, "model": "OPENCV", "params": [fx, fy, cx, cy, k1, k2, p1, p2]}, ...}
images.json   {"1": {"name": "//server/data/.../A/A00001.JPG", "camera_id": 1, "q": [w, x, y, z], "t": [x, y, z]}, ...}
```

- 位姿任选一种：`q`/`qvec` + `t`/`tvec`（COLMAP 世界→相机）、`R`+`t`、`R`+`center`、`camera_to_world`（4×4）；相机可以内联 `"camera": {...}`，
  也接受 `fx`/`fy`/`cx`/`cy` 字段；
- **相机模型**：PINHOLE / SIMPLE_PINHOLE 直接用；SIMPLE_RADIAL / RADIAL / OPENCV / FULL_OPENCV 当作带畸变的原始相机：投影一律用内参不变的
  针孔相机，掩码（和 evaluate 的照片）读入时按原始相机去畸变（掩码最近邻、照片双线性，不另存文件）；畸变系数全 0 时就是针孔；鱼眼模型不支持。
  后续各阶段只看到针孔相机与去畸变后的掩码，不需要任何改动；blocks 模式的相机来自去畸变后的 `sparse\0`，不走这条路；
- **掩码**：`<--mask-dir>/<图像名去掉开头若干级目录，扩展名换成 .png>`，子目录层级照搬图像路径。去掉几级默认自动找（用前几张图从 0 级起试，
  第一个能在掩码目录下找到文件的级数用于全部图像；Windows 下不区分大小写），也可以用 `--image-root PREFIX` 指定要去掉的前缀；
  每张图也可以用 `"mask"` 字段单独指定。两张图对到同一个掩码时报错；
- **坐标**：`--pose-crs auto`（默认）时，相机中心明显是投影坐标（量级 > 10 万米）就用 LOD 根节点 `transform` 的平移（本地原点的 ECEF 坐标）
  换成经纬度、按经度取 UTM 带，把位姿平移到 LOD 本地坐标（本地坐标 = 投影坐标 − 原点投影坐标，在滑梯数据上与 bbox.json 的 Normalizer
  一致到 0.1 mm）；`--pose-crs EPSG:326NN / EPSG:327NN` 指定带号，`none` 表示已是本地坐标。读完后检查相机是否在 LOD 附近，
  坐标系不对会直接报错。

验证：用滑梯数据造了同形态的仿真输入（两文件格式、绝对网络路径的图像名、多级且大小写不同的掩码目录）：针孔相机 + 本地坐标时输出与
`--poses` 结果逐字节一致；再加 OPENCV 畸变（两台相机）与 UTM 50N 坐标时，水位差 6 mm、新增水面点差 0.08%，evaluate 的水 / 非水 PSNR
增量一致（差异来自掩码一次加畸变、一次去畸变的最近邻取整）。

`evaluate` 也接受 `--poses` 或 `--cameras` + `--images`（`--photos DIR` 是照片目录，照片路径与掩码一样自动对上子目录），畸变相机的照片
先去畸变再与针孔渲染比较；它是单独的检查命令，不在 `all` 里运行。

## 规模与性能

- 同一进程连续处理多块（不带 `--tile-name` 或带多个 `--tile-name`、以及 `all`）时，detect / candidates 共享一个 2 GiB 的
  掩码缓存，相邻块共用的视图只解码一次；分机时每台机器多分几块相邻的块更划算；
- candidates 在 CPU 上先剪掉不可能投票的视图（水体签名视图看不到本块、投票区域落不到水格上、待复核格子碰不到水格），
  无水块几乎不再解码掩码；
- samples / write 只读范围相邻块的结果，近邻定尺度只读距离上界以内的分片；reconcile 定水位为 O(n log n)。
  以上都不改变结果：滑梯数据重切成 54 块的压测中，输出与优化前逐字节一致，总耗时 609 s → 316 s
  （candidates 295 s → 70 s，detect 296 s → 228 s）。

## 与 Python 版的对应

| 阶段 | 内容 | Python 来源 |
|---|---|---|
| assign / scene | 每张掩码解码一次 → 32 px 占据格网；poses 模式按可见叶节点给块分配视图 | — |
| detect | 占据格网 ROI 预检；GPU 二值多视图投票 + T×α 可见渲染；稳定岸线锚点；可靠陆地格网 | `workflow_stages.stage_detect`、`mask_voting`、`cuda_contributions`、`prepared_components` |
| reconcile | 共享掩码分量合成水体；岸线定水位与 σ；块范围延伸；全局格网 | `distributed_reconcile`、`rebuild.shore_level`、`domain` |
| candidates | 足迹投票：可能正票（占据超集 + 精确标签）、签名视图正反票、其他视图占据反票、遮挡深度图、自校准似然比、岸线屏障、3×3 闭运算 | `distributed_fill.prepare_candidates`、`body_vote.grid_evidence`、`rebuild` |
| active | 全局 8 连通区域支持 | `reduce_candidates` |
| samples / model | 冲突消解、原水点归属、四叉合并采样、颜色归约 | `finalize_block`、`reduce_plans` |
| write / finalize | 体积清理（水面下、深岸、水面上方、漂浮、渲染保护）、k≤4 近邻定尺度的水面高斯；写回叶节点 bin，更新所有清单的包围盒与点数 | `carve_water_volume`、`_generated`、`write_block` |

bin 带语义字节时（`gssp-semantic16-v1`），新增的水面高斯写成水体类别（`--water-class`，默认 9）、置信度 255。

## 补水范围与节点容量

- 补水范围 = 工程 bbox ∩ ROI 多边形（`roi.json` 的 projected 点经 Normalizer 换到本地坐标；没有 roi.json 时不限）∩ LOD 各块瓦片的 XY 范围。
  足迹格中心必须在范围内；跨边界的采样矩形按中心取舍（中心在范围内整块保留，否则整块丢弃，不拆成单格）；ROI 边缘的水面高斯尺度按到边界的距离收紧。
- 每个叶节点的点数不超过本块原始最细层叶节点的最大记录数（导出时的 max_splat_num，如 61440）。装不下的水面点写进
  `<叶名>_water<k>.bin`，作为兄弟节点挂到同一父节点下（两种布局都处理；旧布局同时生成对应的节点 JSON）。

## 漂浮物去噪（与水无关）

write 阶段对每块最细层的剩余高斯做一次连通簇去噪：中心距离 ≤ `--denoise-radius`（默认 0.5 m）的高斯连成簇，
点数少于 `--denoise-min-points`（默认 30）的小簇删除；碰到块瓦片边界（2 倍半径内）的簇可能连着邻块，保留。
`--denoise 0` 关闭；`--denoise-method sor|radius` 可改用统计离群点 / 半径离群点滤波（默认 `cluster`）。稀疏但与主体相连的高斯（树冠边缘等）属于大簇，不受影响。

## 当前限制

- 相机只支持 PINHOLE / SIMPLE_PINHOLE（有畸变的模型需要先去畸变）；
- 只处理最细层，粗层与 fix 代理不变；
- 没有掩码的视图直接跳过（不在这里调用语义分割）。

构建：`powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1`（依赖与 ai3d-waterpre 相同）。
