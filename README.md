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

位姿 JSON（`scripts\colmap_to_poses.py <sparse> <out.json>` 可以从 COLMAP sparse 生成）：坐标系必须与 LOD 相同（本地坐标），
相机为去畸变后的 PINHOLE / SIMPLE_PINHOLE，掩码对应去畸变后的图像。

```json
{"cameras": [{"id": 1, "model": "PINHOLE", "width": 2640, "height": 1912, "params": [fx, fy, cx, cy]}],
 "images":  [{"name": "DJI_0001.jpg", "camera_id": 1, "qvec": [w, x, y, z], "tvec": [x, y, z]}]}
```

- 位姿任选一种：`qvec`+`tvec`（COLMAP 世界→相机）、`R`+`t`（世界→相机，3×3 嵌套或 9 个数）、`R`+`center`、`camera_to_world`（4×4）；
- 相机可以写在 `cameras`（数组或以 id 为键的对象）里用 `camera_id` 引用，也可以每张图内联 `"camera": {...}`；
  相机也接受 `fx`/`fy`/`cx`/`cy` 字段；
- 掩码默认 `<mask-dir>\<图像名换成 .png>`，也可以每张图给 `"mask"`（相对 `--mask-dir` 或绝对路径）；没有掩码的视图跳过。

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
  足迹格中心必须在范围内；跨边界的采样矩形拆成单格，只留中心在范围内的格；ROI 边缘的水面高斯尺度按到边界的距离收紧。
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
