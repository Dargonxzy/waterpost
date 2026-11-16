# ai3d-waterpost：水域后处理（C++ / CUDA 12.8）

`blockgs/water_postprocessing` 精细层流程（detect → reconcile → apply）的 C++ 版。输入是 gs-exporter 导出的
BIN LOD（新的 fix 布局或旧的逐节点 JSON 布局都可以），输出一份补过水的 LOD；只改各块最细层叶节点的 bin，
粗层与跨块代理原样保留。

```bat
ai3d_waterpost.exe all --project D:\data\滑梯_20261025183928 --work D:\data\滑梯\waterpost_cpp\work --output D:\data\滑梯\waterpost_cpp\lod
```

- `--lod` 默认 `<project>\results\3dgs\lod\bin`，`--mask-dir` 默认 `<project>\.temp\semantic\masks`；
- `--mode blocks`（默认，目前唯一实现）：块清单、块范围、相机取自 `artifacts\blocks\<块>`，高斯取自 LOD 中该块的最细层；
- 分阶段 / 分机：`detect`、`candidates`、`samples`、`write` 可用 `--tile-name` 按块分发，`reconcile`、`active`、`model`、
  `finalize` 是全局步骤；所有阶段共用 `--work`（第一次 detect 写入 `config.json`，后续阶段直接读）；
- `--link-output`：未改写的文件用硬链接代替复制（同一磁盘时）；`--debug` 额外保存逐点投票数组。

## 与 Python 版的对应

| 阶段 | 内容 | Python 来源 |
|---|---|---|
| detect | 每张掩码解码一次 → 32 px 占据格网与 ROI 预检；GPU 二值多视图投票 + T×α 可见渲染；稳定岸线锚点；可靠陆地格网 | `workflow_stages.stage_detect`、`mask_voting`、`cuda_contributions`、`prepared_components` |
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
