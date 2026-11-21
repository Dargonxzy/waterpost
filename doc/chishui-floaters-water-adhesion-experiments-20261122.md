# 漂浮簇 / 水面粘连：实现与本地实验（分支 `fix/floaters-water-adhesion`）

日期：2026-11-22（Asia/Singapore）
依据：[问题分析与方案](chishui-floaters-water-adhesion-analysis-and-plan-20261121.md)
数据：`data\滑梯_20261025183928`（2 块，14.9 M 最细层高斯，1350 张照片）、`data\test2`（2 块，17.0 M，598 张，带语义字节）

## 1. 结论

| 问题 | 实现 | 本地实验结果 | 建议 |
| --- | --- | --- | --- |
| A 独立漂浮簇 | 中等簇（≥ 30 点）几何判定 + **训练照片核验**，只有"照片支持删除且几何上与主体分离"才删除 | 注入的悬空漂浮簇完整清除：滑梯 12→33/40、test2 1→37/40；真实中等簇（539 个）**0 点误删** | 可以对赤水块开启 `--floater-check 1` |
| A（仅几何） | 同上，但不看照片 | 会删掉 9 个（滑梯）/ 27 个（test2）真实簇；照片核验对其中 34 个支持保留，其余 2 个无结论 / 无可用像素，没有一个支持删除 | **不能**只靠几何删除 |
| B 粘连的原水点 | 逐点残留诊断（原因 × 是否粘在大簇上）；可选的岸边带清理 `--carve-edge-band` | 最大一类残留是"水位附近、腐蚀内部区之外"的岸边带，原规则没有分支能删；但删掉后照片吻合度**下降**（test2 岸线带 −0.16 dB） | 岸边带清理保持关闭；B 需要真实粘连样本再定位 |
| C 新水面越界 | 统计新水面 2σ 越出已接受足迹的数量；可选 `--water-edge-sigma K` 把尺度收到边界内 | 越界 >0.2 m 的新水面高斯：滑梯 8561/48235，test2 1928/6686，平均越出约 0.45 m；K=2 后岸线带 PSNR 滑梯 +0.16 dB、test2 +0.12 dB（陆侧 +0.13 / +0.23 dB） | 建议开启 `--water-edge-sigma 2` |

关键发现：在两份真实数据里，几何上"悬空、与主体隔开 ≥ 1.5 m"的中等簇几乎都是真实结构（缺少树干的树冠、灯具、建筑构件等）——照片在这些簇的投影区域里一致支持保留（去掉后区域 PSNR 平均下降 1.6 dB，几乎所有视图都变差）。这验证了原文档的判断："孤立"本身不是删除理由，必须有独立观测。

## 2. 代码改动

| 文件 | 内容 |
| --- | --- |
| `src/waterpost/floaters.hpp/.cpp`（新） | `classify_floaters`：对点数在 `[denoise_min_points, floater_max_points]` 的连通簇计算主体距离 gap、下方支撑 / 水位与离空高度 clearance、上方遮挡、细长度、簇内间距、平均不透明度、多数语义，给出几何判定 |
| `src/waterpost/floater_photos.cpp`（新） | `verify_floaters_with_photos`：在看得到候选簇的训练视图里渲染"原块 + 邻块"与"去掉候选簇"两版，只在簇投影矩形内、两版不同且背后有内容的像素上与照片比较；`combine_floater_evidence`：最终决策 |
| `src/waterpost/photos.hpp`（新） | 照片读取（缩放、畸变去除）从 evaluate 抽出，共用 |
| `src/waterpost/apply_write.cpp` | 去噪第二步接入；足迹格网窗口提到体积清理之前供三处共用；残留水点逐点诊断；岸边带分支（可选）；新水面岸线越界统计与可选收紧；`floaters.json`、`water_residual.npz`（`--debug`）与 `record.json` 新字段 |
| `src/waterpost/evaluate.cpp` | `--regions floaters.json`：按区域比较多个 LOD 与照片；新增岸线带（掩码边界 15 px 内）PSNR |
| `types.hpp` / `project.cpp` / `apps/ai3d_waterpost.cpp` | 新参数（见下） |
| `tests/test_main.cpp` | `test_floater_classification`：悬空、贴近主体、水面小船、下方无观测、屋顶下、线状、块边界 7 种情况 |
| `scripts/experiments/inject_floaters.py`、`check_injected.py`（新） | 合成漂浮簇注入（硬链接复制 LOD，只改补齐区）与逐条核对 |

新参数（默认值不改变现有输出；`floater_check` 默认 0 只报告）：

```text
--floater-check 0|1|2       0 只写 floaters.json；1 几何 + 照片核验后删除；2 仅实验：删全部非边界中等簇
--floater-max-points 2000   大于它的簇视为主体
--floater-gap 1.5           到主体 >= 它才算"分离"（m）
--floater-clearance 1.5     簇底到下方主体表面 / 水面 >= 它才算"悬空"（m）
--floater-elongation 8      线状结构只标记待核查
--floater-photos DIR        训练照片（默认 <project>/.temp/undistortion/images）
--floater-photo-views 24    每簇最多视图数；--floater-photo-gain 0.1 结论所需的区域 PSNR 变化（dB）
--carve-edge-band 0|1       岸边带清理（默认关，实验表明有害）
--water-edge-sigma K        新水面水平 K sigma 收在足迹内（0 关闭，仍统计越界）
evaluate --regions F.json [--region-out CSV] [--region-pad PX]
```

## 3. 问题 A 的判定流程

```text
连通簇（0.5 m）
 ├─ < 30 点：原小簇规则删除（不变）
 ├─ > floater_max_points：主体
 └─ 中等簇 → 几何判定
      keep_block_border          碰到块边界
      keep_near_structure        到主体 < gap
      review_elongated           细长（电线等）
      review_detached_covered    上方有主体（树冠下、桥下）
      review_unobserved_below    下方既无主体也无水面（"没拍到"不等于悬空）
      remove_suspended_floater   离空 >= clearance
      review_detached_supported  其余（如贴着水面的小船）
    → 照片核验（floater_check = 1，非边界簇全部核验）
      photo_favors_drop / photo_favors_keep / photo_inconclusive / photo_unavailable
    → 最终
      remove_photo_confirmed_floater         照片支持删除 且 几何为 remove_suspended / review_detached_supported / review_unobserved_below
      review_near_structure_photo_disputes   照片支持删除，但几何上贴近主体（不删）
      keep_photo_supported                   照片支持保留
      review_suspended_photo_inconclusive / review_suspended_no_photo   几何悬空但照片无结论 / 无照片（不删）
```

照片判据：至少 5 个"两版渲染有差别"的视图；去掉后区域 PSNR 提高 ≥ 0.1 dB 且 ≥ 60% 的视图更接近照片 → 支持删除；对称条件 → 支持保留。

实验中修正过的两处（都来自真实数据的反例）：
1. 只渲染本块时，斜视图里簇背后是黑色（邻块不在渲染里），任何有颜色的团块都显得"更接近照片"，导致注入的漂浮簇被判为支持保留。改为渲染时带上 50 m 内邻块，并跳过"去掉簇后背后为黑"的像素、只比较两版不同的像素。
2. 第一版允许照片单独删除"贴近主体"的簇，滑梯 5 个、test2 3 个真实簇被删；全场景复核显示这些簇没有一致的删除证据（Δ ≈ 0 或为负）。改为贴近主体的簇只标记待核查。

## 4. 实验

### 4.1 真实中等簇的照片证据（滑梯，未注入）

`--floater-check 2` 生成"删除全部中等簇"的 LOD，与只报告的 LOD 一起用 `evaluate --regions` 对比（338 个视图，全场景渲染）：

| 几何判定 | 簇数 | 照片支持删除 | 照片支持保留 | 无结论 | 去掉后区域 PSNR 平均变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| remove_suspended_floater | 9 | 0 | 9 | 0 | −1.64 dB |
| review_unobserved_below | 16 | 0 | 16 | 0 | −2.52 dB |
| review_detached_supported | 9 | 0 | 5 | 4 | −1.20 dB |
| review_detached_covered | 12 | 0 | 5 | 7 | −0.44 dB |
| keep_near_structure | 133 | 0 | 114 | 19 | −0.89 dB |
| keep_block_border | 14 | 0 | 0 | 14 | 0 |

去掉全部 193 个中等簇后整图 PSNR 19.220 → 19.189 dB。滑梯里没有照片支持删除的中等簇。

### 4.2 合成漂浮簇（两份数据）

每块注入 20 个"远"簇（36～72 点，离局部顶面 2～8 m）与 10 个"近"簇（离顶面 0.6～1.4 m），颜色取自附近表面，照片里不存在。全流程（detect 起）重跑，比较原规则与新规则（`--floater-check 1 --water-edge-sigma 2`），按 float32 坐标逐条核对输出 LOD。

| 数据 | 类别 | 簇数 | 完整清除（原规则 → 新规则） | 剩余点（原 → 新） |
| --- | --- | ---: | --- | --- |
| 滑梯 | 远 | 40 | 12 → **33** | 1439 → 349 / 2076 |
| 滑梯 | 近 | 20 | 9 → **15** | 540 → 264 / 1032 |
| test2 | 远 | 40 | 1 → **37** | 2094 → 135 / 2180 |
| test2 | 近 | 20 | 0 → **14** | 1096 → 345 / 1115 |

| 数据 | 真实中等簇 | 仅几何会删除 | 新规则删除 | 新规则标记待核查 |
| --- | ---: | --- | ---: | ---: |
| 滑梯 | 193 | 9 簇 / 1567 点 | **0** | 18 |
| test2 | 346 | 27 簇 / 1842 点 | **0** | 11 |

原规则下被清除的注入簇，都是因为太稀疏、碎成了 < 30 点的小簇。新规则漏掉的远簇（滑梯 7 个、test2 3 个）中，照片无结论或无可用像素的按待核查保留，其余被照片判为支持保留（注入簇颜色取自附近表面，0 阶颜色渲染下有时比背后的半透明区域更接近照片）。没有一个漏网簇是被静默放过的，都在 `floaters.json` 里。

### 4.3 问题 B：残留水点诊断与岸边带

`record.json` 的 `water_residual`（带水证据却未删除的原始点，按第一条未满足的条件归类）：

| 块 | 残留 | 其中语义为水 | 粘在大簇上 | 主要原因 |
| --- | ---: | ---: | ---: | --- |
| 滑梯 Tile_0001 | 25734 | — | 22906 | 岸边带 11433、水面上方不在足迹 4935、可见水份额低 4509 |
| 滑梯 Tile_0002 | 11405 | — | 8788 | 岸边带 3759、不在足迹 3708 |
| test2 S00 | 13481 | 9473 | 12863 | 岸边带 7374、不在足迹 2715、无水体可达 1663 |
| test2 S01 | 36409 | 23388 | 34886 | **无已接受水体可达 15221**、岸边带 9461、不在足迹 8044 |

"岸边带"是原规则的逻辑空档（below 要求腐蚀内部区，deep 要求低于水位 0.5 m）。`--carve-edge-band 1` 填上这个空档后：

| 数据 | 额外删除 | 整图水域 PSNR | 岸线带 PSNR（全 / 水侧 / 陆侧） |
| --- | ---: | --- | --- |
| 滑梯 | +681 | 14.538 → 14.537 | 15.422 → 15.419 / 14.669 → 14.664 / 16.788 → 16.787 |
| test2 | +5205 | 12.538 → **12.417** | 14.941 → **14.782** / 13.043 → 12.823 / 17.896 → 17.799 |

岸边带的原始点在这两份数据里起到了填补新水面与岸之间缝隙的作用，删掉后变差，所以默认关闭。test2 S01 最大的一类残留来自没有被接受的水体，属于 reconcile / active 的接受问题，不是 write 能处理的。B 还需要用户反馈中的具体粘连样本，按 `water_residual.npz`（`--debug`）里的坐标和原因定位。

### 4.4 问题 C：新水面越界

| 数据 | 新水面高斯 | 2σ 越出足迹 > 0.2 m | 平均 / 最大越出 |
| --- | ---: | ---: | --- |
| 滑梯 | 48235 | 8561 | 0.45 / 1.70 m |
| test2 | 6686 | 1928 | 0.43 m |

原因：足迹边缘格的中心离边界只有半格（0.2 m），而覆盖要求 sigma ≥ 格宽（0.4 m）、近邻尺度约 0.4～0.6 m；`Domain::horizontal_clearance` 只约束工程外包框。`--water-edge-sigma 2` 把每轴尺度收到 `max(到足迹边界距离 / 2, 5 cm)`：

| 数据 | 整图（全 / 水 / 陆） | 岸线带（全 / 水侧 / 陆侧） |
| --- | --- | --- |
| 滑梯 base | 19.220 / 14.538 / 21.048 | 15.422 / 14.669 / 16.788 |
| 滑梯 K=2 | 19.208 / 14.525 / 21.053 | **15.582 / 14.845 / 16.913** |
| test2 base | 19.904 / 12.538 / 20.151 | 14.941 / 13.043 / 17.896 |
| test2 K=2 | 19.907 / 12.576 / 20.154 | **15.062 / 13.146 / 18.122** |

### 4.5 耗时

照片核验每块多 9～24 s（194～315 个视图，含邻块 5.7～9.1 M 高斯作背景），write 每块从约 4 s 增至 14～29 s。其余新增统计 < 1 s。

## 5. 赤水上的使用方法

在服务器对 `Node_00101110` 所在任务，用原 work 的**副本**重跑 write / finalize（不要改原 work 与原输出）：

```bash
cp -r <work> <work_fix>
ai3d_waterpost write    --work <work_fix> --output <out_fix> --tile-name Node_00101110 \
    --floater-check 1 --water-edge-sigma 2 [--floater-photos <去畸变照片目录>]
ai3d_waterpost finalize --work <work_fix> --output <out_fix>
```

然后在 `<work_fix>/apply/write/Node_00101110/floaters.json` 里按文档 2.2 的 6 个簇中心查它们的 `geometry` / `photo` / `decision`：

- `remove_photo_confirmed_floater`：已删除，用 313 条记录核对残留为 0；
- `review_near_structure_photo_disputes`：照片支持删除，但离树冠 < 1.5 m。可对该块用更小的 `--floater-gap`（例如 1.0）重跑，或走文档 4.1 的精确记录删除（本分支未实现）；
- `keep_photo_supported`：照片不支持删除，需要人工复核截图判断。

照片目录需与 blocks 模式的去畸变相机一致（`.temp/undistortion/images`）。

## 6. 限制与未做

- 照片核验用 0 阶颜色渲染，与照片的绝对 PSNR 不高（14～20 dB），只用作"有 / 没有这团高斯"的相对比较。被前景完全遮挡、或背后没有重建的簇得不到结论，按待核查保留。
- 渲染背景只带 50 m 内的邻块；超大块或相机离得很远时，背景仍可能不完整。
- 合成漂浮簇的颜色、形状是人为设定的，召回率只说明机制有效，不代表真实漂浮物的召回率。赤水 6 簇需要在服务器上用真实照片验证。
- 文档 4.1 的"按记录精确删除"工具、B 的弱连接分割、C 的局部重采样（目前是收紧尺度，没有再加采样点）均未实现。
- Linux 构建未在本机验证。新文件已加入同一个 CMakeLists，未引入新依赖。

## 7. 复现

实验目录：`data\<数据>\waterpost_exp\fx_*`（work 都从 `base_work` 复制；`fx_syn_*` 为注入实验）。

```bat
:: 注入
python scripts\experiments\inject_floaters.py <project>\results\3dgs\lod\bin <exp>\fx_syn_lod_in <exp>\fx_syn_injected.json
ai3d_waterpost all --project <project> --lod <exp>\fx_syn_lod_in --work <exp>\fx_syn_work --output <exp>\fx_syn_lod_out --floater-check 1 --water-edge-sigma 2 --link-output
python scripts\experiments\check_injected.py <exp>\fx_syn_injected.json <exp>\fx_syn_lod_out <exp>\fx_syn_work
:: 照片对比（区域 / 岸线带）
ai3d_waterpost evaluate --project <project> --variant base=<lodA> --variant new=<lodB> --regions <work>\apply\write\<块>\floaters.json --step 4 --out <csv>
```
