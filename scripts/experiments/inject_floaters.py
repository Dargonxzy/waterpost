"""在 LOD 副本的最细层注入合成漂浮簇（照片里不存在），用于检验漂浮簇判定与照片核验。

    python inject_floaters.py SRC_LOD_BIN DST_LOD_BIN OUT_JSON [--far N] [--near N] [--seed S]

- 未改动的文件用硬链接（同一 NTFS 卷），改动的叶节点写新文件（先写临时文件再替换，不会改到源文件）。
- 簇写进包含簇中心的叶节点的补齐区（记录数 < 补齐数时），文件大小与补齐数不变；更新叶头包围盒。
- far：离局部顶面 2~8 m 的悬空簇；near：离局部顶面 0.6~1.4 m、贴着树冠 / 屋顶的簇（几何上"贴近主体"）。
- OUT_JSON 记录每个注入簇的叶文件、点坐标（float32）、包围盒，供 check_injected.py 逐条核对。
"""
import argparse, glob, json, os, struct, sys
import numpy as np

REC = 32
HDR = 32


def read_leaf(path):
    data = open(path, "rb").read()
    assert data[:4] == b"gssp", path
    padded = struct.unpack("<I", data[4:8])[0]
    rec = np.frombuffer(data, dtype=np.uint8, offset=HDR, count=padded * REC).reshape(padded, REC)
    # 补齐记录：全 0（或只有语义字节 255）。从后往前数真实记录。
    body = rec.copy()
    body[:, 26] = 0
    nonzero = np.flatnonzero(body.any(axis=1))
    count = int(nonzero[-1]) + 1 if len(nonzero) else 0
    return data, padded, count, rec


def encode(xyz, sigma, rgb, opacity, semantic):
    out = np.zeros((len(xyz), REC), np.uint8)
    f32 = np.zeros((len(xyz), 4), np.float32)
    f32[:, :3] = xyz
    f32[:, 3] = sigma ** 2  # yy
    out[:, :16] = f32.view(np.uint8).reshape(-1, 16)
    halves = np.zeros((len(xyz), 5), np.float16)
    halves[:, 0] = sigma ** 2  # xx
    halves[:, 4] = sigma ** 2  # zz
    out[:, 16:26] = halves.view(np.uint8).reshape(-1, 10)
    out[:, 26] = semantic[0]
    out[:, 27] = semantic[1]
    out[:, 28:31] = rgb
    out[:, 31] = opacity
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("out")
    ap.add_argument("--far", type=int, default=20)
    ap.add_argument("--near", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if os.path.exists(a.dst):
        sys.exit("destination exists: " + a.dst)
    # 1) 硬链接整个 LOD
    for root, dirs, files in os.walk(a.src):
        rel = os.path.relpath(root, a.src)
        os.makedirs(os.path.join(a.dst, rel), exist_ok=True)
        for f in files:
            os.link(os.path.join(root, f), os.path.join(a.dst, rel, f))
    finest = sorted(glob.glob(os.path.join(a.src, "L*")), key=lambda p: int(os.path.basename(p)[1:]))[-1]
    level = os.path.basename(finest)
    report = {"level": level, "clusters": []}
    for block_dir in sorted(glob.glob(os.path.join(finest, "*"))):
        block = os.path.basename(block_dir)
        leaves = sorted(glob.glob(os.path.join(block_dir, "*.bin")))
        info = []
        pts = []
        for path in leaves:
            data, padded, count, rec = read_leaf(path)
            box = struct.unpack("<6f", data[8:32])
            xyz = rec[:count, :12].copy().view(np.float32).reshape(-1, 3)
            info.append({"path": path, "padded": padded, "count": count, "box": list(box), "free": padded - count})
            pts.append((xyz, rec[:count]))
        allxyz = np.concatenate([p[0] for p in pts])
        allrec = np.concatenate([p[1] for p in pts])
        lo, hi = allxyz[:, :2].min(0), allxyz[:, :2].max(0)
        # 1 m 顶面高度图
        cell = 1.0
        ij = np.floor((allxyz[:, :2] - lo) / cell).astype(np.int64)
        shape = ij.max(0) + 1
        top = np.full(shape, -np.inf)
        np.maximum.at(top, (ij[:, 0], ij[:, 1]), allxyz[:, 2])
        planned = []
        attempts = 0
        while len(planned) < a.far + a.near and attempts < 20000:
            attempts += 1
            kind = "far" if len(planned) < a.far else "near"
            c = lo + 15 + rng.random(2) * (hi - lo - 30)
            ci = np.floor((c - lo) / cell).astype(int)
            win = top[max(ci[0] - 2, 0):ci[0] + 3, max(ci[1] - 2, 0):ci[1] + 3]
            if not np.isfinite(win).sum() >= 20:
                continue  # 下方要有足够的重建表面（否则是"没有观测"，不在本实验范围）
            local_top = np.nanmax(np.where(np.isfinite(win), win, np.nan))
            if any(np.hypot(*(c - p["c"][:2])) < 8 for p in planned):
                continue
            h = rng.uniform(2.0, 8.0) if kind == "far" else rng.uniform(0.6, 1.4)
            n = int(rng.integers(36, 73))
            spread = rng.uniform(0.2, 0.4)
            offs = rng.normal(0, spread, (n, 3))
            offs[:, 2] = np.abs(offs[:, 2])  # 簇底在 local_top + h
            centre = np.array([c[0], c[1], local_top + h])
            xyz = (centre + offs).astype(np.float32)
            # 颜色取附近顶面点的颜色（像树冠 / 屋顶的碎片），尺度、不透明度取常见范围
            near = np.flatnonzero((np.abs(allxyz[:, 0] - c[0]) < 2) & (np.abs(allxyz[:, 1] - c[1]) < 2) & (allxyz[:, 2] > local_top - 2))
            src = allrec[rng.choice(near, n)] if len(near) else allrec[rng.integers(0, len(allrec), n)]
            sigma = rng.uniform(0.05, 0.15, n).astype(np.float32)
            records = encode(xyz, sigma, src[:, 28:31], rng.integers(130, 230, n).astype(np.uint8), (src[:, 26], src[:, 27]))
            # 叶节点：3D 包含中心且补齐区放得下，否则 XY 包含、z 上界最高的
            choice = None
            for k, lf in enumerate(info):
                b = lf["box"]
                if lf["free"] >= n and b[0] <= c[0] <= b[3] and b[1] <= c[1] <= b[4] and b[2] <= centre[2] <= b[5]:
                    choice = k
                    break
            if choice is None:
                cands = [k for k, lf in enumerate(info) if lf["free"] >= n and lf["box"][0] <= c[0] <= lf["box"][3] and lf["box"][1] <= c[1] <= lf["box"][4]]
                if not cands:
                    continue
                choice = max(cands, key=lambda k: info[k]["box"][5])
            info[choice]["free"] -= n
            planned.append({"c": centre, "kind": kind, "leaf": choice, "records": records, "xyz": xyz, "h": h, "local_top": float(local_top)})
        # 写叶节点
        by_leaf = {}
        for p in planned:
            by_leaf.setdefault(p["leaf"], []).append(p)
        for k, plist in by_leaf.items():
            lf = info[k]
            data, padded, count, rec = read_leaf(lf["path"])
            buf = bytearray(data)
            box = list(lf["box"])
            at = count
            for p in plist:
                m = len(p["records"])
                buf[HDR + at * REC:HDR + (at + m) * REC] = p["records"].tobytes()
                at += m
                box[0:3] = np.minimum(box[0:3], p["xyz"].min(0)).tolist()
                box[3:6] = np.maximum(box[3:6], p["xyz"].max(0)).tolist()
            buf[8:32] = struct.pack("<6f", *box)
            dst = os.path.join(a.dst, os.path.relpath(lf["path"], a.src))
            tmp = dst + ".tmp"
            open(tmp, "wb").write(bytes(buf))
            os.replace(tmp, dst)  # 替换目录项，源文件（硬链接的另一端）不变
        for i, p in enumerate(planned):
            report["clusters"].append({
                "block": block, "id": f"{block}#{i}", "kind": p["kind"], "points": len(p["xyz"]),
                "height_above_top": p["h"], "local_top": p["local_top"],
                "leaf": os.path.relpath(info[p["leaf"]]["path"], a.src).replace("\\", "/"),
                "low": p["xyz"].min(0).astype(float).tolist(), "high": p["xyz"].max(0).astype(float).tolist(),
                "xyz": p["xyz"].astype(float).tolist()})
        print(block, "injected", len(planned), "clusters", sum(len(p["xyz"]) for p in planned), "points")
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
