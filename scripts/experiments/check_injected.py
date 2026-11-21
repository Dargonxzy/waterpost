"""核对注入簇在输出 LOD 最细层里还剩多少点（按 float32 坐标逐条匹配），并汇总 write 的漂浮簇判定。

    python check_injected.py INJECTED_JSON OUTPUT_LOD_BIN WORK_DIR [OUT_JSON]
"""
import glob, json, os, struct, sys
from collections import Counter, defaultdict
import numpy as np


def leaf_xyz(path):
    data = open(path, "rb").read()
    padded = struct.unpack("<I", data[4:8])[0]
    rec = np.frombuffer(data, dtype=np.uint8, offset=32, count=padded * 32).reshape(padded, 32)
    body = rec.copy(); body[:, 26] = 0
    nz = np.flatnonzero(body.any(axis=1)); count = int(nz[-1]) + 1 if len(nz) else 0
    return rec[:count, :12].copy().view(np.float32).reshape(-1, 3)


def main():
    inj = json.load(open(sys.argv[1]))
    out_lod, work = sys.argv[2], sys.argv[3]
    level = inj["level"]
    cache = {}
    def keys(block):
        if block not in cache:
            s = set()
            for f in glob.glob(os.path.join(out_lod, level, block, "*.bin")):
                for p in leaf_xyz(f):
                    s.add(p.tobytes())
            cache[block] = s
        return cache[block]
    # 注入簇 ↔ write 判定的中等簇：按包围盒重叠匹配
    decisions = {}
    for f in glob.glob(os.path.join(work, "apply", "write", "*", "floaters.json")):
        d = json.load(open(f))
        decisions[d["block"]] = d["candidates"]
    rows = []
    for c in inj["clusters"]:
        s = keys(c["block"])
        xyz = np.array(c["xyz"], np.float32)
        left = sum(p.tobytes() in s for p in xyz)
        match = None
        for cand in decisions.get(c["block"], []):
            lo, hi = np.array(cand["low"]), np.array(cand["high"])
            if np.all(np.array(c["low"]) <= hi + 1e-3) and np.all(np.array(c["high"]) >= lo - 1e-3):
                match = cand
                break
        rows.append({"id": c["id"], "kind": c["kind"], "points": c["points"], "left": int(left),
                     "geometry": match["geometry"] if match else "not_a_candidate(merged_or_small)",
                     "photo": match["photo"] if match else "",
                     "decision": match["decision"] if match else "",
                     "gap": match["gap"] if match else None, "clearance": match["clearance"] if match else None,
                     "candidate_points": match["points"] if match else None})
    summary = defaultdict(Counter)
    for r in rows:
        summary[r["kind"]]["clusters"] += 1
        summary[r["kind"]]["points"] += r["points"]
        summary[r["kind"]]["points_left"] += r["left"]
        summary[r["kind"]]["fully_removed"] += r["left"] == 0
        summary[r["kind"]]["geometry_remove"] += r["geometry"].startswith("remove_")
        summary[r["kind"]]["geom:" + r["geometry"]] += 1
        summary[r["kind"]]["photo:" + r["photo"]] += 1
    # 真实（未注入）中等簇的处理
    injected_boxes = [(c["block"], np.array(c["low"]), np.array(c["high"])) for c in inj["clusters"]]
    real = Counter()
    real_removed_points = 0
    for block, cands in decisions.items():
        for cand in cands:
            lo, hi = np.array(cand["low"]), np.array(cand["high"])
            if any(b == block and np.all(l <= hi + 1e-3) and np.all(h >= lo - 1e-3) for b, l, h in injected_boxes):
                continue
            real["candidates"] += 1
            real["geom:" + cand["geometry"]] += 1
            real["final:" + cand["decision"]] += 1
            if cand["decision"].startswith("remove_"):
                real_removed_points += cand["points"]
    result = {"by_kind": {k: dict(v) for k, v in summary.items()}, "real_candidates": dict(real),
              "real_points_removed": real_removed_points, "clusters": rows}
    if len(sys.argv) > 4:
        json.dump(result, open(sys.argv[4], "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "clusters"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
