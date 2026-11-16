"""Compare the Python water_postprocessing run with the C++ ai3d-waterpost run on one dataset.

python compare_python.py <project> <python out dir> <cpp work dir> <lod dir> <report.md>
Points are matched by exact float32 position (PLY rows of Python vs LOD finest records of C++).
"""
import glob, json, os, struct, sys
import numpy as np

project, pyout, work, lod, report_path = sys.argv[1:6]

def latest(pattern):
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None

def read_ply_xyz(path):
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        lines = header.decode('latin1').splitlines()
        count = 0; props = []; in_vertex = False
        for line in lines:
            p = line.split()
            if p[:2] == ['element', 'vertex']: count = int(p[2]); in_vertex = True
            elif p and p[0] == 'element': in_vertex = False
            elif in_vertex and p and p[0] == 'property': props.append((p[2], p[1]))
        types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2', 'char': 'i1'}
        dtype = np.dtype([(n, types[t]) for n, t in props])
        data = np.fromfile(f, dtype=dtype, count=count)
    return np.column_stack([data['x'], data['y'], data['z']]).astype('<f4')

def read_lod_block(block):
    files = sorted(glob.glob(os.path.join(lod, 'L*', block, '*.bin')), key=lambda p: (int(os.path.basename(os.path.dirname(os.path.dirname(p)))[1:]), p))
    level = max(int(os.path.basename(os.path.dirname(os.path.dirname(p)))[1:]) for p in files)
    # Same order as the C++ reader: finest-level leaves sorted by URI.
    uris = sorted(os.path.relpath(p, lod).replace('\\', '/') for p in files if f'/L{level}/' in p.replace('\\', '/'))
    parts = []
    for uri in uris:
        b = np.fromfile(os.path.join(lod, uri), np.uint8); rec = b[32:].reshape(-1, 32)
        pad = (rec[:, :26] == 0).all(1) & (rec[:, 28:] == 0).all(1)
        n = len(rec)
        while n and pad[n - 1]: n -= 1
        parts.append(np.frombuffer(rec[:n, :12].tobytes(), '<f4').reshape(-1, 3))
    return np.concatenate(parts)

def mapping(src, dst):
    """For each src row, the dst row with identical float32 xyz (or -1)."""
    a = np.ascontiguousarray(src).view(np.uint32).reshape(-1, 3).astype(np.uint64)
    b = np.ascontiguousarray(dst).view(np.uint32).reshape(-1, 3).astype(np.uint64)
    ka = (a[:, 0] << 32) | a[:, 1]; kb = (b[:, 0] << 32) | b[:, 1]
    k = np.concatenate([ka, kb]); z = np.concatenate([a[:, 2], b[:, 2]])
    tag = np.r_[np.zeros(len(a), np.int8), np.ones(len(b), np.int8)]
    idx = np.r_[np.arange(len(a)), np.arange(len(b))]
    order = np.lexsort((tag, z, k))
    k, z, tag, idx = k[order], z[order], tag[order], idx[order]
    result = np.full(len(a), -1, np.int64)
    same = (k[:-1] == k[1:]) & (z[:-1] == z[1:]) & (tag[:-1] == 0) & (tag[1:] == 1)
    result[idx[:-1][same]] = idx[1:][same]
    return result

def jaccard(a, b):
    a = set(map(int, a)); b = set(map(int, b))
    u = len(a | b)
    return (len(a & b) / u if u else 1.0), len(a), len(b), len(a & b)

def blocks():
    root = os.path.join(project, 'artifacts', 'blocks')
    for name in ('valid_chunk_list.txt', 'valid_chunks.txt'):
        p = os.path.join(root, name)
        if os.path.exists(p):
            lines = open(p, encoding='utf-8-sig').read().split('\n')[1:]
            return sorted(l.split()[0] for l in lines if l.strip())

lines = ['# Python vs C++ 对比：' + os.path.basename(project.rstrip('\\/')), '']
pyrep = json.load(open(latest(os.path.join(pyout, 'reconcile', 'attempts', 'reconcile', 'global', '*', 'model', 'report.json')), encoding='utf-8'))
cprep = json.load(open(os.path.join(work, 'reconcile', 'report.json'), encoding='utf-8'))
pymodel = json.load(open(latest(os.path.join(pyout, 'apply', 'attempts', 'apply_model', 'global', '*', 'model', 'report.json')), encoding='utf-8'))
cpmodel = json.load(open(os.path.join(work, 'apply', 'model', 'report.json'), encoding='utf-8'))
spacing = cprep['global_grid']['spacing']; origin = np.array(cprep['global_grid']['origin'])

lines += ['## 水体与水位', '', '| 实现 | water_id | 水位 | σ | 锚点 | 接受 |', '|---|---|---|---|---|---|']
py_acc = {s['water_id'] for s in pymodel['surfaces']}; cp_acc = {s['water_id'] for s in cpmodel['surfaces']}
for name, rep, acc in (('Python', pyrep, py_acc), ('C++', cprep, cp_acc)):
    for s in sorted(rep['surfaces'], key=lambda s: -s['anchors']):
        lines.append(f"| {name} | {s['water_id']} | {s['height']:.3f} | {s.get('height_sigma', float('nan')):.3f} | {s['anchors']} | {'是' if s['water_id'] in acc else '否'} |")
lines += ['', f"岸线锚点总数：Python {sum(b['shore'] for b in pyrep['blocks'])}，C++ {sum(b['shore'] for b in cprep['blocks'])}", '']

lines += ['## 逐块', '', '| 块 | 项目 | Python | C++ | 共同 | Jaccard |', '|---|---|---|---|---|---|']
totals = {}
for block in blocks():
    ply = latest(os.path.join(project, 'artifacts', 'blocks', block, '3dgs', 'point_cloud', 'layer_*', '*_point_cloud_clip.ply'))
    pxyz = read_ply_xyz(ply); lxyz = read_lod_block(block)
    m = mapping(pxyz, lxyz)  # PLY row -> LOD row
    shared = int((m >= 0).sum())
    lines.append(f"| {block} | 点数（PLY / LOD / 位置相同） | {len(pxyz)} | {len(lxyz)} | {shared} | |")
    def py_rows(rows):
        rows = np.asarray(rows, np.int64); mm = m[rows]; return mm[mm >= 0], int((mm < 0).sum())
    det = latest(os.path.join(pyout, 'detect', 'attempts', 'detect', block, '*'))
    pdet = np.load(os.path.join(det, block + '.npz'))
    pprep = det + '/prepared'
    cdet = os.path.join(work, 'detect', block)
    cd = np.load(os.path.join(cdet, 'detections.npz'))
    for label, prow, crow in (
            ('水判定点', pdet['water_indices'], cd['water_indices']),
            ('拟合种子', np.load(pprep + '/water_evidence.npz')['water_indices'], np.load(cdet + '/water_evidence.npz')['water_indices']),
            ('岸线锚点', pdet['shore_indices'], cd['shore_indices'])):
        mapped, lost = py_rows(prow)
        j, a, b, c = jaccard(mapped, crow)
        lines.append(f"| {block} | {label} | {len(prow)}（{lost} 个不在 LOD） | {len(crow)} | {c} | {j:.3f} |")
    app = latest(os.path.join(pyout, 'apply', 'attempts', 'apply', block, '*'))
    pa = np.load(os.path.join(app, 'fine', 'detections', block + '_applied.npz'))
    ca = np.load(os.path.join(work, 'apply', 'write', block, 'applied.npz'))
    for label, key, ckey in (('删除总数', 'removed_indices', 'removed_indices'), ('投票删除', 'voted_removed_indices', 'voted_removed_indices'), ('体积清理', 'carved_indices', 'carved_indices')):
        mapped, lost = py_rows(pa[key])
        j, a, b, c = jaccard(mapped, ca[ckey])
        lines.append(f"| {block} | {label} | {len(pa[key])}（{lost} 个不在 LOD） | {len(ca[ckey])} | {c} | {j:.3f} |")
    pf = np.load(os.path.join(app, 'fine', 'footprint.npz'))
    pcells = np.rint((pf['xy'] - origin) / spacing).astype(np.int64)
    pkeys = set(map(tuple, np.column_stack([pcells, pf['labels']]).tolist()))
    cs = np.load(os.path.join(work, 'apply', 'samples', block, 'samples.npz'))
    acc = np.isin(cs['footprint_ids'], list(cp_acc))
    cxy = cs['footprint_cells'][acc]
    # Body IDs can differ between implementations; compare the cell sets.
    pset = set(map(tuple, pcells.tolist())); cset = set(map(tuple, cxy.tolist()))
    u = len(pset | cset)
    lines.append(f"| {block} | 足迹格（0.4 m） | {len(pset)} | {len(cset)} | {len(pset & cset)} | {len(pset & cset) / u if u else 1:.3f} |")
    prec = json.load(open(os.path.join(app, 'block.json'), encoding='utf-8'))
    crec = json.load(open(os.path.join(work, 'apply', 'write', block, 'record.json'), encoding='utf-8'))
    lines.append(f"| {block} | 新增水面高斯 | {prec.get('added')} | {crec['added']} | | |")
lines.append('')
open(report_path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
