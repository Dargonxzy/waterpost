"""Check an output LOD against its input: manifest boxes/counts vs BIN headers, semantic bytes, top views.

python check_lod.py <input lod/bin> <output lod/bin> <png prefix>
"""
import json, os, sys, struct
import numpy as np

def walk(root, rel, base, out, seen):
    doc = json.load(open(os.path.join(root, rel), encoding='utf-8'))
    def node(n):
        c = n.get('content', {}).get('uri')
        if c:
            r = os.path.normpath(os.path.join(base, c)).replace('\\', '/')
            if r.endswith('.json'):
                if r not in seen:
                    seen.add(r); walk(root, r, os.path.dirname(r), out, seen)
            else:
                out.append((rel, r, n['boundingVolume']['box'], n.get('extras', {}), bool(n.get('children'))))
        for ch in n.get('children', []):
            node(ch)
    node(doc['root'])

def leaves(root):
    out = []
    for entry in ('tileset.json', 'tileset.source.json'):
        if os.path.exists(os.path.join(root, entry)):
            walk(root, entry, '', out, set())
    return out

def read_bin(path):
    b = np.fromfile(path, np.uint8)
    count = struct.unpack('<I', b[4:8].tobytes())[0]
    lo = np.frombuffer(b[8:20].tobytes(), '<f4'); hi = np.frombuffer(b[20:32].tobytes(), '<f4')
    rec = b[32:].reshape(-1, 32)
    return count, lo, hi, rec

src, dst, prefix = sys.argv[1:4]
changed = []
problems = 0
for manifest, uri, box, extras, has_children in leaves(dst):
    if not uri.endswith('.bin'):
        continue
    a = os.path.join(src, uri); b = os.path.join(dst, uri)
    if os.path.getsize(a) == os.path.getsize(b) and open(a, 'rb').read() == open(b, 'rb').read():
        continue
    count, lo, hi, rec = read_bin(b)
    pos = np.frombuffer(rec[:, :12].tobytes(), '<f4').reshape(-1, 3)
    real = rec.any(1) & ~((rec[:, :26] == 0).all(1) & (rec[:, 28:] == 0).all(1))
    c = np.array(box[:3]); h = np.abs(np.array(box[3:12]).reshape(3, 3)).sum(0)
    inside = np.all((pos[real] >= c - h - 1e-3) & (pos[real] <= c + h + 1e-3), axis=1)
    if extras.get('gsplatPointCount') not in (None, count) or not inside.all():
        problems += 1
        print('PROBLEM', manifest, uri, 'manifest count', extras.get('gsplatPointCount'), 'header', count, 'outside', int((~inside).sum()))
    changed.append(uri)
print('changed leaf references', len(changed), 'unique', len(set(changed)), 'problems', problems)

def top(root, uris, size=1400):
    pts = []; cols = []; sems = []
    for uri in uris:
        _, _, _, rec = read_bin(os.path.join(root, uri))
        real = ~((rec[:, :26] == 0).all(1) & (rec[:, 28:] == 0).all(1))
        rec = rec[real]
        pts.append(np.frombuffer(rec[:, :12].tobytes(), '<f4').reshape(-1, 3)); cols.append(rec[:, 28:31]); sems.append(rec[:, 26])
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(sems)

finest = sorted({u for _, u, _, _, ch in leaves(src) if u.endswith('.bin') and not ch and u.split('/')[0] == max((x.split('/')[0] for _, x, _, _, c in leaves(src) if x.endswith('.bin') and not c and x.split('/')[0].startswith('L')), key=lambda s: int(s[1:]))})
from PIL import Image
results = {}
for name, root in (('input', src), ('output', dst)):
    p, c, s = top(root, finest)
    results[name] = (p, c, s)
    print(name, 'finest points', len(p), 'semantic water', int((s == 9).sum()))
lo = results['input'][0][:, :2].min(0); hi = results['input'][0][:, :2].max(0)
scale = 1400 / max(hi - lo)
W, H = (np.ceil((hi - lo) * scale) + 1).astype(int)
for name, (p, c, s) in results.items():
    order = np.argsort(p[:, 2])
    x = ((p[order, 0] - lo[0]) * scale).astype(int).clip(0, W - 1); y = ((hi[1] - p[order, 1]) * scale).astype(int).clip(0, H - 1)
    img = np.zeros((H, W, 3), np.uint8); img[y, x] = c[order]
    Image.fromarray(img).save(prefix + '_' + name + '.png')
# changed: removed (red) and added (blue)
img = np.zeros((H, W, 3), np.uint8)
p, c, s = results['input']; x = ((p[:, 0] - lo[0]) * scale).astype(int).clip(0, W - 1); y = ((hi[1] - p[:, 1]) * scale).astype(int).clip(0, H - 1)
img[y, x] = (c * .35).astype(np.uint8)
key_in = {tuple(v) for v in np.round(results['input'][0], 4).tolist()} if len(results['input'][0]) < 3_000_000 else None
Image.fromarray(img).save(prefix + '_base.png')
