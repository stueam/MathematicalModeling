"""Independent checker using only Python integers. No hull generator is imported."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    info = d['info']
    scale = int(info['scale'])
    points = info['points']
    require(len(points) == 21, 'wrong station count')
    require(all(isinstance(x, int) and isinstance(y, int) for x, y in points), 'noninteger station')
    S = [(x * scale, y * scale) for x, y in points]
    leaves = {tuple(item[:4]): item for item in d['leaves']}
    require(len(leaves) == len(d['leaves']), 'duplicate leaf')
    require(not d.get('unresolved'), 'unresolved cells')
    seen = set()
    outside = 0
    max_sq = 0
    stack = [(-2048 * scale, -2048 * scale, 2048 * scale, 2048 * scale, 0)]
    while stack:
        x0, y0, x1, y1, depth = stack.pop()
        key = (x0, y0, x1, y1)
        dx = max(x0, 0, -x1)
        dy = max(y0, 0, -y1)
        if dx * dx + dy * dy > (1800 * scale) ** 2:
            outside += 1
            continue
        if key in leaves:
            item = leaves[key]
            require(item[4] == depth, 'depth mismatch')
            h = [S[i] for i in item[5]]
            require(len(h) >= 3, 'insufficient hull points')
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for i, a in enumerate(h):
                b = h[(i + 1) % len(h)]
                require(a != b, 'zero edge')
                for c in h:
                    z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    require(z >= 0, 'nonconvex or wrong orientation')
                for c in corners:
                    z = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    require(z > 0, 'cell is not strictly within hull')
                    sq = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
                    require(sq < (999 * scale) ** 2, 'distance bound failed')
                    max_sq = max(max_sq, sq)
            seen.add(key)
        else:
            require(depth < info['max_depth'], f'uncovered region {key}')
            mx = (x0 + x1) // 2
            my = (y0 + y1) // 2
            require(x0 < mx < x1 and y0 < my < y1, 'degenerate subdivision')
            stack.extend(
                [
                    (x0, y0, mx, my, depth + 1),
                    (mx, y0, x1, my, depth + 1),
                    (x0, my, mx, y1, depth + 1),
                    (mx, my, x1, y1, depth + 1),
                ]
            )
    require(len(seen) == len(leaves), 'unexpected extra leaves')
    return {
        'passed': True,
        'verified_cells': len(seen),
        'outside_cells': outside,
        'integer_checks': True,
        'max_support_corner_distance_m': max_sq**0.5 / scale,
        'boundary_covered': True,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'certificate', nargs='?', default=str(Path(__file__).parent / 'proofs' / 's21_certificate.json')
    )
    ap.add_argument('--output')
    a = ap.parse_args()
    r = verify(a.certificate)
    txt = json.dumps(r, ensure_ascii=False, indent=2)
    print(txt)
    if a.output:
        Path(a.output).write_text(txt, encoding='utf-8')
