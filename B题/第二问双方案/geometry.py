"""共用几何：保留孔洞和多部件；所有长度单位为米，角度为弧度。"""
import math
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from scipy.spatial.distance import pdist

TAU = 2 * math.pi


def classify(distances, receiving_radius):
    near = np.asarray(distances) <= 5
    none = np.asarray(distances) > receiving_radius
    return near, ~(near | none), none


def disk(s, radius, tol):
    # 内接正多边形的最大弓高不超过 tol。
    n = max(16, math.ceil(math.pi / math.acos(1 - min(tol / radius, .5))))
    return Point(s).buffer(radius, quad_segs=math.ceil(n / 4))


def wedge(s, theta, eps, reach):
    # 三角形远端弦在整个研究区域之外，因而不截短射线。
    r = reach / math.cos(eps) * 1.01
    return Polygon([s, (s[0] + r * math.cos(theta-eps), s[1] + r * math.sin(theta-eps)),
                    (s[0] + r * math.cos(theta+eps), s[1] + r * math.sin(theta+eps))])


def sector(s, ra, rb, a, b, tol):
    step = 2 * math.acos(1 - min(tol / rb, .5))
    angles = np.linspace(a, b, max(2, math.ceil((b-a)/step)+1))
    outer = np.column_stack((s[0]+rb*np.cos(angles), s[1]+rb*np.sin(angles)))
    inner = np.column_stack((s[0]+ra*np.cos(angles[::-1]), s[1]+ra*np.sin(angles[::-1])))
    return Polygon(np.concatenate((outer, inner)))


def diameter(region):
    """取完整几何区域的凸包仅用于最远点对，不用于概率面积。"""
    h = region.convex_hull
    if h.is_empty or h.geom_type == 'Point':
        return 0.
    xy = np.asarray(h.exterior.coords if h.geom_type == 'Polygon' else h.coords)
    return float(pdist(xy).max()) if len(xy) > 1 else 0.


def radius(region):
    return float(shapely.minimum_bounding_radius(region)) if not region.is_empty else 0.


def split_region(region):
    x0, y0, x1, y1 = region.bounds
    xm, ym = (x0+x1)/2, (y0+y1)/2
    return [p for p in (region.intersection(box(a,b,c,d)) for a,b,c,d in
            [(x0,y0,xm,ym), (xm,y0,x1,ym), (x0,ym,xm,y1), (xm,ym,x1,y1)]) if p.area > 0]


def bearing_contributions(phi, eps, width):
    """稀疏 (目标索引, 角度档索引, alpha)，包含环绕及误差卷积。"""
    n = round(TAU / width)
    start = np.floor((phi-eps)/width).astype(int)
    ids, bins, alphas = [], [], []
    for offset in range(math.ceil(2*eps/width)+2):
        k = start + offset
        overlap = np.maximum(0, np.minimum(phi+eps, (k+1)*width)-np.maximum(phi-eps, k*width))
        valid = overlap > 0
        ids.append(np.flatnonzero(valid))
        bins.append(k[valid] % n)
        alphas.append(overlap[valid]/(2*eps))
    return np.concatenate(ids), np.concatenate(bins), np.concatenate(alphas)


def first_region(c, rmax):
    s = (c['x1'], c['y1'])
    return (disk((0,0), c['L'], c['arc_tolerance_m'])
            .intersection(disk(s, rmax, c['arc_tolerance_m']))
            .intersection(wedge(s, math.radians(c['theta1_deg']), math.radians(c['epsilon_deg']), rmax*2))
            .difference(disk(s, 5, min(c['arc_tolerance_m'], .001))))


def cells_for(c, p1, rmax):
    s = (c['x1'], c['y1'])
    th, eps = math.radians(c['theta1_deg']), math.radians(c['epsilon_deg'])
    cells = []
    if c['source_discretization'] == 'square':
        step = c['source_grid_step']
        x0,y0,x1,y1 = p1.bounds
        for x in np.arange(math.floor(x0/step)*step, x1, step):
            for y in np.arange(math.floor(y0/step)*step, y1, step):
                p = p1.intersection(box(x,y,x+step,y+step))
                if p.area > 0:
                    cells.append(p)
    else:
        rr = np.sqrt(np.linspace(25, rmax*rmax, c['radial_bins']+1))
        aa = np.linspace(th-eps, th+eps, c['angular_bins']+1)
        for ra,rb in zip(rr[:-1],rr[1:]):
            for a,b in zip(aa[:-1],aa[1:]):
                p = p1.intersection(sector(s, ra, rb, a, b, c['arc_tolerance_m']*rb/rmax))
                if p.area > 0:
                    cells.append(p)
    # 独立圆弧折线的微小差异会留下边缘细条，必须补回而非丢弃。
    remainder = p1.difference(shapely.union_all(cells))
    if remainder.area > 0:
        cells.extend([p for p in shapely.get_parts(remainder) if p.area > 0])
    if not cells:
        raise ValueError('第一次观测的可行区域为空或零面积；请检查 S1、theta1 和接收半径。')
    return cells


def quadrature(c, cells, s2=None, rmax=None):
    """每个实际裁剪单元取内部点；必要时四分，子权重按真实面积守恒。"""
    leaves, xy, weights = [], [], []
    eps = math.radians(c['epsilon_deg'])
    width = math.radians(c['angle_bin_width_deg'])

    def visit(p, area, level):
        point = p.representative_point()
        x,y = point.x, point.y
        x0,y0,x1,y1 = p.bounds
        extent = math.hypot(x1-x0, y1-y0)
        refine = False
        if s2 is not None and c['adaptive_refinement'] and level < c['max_refinement_level']:
            d = math.hypot(x-s2[0], y-s2[1])
            # extent 是单元内任意点到代表点距离的保守上界。
            cross_ring = any(abs(d-r) <= extent for r in (5, rmax))
            angular_span = math.pi if d <= extent else 2*math.asin(min(1,extent/d))
            refine = cross_ring or angular_span > min(width,eps)
        if refine:
            children = split_region(p)
            total = sum(q.area for q in children)
            for q in children:
                visit(q, area*q.area/total, level+1)
        else:
            leaves.append(p)
            xy.append((x,y))
            weights.append(area)
    for p in cells:
        visit(p, p.area, 0)
    return leaves, np.asarray(xy), np.asarray(weights)
