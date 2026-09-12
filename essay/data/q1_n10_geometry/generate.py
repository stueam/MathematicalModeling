"""Reproducible Q1 experiment: 10 detectors, two halfplane algorithms, plots.

Run: python q1_n10_test.py
Dependencies: numpy, scipy, matplotlib. All outputs default to this data folder.
Uniform angular error is a simulation choice, not an assumed official law.
The estimator receives only detector coordinates and measured bearings.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle, ConnectionPatch

TOL = 1e-8  # metres, because every halfplane normal has unit length
PARALLEL = 1e-12
INK, MUTED = '#263748', '#687888'
BLUE, GREEN, PURPLE, RED = '#3275A8', '#278574', '#7864A2', '#BA543E'


def cross(u, v):
    return float(u[0]*v[1]-u[1]*v[0])


def normalize(A, b):
    A, b = np.asarray(A, float), np.asarray(b, float)
    if A.ndim != 2 or A.shape[1] != 2 or b.shape != (len(A),):
        raise ValueError('Halfplanes must have shapes (k,2) and (k,).')
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError('Non-finite halfplane input.')
    lengths = np.linalg.norm(A, axis=1)
    if np.any(lengths <= PARALLEL):
        raise ValueError('Zero halfplane normal.')
    return A/lengths[:, None], b/lengths


def wedge_constraints(detectors, bearings_deg, error_deg=1.):
    """Return unit-normal inequalities A*x <= b for the closed wedges."""
    p, theta = np.asarray(detectors, float), np.deg2rad(bearings_deg)
    if p.shape != (len(theta), 2) or not 0 < error_deg < 90:
        raise ValueError('Invalid detector coordinates or angular error.')
    eps = np.deg2rad(error_deg)
    lo = np.column_stack((np.cos(theta-eps), np.sin(theta-eps)))
    hi = np.column_stack((np.cos(theta+eps), np.sin(theta+eps)))
    A = np.empty((2*len(p), 2))
    A[0::2] = np.column_stack((lo[:, 1], -lo[:, 0]))
    A[1::2] = np.column_stack((-hi[:, 1], hi[:, 0]))
    return normalize(A, np.einsum('ij,ij->i', A, np.repeat(p, 2, axis=0)))


def region_status(A, b, tol=TOL):
    """Classify empty/unbounded/degenerate/polygon before finite geometry."""
    feasible = linprog([0., 0.], A_ub=A, b_ub=b, bounds=[(None, None)]*2,
                       method='highs')
    if feasible.status == 2:
        return 'empty'
    if not feasible.success:
        raise RuntimeError(f'Feasibility LP failed: {feasible.message}')
    # A recession direction exists iff the unit normals fit a closed semicircle.
    angles = np.sort(np.mod(np.arctan2(A[:, 1], A[:, 0]), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    if gaps.max() >= np.pi-PARALLEL:
        return 'unbounded'
    center = linprog([0., 0., -1.], A_ub=np.column_stack((A, np.ones(len(A)))),
                     b_ub=b, bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
    if not center.success:
        raise RuntimeError(f'Interior LP failed: {center.message}')
    return 'polygon' if center.x[2] > tol else 'degenerate'


def convex_hull(points, tol=TOL):
    """Monotone chain, with explicit point/segment support."""
    unique = []
    for q in sorted(np.asarray(points).reshape(-1, 2).tolist()):
        q = np.array(q)
        if not unique or np.linalg.norm(q-unique[-1]) > tol:
            unique.append(q)
    if len(unique) <= 2:
        return np.asarray(unique).reshape(-1, 2)
    def chain(sequence):
        result = []
        for q in sequence:
            while len(result) >= 2:
                u, v = result[-1]-result[-2], q-result[-1]
                if cross(u, v) > tol*max(1., np.linalg.norm(u), np.linalg.norm(v)):
                    break
                result.pop()
            result.append(q)
        return result
    return np.array(chain(unique)[:-1]+chain(unique[::-1])[:-1])


def intersect_lines(n1, b1, n2, b2):
    det = cross(n1, n2)
    if abs(det) < PARALLEL:
        raise ValueError('Parallel supporting lines.')
    return np.array([(b1*n2[1]-n1[1]*b2)/det,
                     (n1[0]*b2-b1*n2[0])/det])


def enumerate_polygon(A, b, tol=TOL):
    """O(k^3) candidate enumeration; requires a bounded feasible region."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    candidates = []
    for i, j in combinations(range(len(A)), 2):
        if abs(cross(A[i], A[j])) < PARALLEL:
            continue
        q = intersect_lines(A[i], b[i], A[j], b[j])
        if np.all(A@q <= b+tol):
            candidates.append(q)
    return convex_hull(candidates, tol)


def halfplane_deque(A, b, tol=TOL):
    """O(k log k) sorted halfplane intersection for full-dimensional polygons.

    Feasibility/boundedness classification is separate. No artificial bounding
    box and no fallback to the enumeration algorithm are used here.
    """
    A, b = np.asarray(A, float), np.asarray(b, float)
    directions = np.column_stack((-A[:, 1], A[:, 0]))
    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2*np.pi)
    order = np.argsort(angles, kind='stable')
    lines = []
    for i in order:
        if lines and abs(cross(A[lines[-1]], A[i])) < PARALLEL and A[lines[-1]]@A[i] > 0:
            if b[i] < b[lines[-1]]:
                lines[-1] = i
        else:
            lines.append(i)
    if len(lines) > 1 and abs(cross(A[lines[0]], A[lines[-1]])) < PARALLEL and A[lines[0]]@A[lines[-1]] > 0:
        first, last = lines[0], lines.pop()
        lines[0] = first if b[first] <= b[last] else last
    q = deque()
    def point(i, j):
        return intersect_lines(A[i], b[i], A[j], b[j])
    def outside(k, x):
        return A[k]@x > b[k]+tol
    for k in lines:
        while len(q) > 1 and outside(k, point(q[-2], q[-1])):
            q.pop()
        while len(q) > 1 and outside(k, point(q[0], q[1])):
            q.popleft()
        if q and abs(cross(A[q[-1]], A[k])) < PARALLEL:
            raise RuntimeError('Unexpected parallel active lines; check region status/tolerance.')
        q.append(k)
    while len(q) > 2 and outside(q[0], point(q[-2], q[-1])):
        q.pop()
    while len(q) > 2 and outside(q[-1], point(q[0], q[1])):
        q.popleft()
    if len(q) < 3:
        raise RuntimeError('No full-dimensional bounded polygon after deque closure.')
    indices = list(q)
    vertices = np.array([point(i, j) for i, j in zip(indices, indices[1:]+indices[:1])])
    # Remove consecutive duplicate intersections (multiple boundaries at a vertex).
    kept = []
    for v in vertices:
        if not kept or np.linalg.norm(v-kept[-1]) > tol:
            kept.append(v)
    if len(kept) > 1 and np.linalg.norm(kept[0]-kept[-1]) <= tol:
        kept.pop()
    return np.asarray(kept)


def polygon_area(v):
    if len(v) < 3:
        return 0.
    # Translate before shoelace to limit cancellation far from the origin.
    w = v-v[0]
    return abs(np.sum(w[:, 0]*np.roll(w[:, 1], -1)-w[:, 1]*np.roll(w[:, 0], -1)))/2


def diameter(v):
    distances = np.sum((v[:, None]-v[None, :])**2, axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    return float(np.sqrt(distances[i, j])), int(i), int(j)


def minimum_circle(v, tol=TOL):
    """Independent covering test: exhaust all 1/2/3-point support circles."""
    candidates = [(p, 0.) for p in v]
    for p, q in combinations(v, 2):
        candidates.append(((p+q)/2, np.linalg.norm(p-q)/2))
    for p, q, r in combinations(v, 3):
        u, w = q-p, r-p
        if abs(cross(u, w)) <= 1e-12*max(1., np.linalg.norm(u)*np.linalg.norm(w)):
            continue
        offset = np.linalg.solve(2*np.array([u, w]), [u@u, w@w])
        candidates.append((p+offset, np.linalg.norm(offset)))
    for center, radius in sorted(candidates, key=lambda c: c[1]):
        if np.max(np.linalg.norm(v-center, axis=1)) <= radius+tol:
            return np.asarray(center), float(radius)
    raise RuntimeError('No enclosing circle found.')


def geometry_gap(v, w):
    dist = cdist(v, w)
    return float(max(dist.min(axis=0).max(), dist.min(axis=1).max()))


def generate_case(seed, n=10):
    """IID Cartesian detector samples, conditioned only on valid reception.

    Proposals are uniform in [-750,750]^2. Only distances outside 100--950 m
    are rejected; there is no angular stratification, angular sorting, minimum
    separation, surrounding constraint or outcome-based case selection.
    """
    rng = np.random.default_rng(seed)
    source = rng.uniform(-200, 200, 2)
    points=[]
    for _ in range(10000):
        proposal=rng.uniform(-750,750,2)
        if 100 <= np.linalg.norm(proposal-source) <= 950:
            points.append(proposal)
        if len(points)==n:
            break
    if len(points)!=n:
        raise RuntimeError('Insufficient in-range detector positions.')
    p=np.asarray(points)
    distances=np.linalg.norm(p-source,axis=1)
    truth = np.mod(np.degrees(np.arctan2(source[1]-p[:, 1], source[0]-p[:, 0])), 360)
    errors = rng.uniform(-1., 1., n)
    observed = np.mod(truth+errors, 360)
    assert np.linalg.norm(source) < 1800 and np.all(distances < 1000)
    return dict(source=source, detectors=p, truth=truth, errors=errors, bearings=observed)


def direct_angle_audit(case, solved):
    """Check atan2 angular errors independently of the halfplane formula."""
    def errors_at(points):
        displacement=np.asarray(points)[:,None,:]-case['detectors'][None,:,:]
        actual=np.degrees(np.arctan2(displacement[:,:,1],displacement[:,:,0]))
        return (actual-case['bearings'][None,:]+180)%360-180
    v=solved['vertices']
    errors=errors_at(v)
    source_errors=errors_at(case['source'][None,:])
    edge=np.roll(v,-1,axis=0)-v
    outward=np.column_stack((edge[:,1],-edge[:,0]))/np.linalg.norm(edge,axis=1)[:,None]
    edge_probes=(v+np.roll(v,-1,axis=0))/2+.001*outward
    probe_errors=np.abs(errors_at(edge_probes)).max(axis=1)
    assert np.abs(errors).max()<=1+1e-8
    assert np.abs(source_errors).max()<=1+1e-8
    assert np.all(probe_errors>1+1e-8)
    return dict(vertex_max_absolute_angle_error_deg=float(np.abs(errors).max()),
                source_max_absolute_angle_error_deg=float(np.abs(source_errors).max()),
                outward_probe_distance_m=.001,
                outward_probe_max_error_deg=probe_errors.tolist(),
                all_vertices_satisfy_all_wedges=True,
                every_outward_edge_probe_violates_a_wedge=True)




def solve_case(case):
    A, b = wedge_constraints(case['detectors'], case['bearings'])
    status = region_status(A, b)
    if status != 'polygon':
        raise RuntimeError(f'Generated case has unexpected status: {status}')
    start = time.perf_counter_ns(); v = enumerate_polygon(A, b)
    brute_ms = (time.perf_counter_ns()-start)/1e6
    start = time.perf_counter_ns(); w = halfplane_deque(A, b)
    deque_ms = (time.perf_counter_ns()-start)/1e6
    gap = geometry_gap(v, w)
    D, i, j = diameter(v); D2, _, _ = diameter(w)
    center = (v[i]+v[j])/2
    rho = np.max(np.linalg.norm(v-center, axis=1))
    mec_center, mec_radius = minimum_circle(v)
    covered = bool(rho <= D/2+TOL)
    assert len(v) == len(w) and gap < 1e-6
    assert abs(D-D2) < 1e-6
    assert abs(polygon_area(v)-polygon_area(w)) < 1e-6
    assert np.max(A@v.T-b[:, None]) < 1e-6
    source_residual = float(np.max(A@case['source']-b))
    assert source_residual <= TOL
    assert mec_radius >= D/2-TOL
    assert covered == bool(mec_radius <= D/2+TOL)
    metrics = dict(vertices=len(v), area_m2=float(polygon_area(v)), diameter_m=D,
        vertex_gap_m=gap, diameter_gap_m=abs(D-D2),
        area_gap_m2=abs(float(polygon_area(v)-polygon_area(w))),
        source_max_constraint_residual_m=source_residual,
        diameter_circle_radius_m=D/2, max_vertex_radius_m=float(rho),
        coverage_excess_m=float(rho-D/2), coverage_ratio=float(rho/(D/2)),
        covered=covered, minimum_circle_radius_m=mec_radius,
        enumerate_ms=brute_ms, deque_ms=deque_ms)
    return dict(A=A, b=b, vertices=v, other_vertices=w, center=center,
                diameter_indices=(i, j), mec_center=mec_center, metrics=metrics)


def boundary_checks(case, solved):
    checks = []
    def record(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(dict(check=name, passed=True))
    square_A=np.array([[1,0],[-1,0],[0,1],[0,-1],[1,0]],float)
    square_b=np.array([1,1,1,1,2],float)
    record('parallel_duplicate_constraints', geometry_gap(
        enumerate_polygon(square_A,square_b),halfplane_deque(square_A,square_b))<TOL)
    record('empty_region', region_status(square_A[:4],[-1,-1,1,1])=='empty')
    record('unbounded_region', region_status(np.array([[-1.,0],[0,-1.]]),[0,0])=='unbounded')
    for name, bounds, count in [('line_segment',[1,0,0,0],2),('single_point',[0,0,0,0],1)]:
        record(name,region_status(square_A[:4],bounds)=='degenerate' and
               len(enumerate_polygon(square_A[:4],bounds))==count)
    p=case['detectors']; source=case['source']
    for name, errors in [('zero_error',0.),('positive_error_boundary',1.),('negative_error_boundary',-1.)]:
        A,b=wedge_constraints(p,np.mod(case['truth']+errors,360))
        record(name,np.max(A@source-b)<TOL)
        if region_status(A,b)=='polygon':
            record(name+'_algorithm_agreement',geometry_gap(enumerate_polygon(A,b),halfplane_deque(A,b))<1e-6)
    p0=np.array([[0.,0.]])
    A,b=wedge_constraints(p0,[359.8])
    x=100*np.array([np.cos(np.deg2rad(.2)),np.sin(np.deg2rad(.2))])
    record('angle_wrap_360',np.max(A@x-b)<TOL)
    A,b=solved['A'],solved['b']; v=solved['vertices']
    interior=linprog([0.,0.,-1.],A_ub=np.column_stack((A,np.ones(len(A)))),
                     b_ub=b,bounds=[(None,None),(None,None),(0,None)],method='highs')
    independent=HalfspaceIntersection(np.column_stack((A,-b)),interior.x[:2])
    qhull_gap=geometry_gap(v,independent.intersections)
    record('independent_qhull_intersection',qhull_gap<1e-6)
    checks[-1]['vertex_gap_m']=qhull_gap
    permutation=np.random.default_rng(42).permutation(len(A))
    record('constraint_order_invariance',geometry_gap(v,halfplane_deque(A[permutation],b[permutation]))<1e-6)
    for tol in [1e-9,1e-7]:
        record(f'tolerance_{tol:g}',geometry_gap(v,halfplane_deque(A,b,tol))<1e-6)
    weights=np.random.default_rng(73).dirichlet(np.ones(len(v)),size=1000)
    samples=weights@v
    record('convex_samples_inside',np.max(A@samples.T-b[:,None])<1e-6)
    record('sample_distances_bounded_by_D',cdist(samples[::2],samples[1::2]).max()<=solved['metrics']['diameter_m']+TOL)
    # Independent shape fixtures: same diameter, different covering properties.
    for name, height, expected in [('equilateral',np.sqrt(3),False),('obtuse',1/np.sqrt(3),True)]:
        tri=np.array([[-1.,0],[1,0],[0,height]])
        _,radius=minimum_circle(tri)
        record(name+'_covering',bool(radius<=1+TOL)==expected)
    return checks


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def figure_style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei'],
        'font.size':11,'mathtext.fontset':'stix','axes.unicode_minus':False,
        'axes.edgecolor':'#CCD5DC','axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def save_figure(fig, output, name):
    for extension in ['png']:
        fig.savefig(output/f'{name}.{extension}',dpi=300)
    plt.close(fig)


def draw_geometry(case, solved, output):
    figure_style()
    p,theta=case['detectors'],np.deg2rad(case['bearings'])
    v,center=solved['vertices'],solved['mec_center']; metric=solved['metrics']
    radius=metric['minimum_circle_radius_m']
    assert np.max(np.linalg.norm(v-center,axis=1))<=radius+TOL
    fig=plt.figure(figsize=(12.2,6.1))
    overview=fig.add_axes([.055,.14,.43,.72]); zoom=fig.add_axes([.57,.14,.375,.72])
    fig.text(.055,.925,'(a)  整体俯瞰',fontsize=14,weight='bold',color=INK)
    fig.text(.57,.925,'(b)  定位区域与最小覆盖圆',fontsize=14,weight='bold',color=INK)
    colors=[BLUE,GREEN,PURPLE]
    for ax in [overview,zoom]:
        ax.set_aspect('equal');ax.set_axis_off()
        ray_records=zip(p,theta) if ax is overview else []
        for i,(point,angle) in enumerate(ray_records):
            color=colors[i%3]
            ends=np.array([point+1900*np.array([np.cos(angle+s),np.sin(angle+s)]) for s in [-np.pi/180,np.pi/180]])
            if ax is overview:
                ax.add_patch(Polygon([point,*ends],facecolor=color,alpha=.025,edgecolor='none'))
            for end in ends:
                ax.plot(*np.array([point,end]).T,color=color,lw=.65,alpha=.25 if ax is overview else .30,zorder=1)
            if ax is overview:
                end=point+1900*np.array([np.cos(angle),np.sin(angle)])
                ax.plot(*np.array([point,end]).T,color=color,lw=.65,ls=(0,(5,5)),alpha=.36)
        ax.add_patch(Polygon(v,facecolor='#EFC2AC',edgecolor=RED,lw=1.8,zorder=4))
        ax.scatter(*case['source'],marker='*',s=90 if ax is overview else 115,
                   color=INK,edgecolor='white',lw=.7,zorder=9)
    span=np.ptp(p,axis=0).max(); mid=(p.min(axis=0)+p.max(axis=0))/2
    overview.set(xlim=(mid[0]-.61*span,mid[0]+.61*span),ylim=(mid[1]-.61*span,mid[1]+.61*span))
    for i,point in enumerate(p):
        color=colors[i%3]
        overview.scatter(*point,s=35,color=color,edgecolor='white',lw=.8,zorder=6)
        away=(point-mid)/np.linalg.norm(point-mid)
        overview.annotate(rf'$\mathbf{{p}}_{{{i+1}}}$',point,xytext=tuple(away*13),
                          textcoords='offset points',ha='center',va='center',fontsize=11,color=color)
    # Same metre scale along both axes; circle must appear circular.
    width=2*radius*1.45; middle=center
    zoom.set(xlim=(middle[0]-width/2,middle[0]+width/2),ylim=(middle[1]-width/2,middle[1]+width/2))
    zoom.add_patch(Circle(center,radius,facecolor='#EFF7F3',edgecolor='none',zorder=2))
    zoom.add_patch(Circle(center,radius,fill=False,ec=GREEN,lw=1.8,zorder=5))
    zoom.scatter(*v.T,s=26,color=RED,edgecolor='white',lw=.6,zorder=7)
    zoom.scatter(*center,s=32,color=GREEN,marker='+',lw=1.2,zorder=7)
    zoom.annotate(r'$O_*$',center,xytext=(-12,10),textcoords='offset points',
                  ha='center',fontsize=11,color=GREEN,zorder=8)
    # Arrange labels in two columns; vertices separated by millimetres remain distinct.
    for side in [-1,1]:
        indices=[k for k,point in enumerate(v) if (-1 if point[0]<v[:,0].mean() else 1)==side]
        indices.sort(key=lambda k:v[k,1])
        original_y=np.array([v[k,1] for k in indices])
        placed_y=original_y.copy()
        for k in range(1,len(placed_y)):
            placed_y[k]=max(placed_y[k],placed_y[k-1]+width*.062)
        placed_y-=np.mean(placed_y-original_y)
        label_x=(v[:,0].min() if side<0 else v[:,0].max())+side*.105*width
        for k,y in zip(indices,placed_y):
            zoom.annotate(rf'$\mathbf{{v}}_{{{k+1}}}$',v[k],xytext=(label_x,y),
                          ha='center',va='center',fontsize=10.5,color=INK,
                          arrowprops=dict(arrowstyle='-',color='#8596A2',lw=.65),
                          bbox=dict(facecolor='white',ec='none',pad=1,alpha=.9),zorder=8)
    # One simple scale bar on the local plot, avoiding a table of numbers in the figure.
    exponent=10**np.floor(np.log10(width/5))
    length=max(1.,np.floor(width/5/exponent))*exponent
    x=middle[0]-.40*width; y=middle[1]-.40*width
    zoom.plot([x,x+length],[y,y],color=INK,lw=2,zorder=8)
    zoom.text(x+length/2,y+.025*width,f'{length:g} m',ha='center',fontsize=9,color=MUTED)
    focus=width/2
    overview.add_patch(Rectangle(center-focus,2*focus,2*focus,fill=False,ec=MUTED,lw=.8,zorder=8))
    fig.add_artist(ConnectionPatch(xyA=tuple(center+[focus,focus]),coordsA=overview.transData,
                   xyB=(.02,.84),coordsB=zoom.transAxes,color='#AAB5BD',lw=.8,ls=(0,(3,4)),zorder=0))
    handles=[Line2D([0],[0],marker='*',ls='none',color=INK,markersize=10,label='真实源（仅用于验证）'),
             Polygon([[0,0]],facecolor='#EFC2AC',edgecolor=RED,label='定位区域'),
             Line2D([0],[0],color=GREEN,lw=1.8,label='最小覆盖圆')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=len(handles),
               frameon=False,fontsize=9.5,columnspacing=2.1)
    save_figure(fig,output,'q1_n10_geometry')


def draw_validation(rows, output):
    figure_style()
    D=np.array([r['diameter_m'] for r in rows]); covered=np.array([r['covered'] for r in rows])
    ratio=np.array([r['coverage_ratio'] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    fig.subplots_adjust(left=.08,right=.97,bottom=.18,top=.82,wspace=.28)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#E5EBEF',lw=.7,zorder=0);ax.set_axisbelow(True)
    bins=np.histogram_bin_edges(D,bins='auto')
    axes[0].hist([D[covered],D[~covered]],bins=bins,stacked=True,color=[GREEN,'#D99580'],
                 edgecolor='white',lw=.8,label=['直径圆可覆盖','直径圆不可覆盖'])
    axes[0].set(xlabel='定位区域直径 / m',ylabel='算例数')
    axes[0].set_title('(a)  区域直径分布',loc='left',fontsize=13,color=INK,pad=16)
    axes[0].legend(frameon=False,fontsize=9)
    for mask,color,label in [(covered,GREEN,'可覆盖'),(~covered,RED,'不可覆盖')]:
        axes[1].scatter(D[mask],ratio[mask],s=24,color=color,alpha=.7,edgecolor='white',lw=.35,zorder=3)
    axes[1].axhline(1,color=INK,ls=(0,(5,3)),lw=1,zorder=2)
    axes[1].set(xlabel='定位区域直径 / m',ylabel=r'覆盖比 $\rho/(D/2)$')
    axes[1].set_ylim(.99,max(1.04,ratio.max()+.02))
    axes[1].set_title('(b)  直径圆覆盖判定',loc='left',fontsize=13,color=INK,pad=16)
    fig.text(.08,.94,f'{len(rows)} 组随机验证 · 每组 10 个检测点',fontsize=15,weight='bold',color=INK)
    save_figure(fig,output,'q1_n10_validation')




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--trials',type=int,default=200)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    case=generate_case(args.seed);solved=solve_case(case)
    angle_audit=direct_angle_audit(case,solved)
    checks=boundary_checks(case,solved)
    rows=[]
    for trial in range(args.trials):
        # The displayed case is trial 0; no selection by shape or covering outcome.
        seed=args.seed+trial
        current=solved if trial==0 else solve_case(generate_case(seed))
        rows.append(dict(trial=trial,seed=seed,**current['metrics']))
    benchmark={}
    for name,algorithm in [('enumerate',enumerate_polygon),('deque',halfplane_deque)]:
        times=[]
        for _ in range(30):
            start=time.perf_counter_ns();algorithm(solved['A'],solved['b'])
            times.append((time.perf_counter_ns()-start)/1e6)
        benchmark[name+'_median_ms']=float(np.median(times))
    v=solved['vertices']; i,j=solved['diameter_indices']
    summary=dict(seed=args.seed,n_detectors=10,error_bound_deg=1.,n_trials=args.trials,
        data_kind='synthetic; independent Cartesian detectors conditioned on reception distance, uniform bounded errors',
        detector_sampling=dict(proposal_square_m=[-750,750],source_square_m=[-200,200],
                               accepted_distance_range_m=[100,950],angular_stratification=False),
        estimator_inputs='detector coordinates and measured bearings only',
        tolerance_m=TOL,main_case=solved['metrics'],benchmark_30_repeats=benchmark,
        true_source_m=case['source'].tolist(),diameter_endpoints_m=v[[i,j]].tolist(),
        diameter_circle_center_m=solved['center'].tolist(),
        minimum_circle_center_m=solved['mec_center'].tolist(),
        batch=dict(algorithm_agreement_count=len(rows),
            max_vertex_gap_m=max(r['vertex_gap_m'] for r in rows),
            max_diameter_gap_m=max(r['diameter_gap_m'] for r in rows),
            covered_count=sum(r['covered'] for r in rows),
            not_covered_count=sum(not r['covered'] for r in rows),
            diameter_min_m=min(r['diameter_m'] for r in rows),
            diameter_max_m=max(r['diameter_m'] for r in rows)),
        checks=checks,direct_angle_audit=angle_audit)
    write_csv(output/'q1_n10_observations.csv',[dict(detector=i+1,x_m=p[0],y_m=p[1],
        bearing_deg=case['bearings'][i],true_bearing_deg=case['truth'][i],
        error_deg=case['errors'][i],distance_to_source_m=np.linalg.norm(p-case['source']))
        for i,p in enumerate(case['detectors'])])
    write_csv(output/'q1_n10_vertices.csv',[dict(vertex=k+1,x_m=q[0],y_m=q[1],
        is_diameter_endpoint=k in [i,j]) for k,q in enumerate(v)])
    write_csv(output/'q1_n10_trials.csv',rows)
    (output/'q1_n10_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    draw_geometry(case,solved,output)
    print(json.dumps(dict(main_case=solved['metrics'],batch=summary['batch'],
                          checks_passed=len(checks)),indent=2))
    print('Outputs: q1_n10_geometry.png, CSV, JSON.')


if __name__=='__main__':
    main()
