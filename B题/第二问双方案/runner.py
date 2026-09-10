"""配置、粗到细多区域搜索、CSV/GeoJSON/PNG 交付。"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
from shapely.geometry import mapping
from models import Bayesian, FixedRadius
from geometry import TAU

DEFAULTS = dict(L=1800.,epsilon_deg=1.,R_min=1000.,R_max=1500.,credible_mass=.95,
                radial_bins=48,angular_bins=6,source_discretization='polar',source_grid_step=10.,
                adaptive_refinement=True,max_refinement_level=1,arc_tolerance_m=.03,
                angle_bin_width_deg=1.,candidate_grid_step=600.,search_domain='source_disk',
                rectangle=[-1800.,-1800.,1800.,1800.],refine_levels=2,refine_keep=4,
                near_optimal_relative_tolerance=.05,near_optimal_absolute_tolerance_m=.01,
                zero_objective_threshold_m=1e-8,seed=20260911,input_label='DEMO',plots=True)


def load_config(path, method, overrides=None):
    supplied = json.loads(Path(path).read_text(encoding='utf-8'))
    c = {**DEFAULTS,**supplied,**(overrides or {})}
    for key in ['x1','y1','theta1_deg']+(['R0'] if method == 'fixed' else []):
        if key not in c:
            raise ValueError(f'缺少必需输入 {key}；不能假定题目给定了具体数值。')
    unknown = set(c)-set(DEFAULTS)-{'x1','y1','theta1_deg','R0','output_dir'}
    if unknown:
        raise ValueError(f'未知配置项：{sorted(unknown)}')
    for key,value in c.items():
        if isinstance(value,(int,float)) and not math.isfinite(value):
            raise ValueError(f'{key} 必须有限')
    for key in ['L','epsilon_deg','source_grid_step','candidate_grid_step','arc_tolerance_m','angle_bin_width_deg']:
        if c[key] <= 0:
            raise ValueError(f'{key} 必须大于零')
    if c['epsilon_deg']+c['angle_bin_width_deg']/2 >= 90:
        raise ValueError('误差半宽加半档宽必须小于90度')
    if abs(360/c['angle_bin_width_deg']-round(360/c['angle_bin_width_deg'])) > 1e-9:
        raise ValueError('angle_bin_width_deg 必须整除360')
    for key in ['radial_bins','angular_bins','refine_keep','refine_levels','max_refinement_level']:
        if not isinstance(c[key],int) or c[key] < (0 if key in ['refine_levels','max_refinement_level'] else 1):
            raise ValueError(f'{key} 必须为合法整数')
    if c['search_domain'] not in ['source_disk','extended_disk','rectangle']:
        raise ValueError('search_domain 应为 source_disk、extended_disk 或 rectangle')
    if c['source_discretization'] not in ['polar','square']:
        raise ValueError('source_discretization 应为 polar 或 square')
    if len(c['rectangle']) != 4 or not all(math.isfinite(v) for v in c['rectangle']):
        raise ValueError('rectangle 应为四个有限数 [xmin,ymin,xmax,ymax]')
    a,b,d,e = c['rectangle']
    if d <= a or e <= b:
        raise ValueError('矩形边界顺序错误')
    for key in ['near_optimal_relative_tolerance','near_optimal_absolute_tolerance_m','zero_objective_threshold_m']:
        if c[key] < 0:
            raise ValueError(f'{key} 不能为负')
    if not 0 < c['credible_mass'] <= 1 or not 5 < c['R_min'] < c['R_max']:
        raise ValueError('需要 0 < credible_mass <= 1 及 5 < R_min < R_max')
    if method == 'fixed' and not 1000 <= c['R0'] <= 1500:
        raise ValueError('题设条件下 R0 应在 [1000,1500] 米内')
    c.setdefault('output_dir',f'results/{method}')
    return c


def search(model, progress=True):
    c = model.c
    limit = c['L']+(model.rmax if c['search_domain']=='extended_disk' else 0)
    bounds = c['rectangle'] if c['search_domain']=='rectangle' else [-limit,-limit,limit,limit]
    def inside(x,y):
        if c['search_domain'] == 'rectangle':
            return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
        return math.hypot(x,y) <= limit+1e-9
    rows,seen = [],set()
    start = time.perf_counter()
    def evaluate_batch(points,level,spacing):
        pending = sorted({(round(float(x),8),round(float(y),8)) for x,y in points
                          if inside(x,y)}-seen)
        if progress:
            print(f'level={level}, spacing={spacing:g} m, new candidates={len(pending)}',flush=True)
        last = time.perf_counter()
        for i,(x,y) in enumerate(pending):
            r = model.evaluate((x,y))
            rows.append(dict(x=x,y=y,J=r['J'],p_near=r['p_near'],p_direction=r['p_direction'],
                             p_no_signal=r['p_no_signal'],normalization_error=r['normalization_error'],
                             level=level,spacing_m=spacing,
                             effective_angle_fallbacks=r['diagnostics'].get('effective_angle_fallbacks',0),
                             direction_mass_correction=r['diagnostics'].get('direction_geometry_mass',0)-
                             r['diagnostics'].get('direction_quadrature_mass',0)))
            seen.add((x,y))
            if progress and (time.perf_counter()-last > 10 or i==len(pending)-1):
                elapsed = time.perf_counter()-start
                print(f'  {i+1}/{len(pending)}, elapsed={elapsed:.1f}s, mean={elapsed/len(rows):.3f}s/point',flush=True)
                last = time.perf_counter()
    step = c['candidate_grid_step']
    xx = np.append(np.arange(bounds[0],bounds[2],step),bounds[2])
    yy = np.append(np.arange(bounds[1],bounds[3],step),bounds[3])
    evaluate_batch([(x,y) for x in xx for y in yy]+[model.s1,(0,0)],0,step)
    for level in range(1,c['refine_levels']+1):
        seeds = []
        for row in sorted(rows,key=lambda r:r['J']):
            p = np.array([row['x'],row['y']])
            if all(np.linalg.norm(p-q) >= step for q in seeds):
                seeds.append(p)
            if len(seeds) >= c['refine_keep']:
                break
        offsets = np.linspace(-step,step,5)
        step /= 2
        evaluate_batch([p+(dx,dy) for p in seeds for dx in offsets for dy in offsets],level,step)
    if not rows:
        raise ValueError('搜索网格未产生候选点，请缩小步长')
    rows.sort(key=lambda r:r['J'])
    j = rows[0]['J']
    threshold = (j+c['near_optimal_absolute_tolerance_m'] if j<=c['zero_objective_threshold_m'] else
                 j*(1+c['near_optimal_relative_tolerance']))
    return rows,[r for r in rows if r['J'] <= threshold],threshold


def write_csv(path, rows, fields=None):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def geometry_json(path, geometries):
    data = {'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{'name':name,'coordinate_units':'m'},'geometry':mapping(g)}
        for name,g in geometries.items() if not g.is_empty]}
    Path(path).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')


def plot_outputs(out, model, rows, near, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = model.c
    def draw(ax,g,**kw):
        if g.is_empty:
            return
        if g.geom_type == 'Polygon':
            xy = np.asarray(g.exterior.coords)
            ax.plot(xy[:,0],xy[:,1],**kw)
            for hole in g.interiors:
                xy = np.asarray(hole.coords)
                ax.plot(xy[:,0],xy[:,1],**kw)
        elif hasattr(g,'geoms'):
            for p in g.geoms:
                draw(ax,p,**kw)
    label = c['input_label']
    best = rows[0]
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    angle = np.linspace(0,TAU,400)
    axes[0].plot(c['L']*np.cos(angle),c['L']*np.sin(angle),'k--',lw=.8)
    for ax in axes:
        draw(ax,model.p1,color='tab:blue',lw=1)
        ax.plot(*model.s1,'rx')
        ax.set(xlabel='East x (m)',ylabel='North y (m)',aspect='equal')
    for cell in model.cells:
        draw(axes[1],cell,color='tab:blue',lw=.3)
    axes[0].set_title(f'{label}: source disk and first feasible region')
    axes[1].set_aspect('auto')
    axes[1].set_title(f'{label}: clipped {c["source_discretization"]} cells (axes scaled separately)')
    fig.tight_layout(); fig.savefig(out/'first_region.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter([r['x'] for r in rows],[r['y'] for r in rows],c=[r['J'] for r in rows],s=28)
    fig.colorbar(sc,ax=ax,label=model.metric)
    ax.scatter([r['x'] for r in near],[r['y'] for r in near],facecolors='none',edgecolors='red',s=80,label='near optimal samples')
    ax.plot(best['x'],best['y'],'r*',ms=12,label='best sampled point')
    ax.plot(*model.s1,'kx',label='S1')
    ax.set(xlabel='East x (m)',ylabel='North y (m)',aspect='equal',title=f'{label}: objective field (sampled)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'objective.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,4))
    centers = (np.arange(model.nbins)+.5)*c['angle_bin_width_deg']
    ax.bar(centers,result['bins'],width=c['angle_bin_width_deg'])
    ax.set(xlabel='Measured bearing (degree)',ylabel='Unconditional bin probability',
           title=f'{label}: near={result["p_near"]:.5f}, no_signal={result["p_no_signal"]:.5f}')
    fig.tight_layout(); fig.savefig(out/'bearing_probabilities.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4))
    for name,g in result['regions'].items():
        draw(ax,g,lw=1,label=name)
    ax.set(xlabel='East x (m)',ylabel='North y (m)',aspect='equal',title=f'{label}: representative posterior regions')
    handles,labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels,handles))
    ax.legend(unique.values(),unique.keys(),fontsize=7)
    fig.tight_layout(); fig.savefig(out/'posteriors.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4))
    sc = ax.scatter(model.xy[:,0],model.xy[:,1],c=model.weights/model.areas,s=8)
    fig.colorbar(sc,ax=ax,label='First posterior density (1/m^2)')
    ax.set(xlabel='East x (m)',ylabel='North y (m)',title=f'{label}: first posterior (axes scaled separately)')
    fig.tight_layout(); fig.savefig(out/'first_posterior.png',dpi=160); plt.close(fig)


def run(c,method):
    out = Path(c['output_dir'])
    # 路径相对于新项目，避免运行目录不同导致输出散落。
    if not out.is_absolute():
        out = Path(__file__).parent/out
    out.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    model = Bayesian(c) if method=='bayes' else FixedRadius(c)
    print(f'{method}: P1 area={model.p1.area:.6f} m^2; cells={len(model.cells)}',flush=True)
    rows,near,threshold = search(model)
    best = model.evaluate((rows[0]['x'],rows[0]['y']),details=True)
    write_csv(out/'field.csv',rows)
    write_csv(out/'near_optimal.csv',near)
    write_csv(out/'bearing_bins.csv',[dict(bin=k,lo_deg=k*c['angle_bin_width_deg'],
              hi_deg=(k+1)*c['angle_bin_width_deg'],probability=float(p),loss_m=float(best['losses'][k]),
              representative_angle_deg=float(best['representative_angle_deg'][k]))
              for k,p in enumerate(best['bins'])])
    posterior = [dict(x=float(p[0]),y=float(p[1]),area_m2=float(a),posterior_mass=float(w))
                 for p,a,w in zip(model.xy,model.areas,model.weights)]
    if method=='bayes':
        for row,d in zip(posterior,model.d1):
            row.update(R_conditional_min_m=max(c['R_min'],float(d)),R_conditional_max_m=c['R_max'])
    write_csv(out/'first_posterior.csv',posterior)
    geometry_json(out/'regions.geojson',{'P1':model.p1,**best['regions']})
    geometry_json(out/'cells.geojson',{str(i):p for i,p in enumerate(model.cells)})
    if c['plots']:
        plot_outputs(out,model,rows,near,best)
    summary = dict(method=method,metric=model.metric,input_label=c['input_label'],best=rows[0],
                   near_optimal_threshold_m=threshold,near_optimal_sample_count=len(near),
                   evaluated_candidates=len(rows),elapsed_seconds=time.perf_counter()-start,
                   first_region_area_m2=model.p1.area,cell_area_relative_error=abs(model.areas.sum()/model.p1.area-1),
                   max_probability_error=max(r['normalization_error'] for r in rows),
                   best_diagnostics=best['diagnostics'],near_loss_m=best['near_loss'],
                   no_signal_loss_m=best['no_signal_loss'],
                   finest_candidate_spacing_m=c['candidate_grid_step']/2**c['refine_levels'],
                   caveat='Numerical sampled optimum in the configured domain; demo is not an official answer.',config=c)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['method','metric','best','elapsed_seconds']},indent=2),flush=True)
    return summary


def main(method):
    parser = argparse.ArgumentParser(description='B题第二问：'+method)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output-dir')
    parser.add_argument('--no-plots',action='store_true')
    args = parser.parse_args()
    overrides = {}
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    if args.no_plots:
        overrides['plots'] = False
    try:
        run(load_config(args.config,method,overrides),method)
    except (ValueError,ArithmeticError) as e:
        parser.exit(2,f'输入或数值错误：{e}\n')
