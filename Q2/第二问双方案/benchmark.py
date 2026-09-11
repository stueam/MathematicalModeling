"""同一批隐藏真值上的配对实验：选点、产生真实反馈、计算实际误差。

方案二分为假定 R0=1200 和已知真实 R 的 oracle 对照。额外使用统一的
未知 R 完整可行区域评价器，区分选点效果与各算法输出区域定义的差异。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from geometry import TAU, disk, diameter, radius
from models import Bayesian, FixedRadius
from runner import load_config, search, write_csv

ROOT = Path(__file__).parent
VIEWS = [
    ('center', (0., 0.), 359.),
    ('offset', (850., -500.), 145.),
    ('boundary', (1600., 0.), 85.),
]


def none_support(model, s):
    """存在同一个 R 满足首次收到、第二次未收到的完整位置支持。"""
    s = np.asarray(s)
    support = model.p1.difference(disk(s, model.c['R_min'], model.c['arc_tolerance_m']))
    v = s-model.s1
    u = v/np.linalg.norm(v)
    t = np.array([-u[1], u[0]])
    mid = (s+model.s1)/2
    reach = 4*(np.linalg.norm(s)+np.linalg.norm(model.s1)+model.c['L']+model.rmax)
    half = Polygon([mid+t*reach, mid-t*reach, mid-t*reach-u*reach, mid+t*reach-u*reach])
    return support.intersection(half)


def prepare_observer(model, s):
    near, direction, none = model.regions(s)
    result = dict(near=near, direction=direction,
                  none=none_support(model, s) if isinstance(model, Bayesian) else none)
    if isinstance(model, Bayesian):
        delta = model.xy-np.asarray(s)
        d2 = np.linalg.norm(delta, axis=1)
        reception = np.divide(model.survival(np.maximum(model.d1, d2)), model.surv1,
                              out=np.zeros_like(d2), where=model.surv1 > 0)
        result.update(phi=np.arctan2(delta[:, 1], delta[:, 0]),
                      wn=model.weights*(d2 <= 5),
                      wz=model.weights*(d2 > 5)*(1-reception),
                      wd=model.weights*(d2 > 5)*reception)
    return result


def infer(model, s, observation, measured, prepared, common=False):
    """common=True 时两种选点都用未知 R、完整支持、连续示向度评价。"""
    if observation == 'near':
        support = prepared['near']
        weights = prepared.get('wn')
    elif observation == 'no_signal':
        support = prepared['none']
        weights = prepared.get('wz')
    else:
        if not common and isinstance(model, Bayesian):
            k = int((measured % TAU)/model.width)
            center = (k+.5)*model.width
            delta = (prepared['phi']-center+math.pi) % TAU-math.pi
            alpha = np.maximum(0., np.minimum(model.width/2, delta+model.eps)-
                               np.maximum(-model.width/2, delta-model.eps))/(2*model.eps)
            weights = prepared['wd']*alpha
            support = model.bearing_region(prepared['direction'], s, center, model.eps+model.width/2)
        else:
            support = model.bearing_region(prepared['direction'], s, measured)
            weights = None
    if common or isinstance(model, FixedRadius):
        return support
    if weights.sum() <= 0:
        return Polygon()
    return model.credible(weights, support)[1]


def score(region, truth):
    if region.is_empty or region.area <= 0:
        return dict(valid=0, error_m=None, cover_radius_m=None, diameter_m=None,
                    truth_covered=0, truth_covered_with_arc_tolerance=0,
                    clearance_success=0, certified_20m_cover=0)
    circle = shapely.minimum_bounding_circle(region)
    center = circle.centroid
    error = float(np.linalg.norm(np.asarray(truth)-[center.x, center.y]))
    point = Point(truth)
    r = radius(region)
    return dict(valid=1, error_m=error, cover_radius_m=r, diameter_m=diameter(region),
                truth_covered=int(region.covers(point)),
                truth_covered_with_arc_tolerance=int(region.distance(point) <= .06),
                clearance_success=int(error <= 20), certified_20m_cover=int(r <= 20))


def sample_worlds(s1, theta, true_r, n, rng):
    """对给定 R、首次正常示向度，按 P1 的真实面积均匀采样。

    保持圆弧为解析几何，不从求解器网格中抽真值。先给定首次信息，
    再抽与该信息相容的隐状态，是条件模拟；不是读取或泄露目标坐标。
    """
    batches = []
    count = 0
    attempts = 0
    while count < n:
        size = max(1024, 2*(n-count))
        r = np.sqrt(rng.uniform(25., true_r**2, size))
        beta = theta+rng.uniform(-math.pi/180, math.pi/180, size)
        xy = np.asarray(s1)+r[:, None]*np.column_stack((np.cos(beta), np.sin(beta)))
        keep = np.linalg.norm(xy, axis=1) <= 1800
        batches.append(xy[keep])
        count += int(keep.sum())
        attempts += size
        if attempts > 10000000:
            raise RuntimeError('该场景首次观测条件的接受率过低')
    return np.concatenate(batches)[:n]


def choose(view_name, s1, theta, method, r0, output):
    name = 'bayes' if method == 'bayes' else 'fixed'
    c = load_config(ROOT/'configs'/f'demo_{name}.json', name)
    c.update(x1=s1[0], y1=s1[1], theta1_deg=theta, plots=False)
    if name == 'fixed':
        c['R0'] = r0
    source_hash = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in
                                         ['models.py', 'geometry.py', 'runner.py'])).hexdigest()
    cache = output/'plans'/f'{view_name}_{method}_{r0:g}.json'
    start = time.perf_counter()
    model = Bayesian(c) if name == 'bayes' else FixedRadius(c)
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['config'] == c and data['source_sha256'] == source_hash:
            print(f'REUSE {view_name}/{method}/{r0:g}: {data["point"]}', flush=True)
            return model, data
    print(f'SEARCH {view_name}/{method}/{r0:g}', flush=True)
    rows, near, threshold = search(model)
    data = dict(config=c, source_sha256=source_hash, point=[rows[0]['x'], rows[0]['y']],
                predicted_J=rows[0]['J'], metric=model.metric,
                selection_seconds=time.perf_counter()-start, candidates=len(rows),
                near_optimal=near, near_optimal_threshold=threshold)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return model, data


def stats(rows, prefix):
    valid = [r for r in rows if r[prefix+'valid']]
    errors = np.array([r[prefix+'error_m'] for r in valid])
    return dict(trials=len(rows), valid_trials=len(valid), failed_trials=len(rows)-len(valid),
                mean_error_m=float(errors.mean()) if len(errors) else None,
                median_error_m=float(np.median(errors)) if len(errors) else None,
                p90_error_m=float(np.quantile(errors, .9)) if len(errors) else None,
                mean_cover_radius_m=float(np.mean([r[prefix+'cover_radius_m'] for r in valid])) if valid else None,
                truth_coverage=float(np.mean([r[prefix+'truth_covered'] for r in rows])),
                truth_coverage_with_arc_tolerance=float(np.mean([r[prefix+'truth_covered_with_arc_tolerance'] for r in rows])),
                clearance_success_rate=float(np.mean([r[prefix+'clearance_success'] for r in rows])),
                certified_20m_cover_rate=float(np.mean([r[prefix+'certified_20m_cover'] for r in rows])))


def summarize(rows, plans, samples, seed, output):
    summary = dict(seed=seed, samples_per_view_and_radius=samples,
                   unique_worlds=len({r['world_id'] for r in rows}),
                   observation_trials=len(rows), algorithms={})
    groups = []
    for method in ['bayes', 'fixed_assumed', 'fixed_oracle']:
        subset = [r for r in rows if r['method'] == method]
        plan_subset = [p for p in plans if p['method'] == method]
        summary['algorithms'][method] = dict(
            native=stats(subset, 'native_'), common_unknown_R=stats(subset, 'common_'),
            direction_rate=float(np.mean([r['observation'] == 'direction' for r in subset])),
            no_signal_rate=float(np.mean([r['observation'] == 'no_signal' for r in subset])),
            mean_selection_seconds=float(np.mean([p['selection_seconds'] for p in plan_subset])),
            mean_move_to_second_point_m=float(np.mean([r['move_m'] for r in subset])))
        for group in sorted({(r['view'], r['true_R']) for r in subset}):
            selected = [r for r in subset if (r['view'], r['true_R']) == group]
            for evaluation in ['native', 'common']:
                groups.append(dict(method=method, view=group[0], true_R=group[1],
                                   evaluation=evaluation, **stats(selected, evaluation+'_')))
    # 同一隐藏世界上的差值；空后验是失败，不将其当成零误差参与均值。
    paired = []
    index = {(r['world_id'], r['method']):r for r in rows}
    for world in sorted({r['world_id'] for r in rows}):
        a,b = index[(world,'bayes')],index[(world,'fixed_assumed')]
        if a['common_valid'] and b['common_valid']:
            paired.append(a['common_error_m']-b['common_error_m'])
    if paired:
        rng = np.random.default_rng(seed+1)
        paired = np.asarray(paired)
        means = np.mean(rng.choice(paired, (2000,len(paired)), replace=True), axis=1)
        summary['paired_common_error_bayes_minus_fixed_assumed'] = dict(
            mean_difference_m=float(paired.mean()),
            conditional_bootstrap_95_interval_m=np.quantile(means,[.025,.975]).tolist(),
            interpretation='Negative favors Bayesian; interval is conditional on the selected views/radii/plans, not generalization to all scenes.')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'trials.csv', rows)
    write_csv(output/'by_case.csv', groups)
    write_csv(output/'selected_points.csv', plans)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def run(samples, seed, radii, views, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, plans, worlds = [], [], []
    for view_index,(view,s1,theta) in enumerate(views):
        # 未知 R 两种策略只知道 S1、theta，不读取任何隐藏真值。
        bayes,bplan = choose(view,s1,theta,'bayes',0,output)
        fixed,fplan = choose(view,s1,theta,'fixed_assumed',1200,output)
        for method,plan in [('bayes',bplan),('fixed_assumed',fplan)]:
            plans.append(dict(view=view,method=method,R_input=0 if method=='bayes' else 1200,
                              x2=plan['point'][0],y2=plan['point'][1],
                              selection_seconds=plan['selection_seconds'],J=plan['predicted_J']))
        for radius_index,true_r in enumerate(radii):
            oracle,oplan = choose(view,s1,theta,'fixed_oracle',true_r,output)
            plans.append(dict(view=view,method='fixed_oracle',R_input=true_r,
                              x2=oplan['point'][0],y2=oplan['point'][1],
                              selection_seconds=oplan['selection_seconds'],J=oplan['predicted_J']))
            rng = np.random.default_rng(np.random.SeedSequence([seed,view_index,radius_index]))
            truths = sample_worlds(s1,math.radians(theta),true_r,samples,rng)
            # 共同随机数配对，保证每种策略的边际误差仍为 U[-1°,1°]。
            errors = rng.uniform(-math.pi/180,math.pi/180,samples)
            methods = [('bayes',bayes,bplan),('fixed_assumed',fixed,fplan),('fixed_oracle',oracle,oplan)]
            for trial,(truth,error2) in enumerate(zip(truths,errors)):
                world_id = f'{view}_{true_r:g}_{trial}'
                worlds.append(dict(world_id=world_id,view=view,true_R=true_r,gx=truth[0],gy=truth[1],
                                   x1=s1[0],y1=s1[1],theta1_deg=theta,
                                   first_error_deg=math.degrees((math.radians(theta)-math.atan2(truth[1]-s1[1],truth[0]-s1[0])+math.pi)%TAU-math.pi),
                                   second_error_deg=math.degrees(error2)))
            for method,model,plan in methods:
                s = np.asarray(plan['point'])
                native_prepared = prepare_observer(model,s)
                common_prepared = prepare_observer(bayes,s)
                for trial,(truth,error2) in enumerate(zip(truths,errors)):
                    d = np.linalg.norm(truth-s)
                    observation = 'near' if d<=5 else ('direction' if d<=true_r else 'no_signal')
                    measured = (math.atan2(truth[1]-s[1],truth[0]-s[0])+error2)%TAU
                    if np.linalg.norm(s-np.asarray(s1)) < 1e-9:
                        measured = math.radians(theta)
                        native = model.credible(model.weights)[1] if isinstance(model,Bayesian) else model.p1
                        common = bayes.p1
                    else:
                        native = infer(model,s,observation,measured,native_prepared)
                        common = infer(bayes,s,observation,measured,common_prepared,common=True)
                    native_score,common_score = score(native,truth),score(common,truth)
                    rows.append(dict(world_id=f'{view}_{true_r:g}_{trial}',view=view,true_R=true_r,
                                     method=method,gx=float(truth[0]),gy=float(truth[1]),
                                     x2=float(s[0]),y2=float(s[1]),observation=observation,
                                     measured_deg=math.degrees(measured) if observation=='direction' else None,
                                     move_m=float(np.linalg.norm(s-np.asarray(s1))),
                                     **{'native_'+k:v for k,v in native_score.items()},
                                     **{'common_'+k:v for k,v in common_score.items()}))
                print(f'OBSERVED {view} R={true_r:g} {method}: {stats(rows[-samples:],"native_")}',flush=True)
            # 每个场景立即落盘，避免长批次结束前没有可恢复的记录。
            write_csv(output/'trials.csv',rows)
            write_csv(output/'worlds.csv',worlds)
    return summarize(rows,plans,samples,seed,output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20260912)
    parser.add_argument('--radii',type=float,nargs='+',default=[1050,1250,1450])
    parser.add_argument('--views',nargs='+',choices=[v[0] for v in VIEWS],default=[v[0] for v in VIEWS])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/benchmark')
    args = parser.parse_args()
    if args.samples<1 or any(not 1000<=r<=1500 for r in args.radii):
        parser.error('samples 必须为正，radii 必须在 [1000,1500]')
    run(args.samples,args.seed,args.radii,[v for v in VIEWS if v[0] in args.views],args.output_dir)
