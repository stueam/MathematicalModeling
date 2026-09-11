"""模型内验证及加密检查；不调用也不冒充官方模拟器。"""
import argparse
import json
from pathlib import Path
import time
import unittest
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from geometry import TAU, bearing_contributions, diameter, disk, radius, split_region, classify
from models import FixedRadius, Bayesian
from runner import load_config, write_csv, search

ROOT = Path(__file__).parent


def config(method='fixed',**kw):
    c = load_config(ROOT/'configs'/f'demo_{method}.json',method)
    return {**c,**kw}


def reference_alpha(phi,eps,width):
    """独立慢路径：逐点、逐档、三个周期平移求交。"""
    out = np.zeros((len(phi),round(TAU/width)))
    for j,p in enumerate(phi % TAU):
        for k in range(out.shape[1]):
            for shift in [-TAU,0,TAU]:
                out[j,k] += max(0,min(p+eps,(k+1)*width+shift)-max(p-eps,k*width+shift))/(2*eps)
    return out


class Checks(unittest.TestCase):
    def test_circular_sparse_against_reference(self):
        phi = np.radians([0,.01,359.99,180,45,-.8])
        eps,width = np.radians([1,.5])
        ids,ks,a = bearing_contributions(phi,eps,width)
        actual = np.zeros((len(phi),720))
        np.add.at(actual,(ids,ks),a)
        np.testing.assert_allclose(actual,reference_alpha(phi,eps,width),atol=1e-12)
        np.testing.assert_allclose(actual.sum(axis=1),1,atol=1e-12)

    def test_classification_boundaries(self):
        near,direction,none = classify(np.array([5.,1200.,1200.001]),1200.)
        np.testing.assert_array_equal(near,[True,False,False])
        np.testing.assert_array_equal(direction,[False,True,False])
        np.testing.assert_array_equal(none,[False,False,True])

    def test_same_point_is_deterministic(self):
        for cls,name in [(FixedRadius,'fixed'),(Bayesian,'bayes')]:
            m = cls(config(name))
            r = m.evaluate(m.s1)
            self.assertEqual(np.count_nonzero(r['bins']),1)
            self.assertEqual(r['p_direction'],1)
            expected = diameter(m.p1) if name=='fixed' else m.credible(m.weights)[0]
            self.assertAlmostEqual(r['J'],expected)

    def test_geometry_and_near(self):
        near = disk((0,0),5,.001)
        self.assertTrue(near.intersection(box(10,10,11,11)).is_empty)
        self.assertGreater(diameter(near.intersection(box(0,-8,8,8))),0)
        self.assertLessEqual(diameter(near),10+1e-10)
        self.assertAlmostEqual(diameter(near.intersection(box(-8,-8,8,8))),10)
        self.assertAlmostEqual(near.intersection(box(5,-1,6,1)).area,0)
        multi = shapely.union_all([box(-10,0,-9,1),box(9,0,10,1)])
        self.assertAlmostEqual(diameter(multi),np.sqrt(401))
        self.assertEqual(diameter(Polygon()),0)
        self.assertEqual(diameter(Point(1,2)),0)
        self.assertAlmostEqual(radius(multi),np.sqrt(401)/2)
        triangle = Polygon([(0,0),(2,0),(1,np.sqrt(3))])
        self.assertAlmostEqual(diameter(triangle),2)
        self.assertAlmostEqual(radius(triangle),2/np.sqrt(3))

    def test_clipped_area_and_weights(self):
        for mode in ['polar','square']:
            m = FixedRadius(config(x1=1700.,theta1_deg=80.,source_discretization=mode))
            self.assertAlmostEqual(m.areas.sum()/m.p1.area,1,places=9)
            self.assertTrue(np.all(m.weights>=0))
            self.assertAlmostEqual(m.weights.sum(),1)
            for p in m.cells[:10]:
                children = split_region(p)
                self.assertAlmostEqual(sum(q.area for q in children)/p.area,1,places=9)
                self.assertTrue(p.covers(p.representative_point()))

    def test_joint_radius_conditioning(self):
        m = Bayesian(config('bayes'))
        np.testing.assert_allclose(m.survival(np.array([500,1000,1250,1500,1600])),[1,1,.5,0,0])
        # 已知 d1=1350 后，在 d2=1400 接收的条件概率为100/150。
        self.assertAlmostEqual(float(m.survival(1400)/m.survival(1350)),2/3)
        _,_,mass = m.credible(m.weights)
        self.assertGreaterEqual(mass,m.c['credible_mass'])
        # 固定 R 下后验面积密度恒定，不能偷偷加入距离衰减。
        f = FixedRadius(config())
        np.testing.assert_allclose(f.weights/f.areas,1/f.areas.sum())

    def test_probabilities_and_parallel_bearings(self):
        for cls,name in [(FixedRadius,'fixed'),(Bayesian,'bayes')]:
            m = cls(config(name))
            for s in [(600,0),(600,300),(3000,3000),(800,-14)]:
                r = m.evaluate(s)
                self.assertLess(r['normalization_error'],1e-10)
                self.assertGreaterEqual(r['J'],0)
                self.assertLessEqual(r['near_loss'],(10 if name=='fixed' else 5)+1e-8)
            self.assertAlmostEqual(m.evaluate((3000,3000))['p_no_signal'],1)

    def test_empty_and_bad_config(self):
        with self.assertRaises(ValueError):
            FixedRadius(config(x1=5000,theta1_deg=0))
        with self.assertRaises(ValueError):
            load_config(ROOT/'configs/demo_fixed.json','fixed',{'angle_bin_width_deg':7})
        with self.assertRaises(ValueError):
            load_config(ROOT/'configs/demo_fixed.json','fixed',{'candidate_grid_step':0})

    def test_rectangle_search_bounds(self):
        m = FixedRadius(config(search_domain='rectangle',rectangle=[400,200,500,300],refine_levels=0))
        rows,near,threshold = search(m,progress=False)
        self.assertEqual(len(rows),4)
        self.assertTrue(all(400<=r['x']<=500 and 200<=r['y']<=300 for r in rows))
        self.assertTrue(all(r['J']<=threshold for r in near))


def monte_carlo(model,s,n=60000):
    """从真实圆弧 P1 面积拒绝采样，不复用求解器的离散点/裁剪区域。"""
    rng = np.random.default_rng(model.c['seed'])
    positions,receiving = [],[]
    count = 0
    while count < n:
        size = max(1000,2*(n-count))
        rr = np.sqrt(rng.uniform(25,model.rmax**2,size))
        beta = np.radians(model.c['theta1_deg'])+rng.uniform(-model.eps,model.eps,size)
        xy = model.s1+rr[:,None]*np.column_stack((np.cos(beta),np.sin(beta)))
        keep = np.linalg.norm(xy,axis=1)<=model.c['L']
        if isinstance(model,Bayesian):
            # 联合先验抽 R，再用首次接收事件拒绝；R 不会在第二次重新抽样。
            radii = rng.uniform(model.c['R_min'],model.c['R_max'],size)
            keep &= rr <= radii
        else:
            radii = np.full(size,model.rmax)
        positions.append(xy[keep]); receiving.append(radii[keep])
        count += int(keep.sum())
    xy = np.concatenate(positions)[:n]
    rr = np.concatenate(receiving)[:n]
    delta = xy-np.asarray(s)
    d = np.linalg.norm(delta,axis=1)
    near,none = d<=5,d>rr
    direction = ~(near|none)
    theta = (np.arctan2(delta[direction,1],delta[direction,0])+rng.uniform(-model.eps,model.eps,direction.sum()))%TAU
    bins = np.bincount((theta/model.width).astype(int),minlength=model.nbins)/n
    observed = np.r_[near.mean(),none.mean(),bins]
    r = model.evaluate(s)
    predicted = np.r_[r['p_near'],r['p_no_signal'],r['bins']]
    return dict(n=n,point=list(s),max_absolute_error=float(np.abs(predicted-observed).max()),
                total_variation=float(np.abs(predicted-observed).sum()/2),
                near_mc=float(near.mean()),near_predicted=r['p_near'],
                no_signal_mc=float(none.mean()),no_signal_predicted=r['p_no_signal'])


def numerical_report(output,with_search=False):
    output.mkdir(parents=True,exist_ok=True)
    rows,mc = [],{}
    for cls,name in [(FixedRadius,'fixed'),(Bayesian,'bayes')]:
        base = config(name)
        point = (600.,300.)
        variants = [('base',{}),('radial_x2',{'radial_bins':base['radial_bins']*2}),
                    ('angular_x2',{'angular_bins':base['angular_bins']*2}),
                    ('observation_half',{'angle_bin_width_deg':base['angle_bin_width_deg']/2}),
                    ('arc_half',{'arc_tolerance_m':base['arc_tolerance_m']/2})]
        reference = None
        for label,changes in variants:
            c = {**base,**changes}
            start = time.perf_counter()
            m = cls(c)
            r = m.evaluate(point)
            if reference is None:
                reference = r['J']
                mc[name] = monte_carlo(m,point)
                mc[name+'_with_no_signal'] = monte_carlo(m,(0.,800.))
                mc[name+'_with_near'] = monte_carlo(m,(800.,-14.))
            row = dict(method=name,variant=label,x=point[0],y=point[1],J=r['J'],
                       difference_from_base_m=r['J']-reference,p_near=r['p_near'],
                       p_direction=r['p_direction'],p_no_signal=r['p_no_signal'])
            if with_search:
                found,_,_ = search(m,progress=False)
                row.update(search_best_x=found[0]['x'],search_best_y=found[0]['y'],search_Jmin=found[0]['J'])
            row['elapsed_seconds'] = time.perf_counter()-start
            rows.append(row)
            print(row,flush=True)
    write_csv(output/'convergence.csv',rows)
    (output/'monte_carlo.json').write_text(json.dumps(mc,indent=2),encoding='utf-8')
    print(json.dumps(mc,indent=2),flush=True)
    # 这是有限精度演示的宽容诊断门槛，不是严格精度证明。
    if any(r['total_variation']>.08 for r in mc.values()):
        raise AssertionError('蒙特卡洛概率差异过大，请加密位置网格')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numerical',action='store_true')
    parser.add_argument('--with-search',action='store_true',help='每种加密参数重新搜索，检查最优点移动')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    output = ROOT/'results/validation'
    output.mkdir(parents=True,exist_ok=True)
    (output/'tests.json').write_text(json.dumps(dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful()),indent=2),encoding='utf-8')
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.numerical:
        numerical_report(ROOT/'results/validation',args.with_search)
