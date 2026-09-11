import itertools
import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from q2_geometry import minimum_circle, area_samples, direction_region
from q2_lookahead import expected_cost
from q3_strategy import envelope_circle
from solver import build_strategy
from offline_environment import OfflineEnvironment


def exhaustive_circle(p):
    """Independent small-n oracle: enumerate support sets of size 1,2,3."""
    centers=list(p.copy())
    centers.extend((a+b)/2 for a,b in itertools.combinations(p,2))
    for a,b,c in itertools.combinations(p,3):
        matrix=2*np.array([b-a,c-a])
        if abs(np.linalg.det(matrix))>1e-8:
            centers.append(np.linalg.solve(matrix,np.array([b@b-a@a,c@c-a@a])))
    return min(float(np.linalg.norm(p-c,axis=1).max()) for c in centers)


class Q2Tests(unittest.TestCase):
    def test_circle_against_independent_oracle(self):
        rng=np.random.default_rng(5021)
        for _ in range(80):
            points=rng.uniform(-2000,2000,(8,2))
            center,radius=minimum_circle(points)
            self.assertAlmostEqual(radius,exhaustive_circle(points),places=6)
            self.assertLessEqual(radius,envelope_circle(points)[1]+1e-7)
            self.assertLessEqual(np.linalg.norm(points-center,axis=1).max(),radius)

    def test_circle_degeneracy_translation_and_diameter(self):
        for points,expected in [([[3,4]],0),([[0,0],[4,0],[1,0]],2),
                                ([[0,0],[40,0],[20,20*math.sqrt(3)]],40/math.sqrt(3))]:
            points=np.array(points,dtype=float)
            for offset in [np.zeros(2),np.array([1e6,-1e6])]:
                self.assertAlmostEqual(minimum_circle(points+offset)[1],expected,places=6)
        with self.assertRaises(ValueError):minimum_circle([])

    def test_area_quadrature_refines_and_rejects_empty(self):
        p=np.array([[0.,0.],[100.,0.],[0.,10.]])
        points,w=area_samples(p,12)
        self.assertTrue((points[:,0]/100+points[:,1]/10<=1+1e-12).all())
        self.assertTrue((w>=0).all())
        self.assertAlmostEqual(w.sum(),1.)
        np.testing.assert_allclose(w@points,p.mean(axis=0),atol=.04)
        np.testing.assert_array_equal(area_samples(p,3)[0],points[:8])
        with self.assertRaises(ValueError):area_samples(np.array([[0.,0.],[1.,0.],[2.,0.]]))

    def test_near_cost_and_probability(self):
        p=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        pts,w=area_samples(p)
        result=expected_cost(p,np.zeros(2),np.array([30.,40.]),1,pts,w)
        self.assertEqual(result['cost_s'],21.) # move10 + measure6 + clear5
        self.assertEqual(result['ready_probability'],1.)
        self.assertEqual(result['probability_mass'],1.)
        self.assertEqual(result['no_signal_probability'],0.)
        with self.assertRaises(ValueError):
            expected_cost(p,np.array([1500.,0.]),np.zeros(2),0,pts,w)

    def test_wrap_and_extreme_bearing_keep_generated_target(self):
        poly=np.array([[-100.,-100.],[100.,-100.],[100.,100.],[-100.,100.]])
        station=np.array([200.,0.])
        for target in [np.array([-80.,.001]),np.array([20.,-.001])]:
            angle=np.degrees(np.arctan2(*(target-station)[::-1]))
            for err in [-1.,0.,1.]:
                clipped=direction_region(poly,station,round(float(angle+err),2)%360)
                center,radius=minimum_circle(clipped)
                self.assertLessEqual(np.linalg.norm(target-center),radius+1e-6)

    def test_receiver_channel_and_repeated_points(self):
        env=OfflineEnvironment(dict(error_mode='zero',targets=[dict(channel=2,position=[200.,0.],radius=1000.)]))
        model=build_strategy(env.action,'q2')
        model.measure(np.zeros(2),2)
        model.clear(np.zeros(2),1)
        self.assertEqual(model.receiver_channel,2)
        poly=model.polys[2]
        center,radius=model.region_circle(poly)
        choices=model.candidates(2,poly,center,radius,np.zeros(2))
        self.assertTrue(all(np.linalg.norm(p)>1 for p in choices))
        self.assertTrue(all(np.linalg.norm(poly-p,axis=1).max()<=999 for p in choices))


if __name__=='__main__':unittest.main()
