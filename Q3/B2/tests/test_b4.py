import itertools
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from b4_strategy import detour_seconds,channel_order,service_seconds
from solver import build_strategy
from offline_environment import OfflineEnvironment


class B4Tests(unittest.TestCase):
    def model(self):
        env=OfflineEnvironment(dict(error_mode='zero',targets=[
            dict(channel=2,position=[600.,30.],radius=1000.),
            dict(channel=3,position=[620.,-30.],radius=1000.)]))
        model=build_strategy(env.action,'b4')
        model.scan(np.zeros(2))
        return model,env

    def test_channel_order_is_minimum_switch_cost(self):
        for current in [1,2,3,4]:
            channels=[2,3,4]
            costs=[5*len(order)+sum(c!=(current if i==0 else order[i-1])
                    for i,c in enumerate(order)) for order in itertools.permutations(channels)]
            order=channel_order(channels,current,3)
            actual=5*len(order)+sum(c!=(current if i==0 else order[i-1]) for i,c in enumerate(order))
            self.assertEqual(actual,min(costs))
            self.assertEqual(actual,service_seconds(order,current))

    def test_detour_and_subset_cost(self):
        a,b=np.array([0.,0.]),np.array([100.,0.])
        self.assertEqual(detour_seconds(a,np.array([50.,0.]),b),0.)
        self.assertGreater(detour_seconds(a,np.array([50.,50.]),b),0.)
        model,_=self.model();model.receiver_channel=2
        self.assertEqual(model.choose_subset({2:5.5,3:5.5,4:10.},None),[2,4])
        self.assertEqual(model.choose_subset({3:10.},2),[])

    def test_averaging_does_not_imply_joint_reception(self):
        model,_=self.model()
        model.polys[2]=np.array([[1490.,-10.],[1510.,-10.],[1510.,10.],[1490.,10.]])
        model.polys[3]=-model.polys[2]
        self.assertTrue(model.admissible(2,np.array([1500.,0.])))
        self.assertTrue(model.admissible(3,np.array([-1500.,0.])))
        self.assertFalse(model.admissible(2,np.zeros(2)))
        self.assertFalse(model.admissible(3,np.zeros(2)))

    def test_joint_execution_keeps_sector_pending_and_uses_one_location(self):
        model,env=self.model()
        point=np.array([750.,0.])
        self.assertTrue(model.admissible(2,point) and model.admissible(3,point))
        remaining={0,1,2}
        before=len(model.trace)
        model.execute_task(['joint',dict(channels=[2,3],gain_s=10.),point],remaining)
        self.assertEqual(remaining,{0,1,2})
        self.assertEqual(model.joint_measurements,2)
        observed=[r for r in model.trace[before:] if r['path']=='/measure']
        self.assertEqual([r['channel'] for r in observed],[2,3])
        self.assertTrue(all(np.allclose(r['position'],point) for r in observed))
        self.assertTrue(model._after_joint)
        baseline=['clear',2,model.routing_center(2)]
        self.assertIs(model.select_task([baseline],[0],[],remaining),baseline)

    def test_low_gain_preserves_baseline(self):
        model,_=self.model()
        baseline=['clear',2,model.routing_center(2)]
        # An observation with no predicted shrinkage must not trigger joint detours.
        with patch('b4_strategy.expected_cost',return_value=dict(cost_s=50.,
                ready_probability=0.,expected_radius_m=1e6)):
            self.assertIsNone(model.plan_joint(baseline))

    def test_same_point_is_not_new_information(self):
        model,_=self.model()
        self.assertFalse(model.admissible(2,np.zeros(2)))
        model.cleared.add(2)
        self.assertFalse(model.admissible(2,np.array([750.,0.])))

    def test_joint_probe_finishes_the_original_primary_task(self):
        model,env=self.model()
        point=np.array([750.,0.])
        remaining={0,1,2}
        model.execute_task(['joint',dict(channels=[3,2],primary=2,gain_s=10.),point],remaining)
        self.assertIn(2,model.cleared)
        self.assertIn(2,env.cleared)
        self.assertEqual(model.joint_measurements,2)


if __name__=='__main__':unittest.main()
