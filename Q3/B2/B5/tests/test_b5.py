"""B5 behavior contracts and physical edge cases; no official interface."""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from B5.b5_strategy import B5Strategy
from offline_environment import OfflineEnvironment
from solver import build_strategy


class B5Tests(unittest.TestCase):
    def model(self, targets=None):
        targets = targets or [dict(channel=2, position=[600.,30.], radius=1000.),
                              dict(channel=3, position=[620.,-30.], radius=1000.)]
        env = OfflineEnvironment(dict(error_mode='zero', targets=targets))
        model = build_strategy(env.action, 'b5')
        return model, env

    def test_station_cover_includes_outer_seams(self):
        sites = np.array([[1000*math.cos(k*math.tau/7),1000*math.sin(k*math.tau/7)] for k in range(7)])
        for r in (1000.,1800.):
            for a in np.linspace(0,math.tau,1001):
                p = r*np.array([math.cos(a),math.sin(a)])
                self.assertLess(np.linalg.norm(sites-p,axis=1).min(), 1000.)

    def test_cross_sector_is_soft_but_large_detour_waits(self):
        m,_=self.model();m.backbone=[(0,np.array([1000.,0.]))]
        m.position=np.array([800.,600.])
        self.assertNotEqual(m.sector(m.position),0)
        self.assertTrue(m.route_eligible(m.position))
        self.assertFalse(m.route_eligible(np.array([-1000.,0.])))
        self.assertTrue(m.route_eligible(np.array([-1000.,0.]),urgent=True))

    def test_one_shared_probe_does_not_finish_sources_or_sector(self):
        m,e=self.model();m.scan(np.zeros(2))
        m.backbone=[(0,np.array([1000.,0.]))]
        p=np.array([750.,0.]);before=len(m.trace)
        with patch.object(m,'localize',side_effect=AssertionError('No atomic source localization')):
            served=m.execute_local(dict(kind='probe',point=p,channels=[2,3]))
        self.assertEqual(served,[2,3]);self.assertEqual(len(m.backbone),1)
        self.assertEqual([a['path'] for a in m.trace[before:]],['/measure','/measure'])
        self.assertFalse(m.admissible(2,p))
        self.assertEqual(m.b5_stats['shared_stops'],1)

    def test_no_signal_preserves_true_position_and_is_not_repeated(self):
        m,e=self.model([dict(channel=2,position=[1600.,0.],radius=1000.)])
        m.scan(np.zeros(2));m.measure(np.array([900.,50.]),2)
        p=np.array([0.,0.]);self.assertFalse(m.new_measurement(2,p))
        poly=m.polys[2];g=np.array([1600.,0.])
        v=np.roll(poly,-1,axis=0)-poly;delta=g-poly
        cross=v[:,0]*delta[:,1]-v[:,1]*delta[:,0]
        self.assertTrue(np.all(cross>=-1e-4) or np.all(cross<=1e-4))

    def test_cached_metric_cannot_revive_redundant_observation(self):
        m,_=self.model();m.scan(np.zeros(2));p=np.array([750.,0.])
        self.assertIsNotNone(m.metrics(2,p,2))
        old=m.polys[2].tobytes()
        # This history change can happen when a new constraint is redundant.
        m.measurements[2].append((p.copy(),0.))
        self.assertEqual(m.polys[2].tobytes(),old)
        self.assertIsNone(m.metrics(2,p,2))

    def test_empty_channel_needs_cover_or_sixteen_sources(self):
        m,_=self.model();m.backbone=[(0,np.array([1000.,0.]))]
        m.refresh_absence();self.assertEqual(m.absent,set())
        m.cleared=set(range(1,17));m.refresh_absence()
        self.assertEqual(m.absent,set(range(17,21)));self.assertFalse(m.backbone)

    def test_uncertain_failed_clear_is_not_repeated_at_same_point(self):
        m,e=self.model();m.scan(np.zeros(2));p=np.array([800.,0.])
        m.execute_local(dict(kind='clear',point=p,channels=[2],guaranteed=False))
        self.assertNotIn(2,m.cleared);self.assertTrue(np.allclose(m.failed_points[2][0],p))

    def test_optical_grid_is_interruptible_and_covers_degenerate_line(self):
        m,e=self.model([dict(channel=2,position=[45.,0.],radius=1000.)])
        m.polys[2]=np.array([[0.,0.],[80.,0.]])
        m.measurements[2]=[]
        actions=0
        while 2 not in m.cleared:
            plan=m.recover_one(2);before=len(m.trace);m.execute_local(plan)
            self.assertEqual(len(m.trace)-before,1);actions+=1
        self.assertLessEqual(actions,10)

    def test_sector_budget_forces_scan_even_with_optimistic_scores(self):
        m,e=self.model()
        m.b5_config['reuse_coverage']=False
        m.b5_config['sector_action_limit']=1
        calls=[]
        def hold(info,urgent):
            calls.append(True)
            return dict(kind='clear',point=np.array([500.,500.]),channels=[2],gain_s=1e9,guaranteed=False)
        # Inspect early progress only; intentionally fake decisions never solve.
        m.b5_config['max_macros']=4
        with patch.object(m,'choose_local',side_effect=hold):
            with self.assertRaisesRegex(RuntimeError,'macro budget'):
                m.run()
        self.assertEqual([r['kind'] for r in m.b5_log],['clear','scan','clear','scan'])

    def test_moved_anchors_cover_continuous_sector_samples(self):
        m,_=self.model()
        m.position=np.array([1200.,500.])
        m.backbone=[(k,1000*np.array([math.cos(k*math.tau/7),math.sin(k*math.tau/7)])) for k in range(7)]
        m.adjust_backbone({2:(np.array([1500.,400.]),100.),3:(np.array([-1600.,0.]),80.)})
        for k,p in m.backbone:
            for r in np.linspace(1000,1800,11):
                angles=np.linspace(k*math.tau/7-math.pi/7,k*math.tau/7+math.pi/7,101)
                positions=r*np.column_stack((np.cos(angles),np.sin(angles)))
                self.assertLess(np.linalg.norm(positions-p,axis=1).max(),1000.)

    def test_coverage_reuse_requires_real_unknown_channel_scan(self):
        m,e=self.model();p=np.array([1000.,0.]);m.position=p.copy()
        m.backbone=[(0,p.copy()),(1,np.array([623.4898,781.8315]))]
        self.assertIn(0,m.reusable_sectors())
        self.assertEqual(len(m.backbone),2)  # Position alone is not evidence.
        before=len(m.trace);m.scan_reused([0])
        measured={r['channel'] for r in m.trace[before:] if r['path']=='/measure'}
        self.assertEqual(measured,set(range(1,21)))
        self.assertEqual([k for k,_ in m.backbone],[1])

    def test_waiting_source_becomes_required_without_cross_zone_pingpong(self):
        m,e=self.model();m.b5_config.update(max_wait=2,max_macros=3,reuse_coverage=False,
                                           sector_action_limit=20)
        urgent=[]
        def hold(info,required):
            urgent.append(required)
            return dict(kind='clear',point=np.array([500.,500.]),channels=[3],gain_s=1e9,guaranteed=False)
        with patch.object(m,'route_eligible',return_value=True),patch.object(m,'choose_local',side_effect=hold):
            with self.assertRaisesRegex(RuntimeError,'macro budget'):m.run()
        self.assertEqual(urgent[0],None);self.assertEqual(urgent[1],2)

    def test_small_complete_flow_and_real_feedback_only(self):
        m,e=self.model();m.run()
        self.assertEqual(e.cleared,set(e.targets))
        self.assertTrue(m.absent.isdisjoint(e.targets))
        self.assertEqual(len(m.cleared|m.absent),20)
        self.assertTrue(any(r['kind']=='probe' for r in m.b5_log))
        self.assertTrue(all(r['action_count']>0 for r in m.b5_log))


if __name__=='__main__':unittest.main()
