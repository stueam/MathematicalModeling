import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from q4.core import Belief, DOMAIN
from q4.coverage import certifies
from q4.external_candidates import external_polar_points, heading_moment
from q4.hull_certificate import ConvexChannel, convex_piece_certifies, hull_certifies
from q4.posterior import Model
from q4.sector import ring_stations
from q4.shared import Action, Observation
from surround_audit import angle_gaps


def test_heading_moment_has_correct_direction_and_uniform_limit():
    b=Belief();model=Model()
    u,kappa,probability=heading_moment(model.posterior(b.channels[1]))
    assert kappa<1e-12 and probability==pytest.approx(.5)
    b.channels[1].update(Observation(Action('measure',(0.,0.),1),'direction',0.))
    u,kappa,probability=heading_moment(model.posterior(b.channels[1]))
    # Position quadrature is not perfectly mirror symmetric; the heading
    # integral at each position remains analytic.
    assert u[0]<-.999 and abs(u[1])<1e-4
    assert abs(kappa-2/math.pi)<.002


def test_external_polar_requires_complete_continuum_certificate():
    assert len(external_polar_points())==22
    assert certifies(DOMAIN,external_polar_points())


def test_convex_certificate_requires_surrounding_and_range_for_whole_piece():
    q=Point(0,0).buffer(10)
    surrounding=[(500.,0.),(-250.,433.),(-250.,-433.)]
    assert convex_piece_certifies(q,surrounding)
    assert not convex_piece_certifies(q,[(500.,0.),(600.,100.),(600.,-100.)])
    assert not convex_piece_certifies(q,[(1500.,0.),(-750.,1300.),(-750.,-1300.)])
    # Three points surround the representative point but not all of Q.
    assert not convex_piece_certifies(Point(0,0).buffer(600),surrounding)


def test_convex_certificate_keeps_range_boundary_margin_and_degenerate_cases():
    q=Point(0,0)
    assert convex_piece_certifies(q,[(-999.5,0.),(999.5,0.)])
    assert not convex_piece_certifies(q,[(-999.5001,0.),(999.5001,0.)])
    assert not convex_piece_certifies(Polygon([(0,0),(1,0),(0,1)]),[(-10.,0.),(10.,0.)])
    assert not convex_piece_certifies(q,[])


def test_hull_channel_uses_actual_channel_history_and_keeps_unproved_slivers():
    channel=ConvexChannel()
    for point in reversed(ring_stations()):
        channel.update(Observation(Action('measure',point,1),'no_signal'))
    assert channel.status=='absent_certified'
    untouched=ConvexChannel();untouched.region=Point(0,0).buffer(1e-5)
    untouched.update(Observation(Action('measure',(1500.,0.),2),'no_signal'))
    assert not untouched.region.is_empty
    assert hull_certifies(DOMAIN,ring_stations())


def test_independent_gap_distinguishes_distance_coverage_from_surrounding():
    samples=np.array([[0.,0.]])
    assert angle_gaps(samples,[(900.,0.),(850.,100.),(850.,-100.)])[0]>180
    assert angle_gaps(samples,[(900.,0.),(-450.,779.),(-450.,-779.)])[0]<180
    assert angle_gaps(samples,[(-500.,0.),(500.,0.)])[0]==180
