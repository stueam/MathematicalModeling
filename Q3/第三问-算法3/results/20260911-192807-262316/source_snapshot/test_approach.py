"""Full hard-support clearance lens, independent of Bayesian samples."""
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from bayes_tsp.approach import ApproachConfig, ApproachPolicy, nearest_safe_point
from bayes_tsp.shared import Belief


def test_thin_support_has_a_larger_safe_lens_than_inscribed_circle():
    vertices = np.array([[-19., 0.], [19., 0.]])
    point = nearest_safe_point(vertices, (0., 100.), (0., 1.), radius=20.)
    assert point == pytest.approx((0., math.sqrt(20**2-19**2)), abs=1e-7)
    assert np.linalg.norm(vertices-point, axis=1).max() <= 20.


def test_single_active_circle_and_already_safe_point():
    vertices = np.array([[-1., 0.], [1., 0.], [0., .2]])
    point = nearest_safe_point(vertices, (100., 0.), (0., 0.), radius=20.)
    assert point == pytest.approx((19., 0.))
    assert nearest_safe_point(vertices, (2., 3.), (0., 0.), radius=20.) == (2., 3.)


def test_invalid_witness_is_not_returned_as_safe():
    with pytest.raises(ValueError, match='witness'):
        nearest_safe_point([[0., 0.]], (100., 0.), (30., 0.))


def test_safe_lens_changes_only_endpoint_and_retains_full_geometry():
    belief = Belief(position=(0., 100.))
    channel = belief.channels[1]
    channel.status = 'detected'
    channel.region = Polygon([(-18., -.1), (18., -.1), (18., .1), (-18., .1)])
    before = channel.region.wkb
    policy = ApproachPolicy(ApproachConfig(resolution=8))
    point = policy.guaranteed_clear(belief, 1).position
    assert point[1] > 8.
    assert np.linalg.norm(np.asarray(channel.region.exterior.coords)-point, axis=1).max() <= 19.999
    assert channel.region.wkb == before
    assert channel.status == 'detected'
    assert not belief.done()
    assert not channel.history
