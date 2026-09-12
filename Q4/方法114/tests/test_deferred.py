import numpy as np
import pytest

from q4.core import Belief
from q4.deferred import observation_branches, insertion_distance
from q4.posterior import Model
from q4.shared import Action, Observation


def test_observation_partition_preserves_probability_and_position_marginal():
    b=Belief();b.channels[1].update(Observation(Action('measure',(0.,0.),1),'direction',45.))
    post=Model().posterior(b.channels[1])
    for point in ((0.,100.),(800.,800.),(2000.,2000.)):
        branches=observation_branches(post,point)
        assert sum(p for _,p,_ in branches)==pytest.approx(1.)
        reconstructed=sum(p*w for _,p,w in branches)
        assert np.max(abs(reconstructed-post.weights))<1e-12
    assert any(name=='no_signal' for name,_,_ in observation_branches(post,(2000.,2000.)))


def test_open_route_insertion_can_defer_a_source_without_dropping_it():
    # Finish at x=300 later, instead of visiting it before x=100 and x=200.
    assert insertion_distance((0.,0.),[(100.,0.),(200.,0.)],(300.,0.))==100.
    # A source already on a necessary leg adds no travel, but its service cost
    # remains separately charged by the policy.
    assert insertion_distance((0.,0.),[(100.,0.),(200.,0.)],(150.,0.))==0.
    assert insertion_distance((0.,0.),[],(300.,400.))==500.
