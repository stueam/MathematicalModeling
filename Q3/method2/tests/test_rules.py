import copy
import numpy as np
import pytest
import torch
from q3ppo.physics import Config, WorldBatch, NO_SIGNAL, NEAR, DIRECTION, CLEARED, CLEAR_FAILED
from q3ppo.belief import BeliefBatch, STATIONS, ANGLE_EPS
from q3ppo.env import RadioEnv, heuristic_actions
from q3ppo.model import ActorCritic, tensor_obs, gae


def fixture_world(n=1):
    w = WorldBatch(Config(num_envs=n, error_mode="zero"))
    w.present[:] = False
    w.present[:, 0] = True
    w.xy[:, 0] = [0, 0]
    w.radius[:, 0] = 1200
    return w


@pytest.mark.parametrize("distance,expected", [(0, NEAR), (5, NEAR), (5+1e-7, DIRECTION),
                                                (1200, DIRECTION), (1200+1e-7, NO_SIGNAL)])
def test_measure_boundaries(distance, expected):
    w = fixture_world()
    result, angle, duration = w.step([[distance, 0]], [0], [False])
    assert result[0] == expected
    assert np.isfinite(angle[0]) == (expected == DIRECTION)
    assert duration[0] == pytest.approx(distance / 5 + 5, abs=1e-6)


@pytest.mark.parametrize("distance,expected", [(20, CLEARED), (20+1e-7, CLEAR_FAILED)])
def test_clear_boundaries(distance, expected):
    w = fixture_world()
    w.channel[0] = 7
    result, _, duration = w.step([[distance, 0]], [0], [True])
    assert result[0] == expected
    assert w.channel[0] == 7
    assert duration[0] == pytest.approx(distance/5 + (5 if expected == CLEARED else 3), abs=1e-6)
    if expected == CLEARED:
        result, _, duration = w.step([[distance, 0]], [0], [True])
        assert result[0] == CLEAR_FAILED and duration[0] == 3
        result, _, _ = w.step([[distance, 0]], [0], [False])
        assert result[0] == NO_SIGNAL


def test_attachment_timing_example():
    w = fixture_world()
    w.present[:] = False
    times = []
    for p, c, clear in [([300,400],0,False), ([300,400],1,False),
                         ([300,0],2,True), ([300,0],1,False)]:
        w.step([p], [c], [clear]); times.append(w.time[0])
    np.testing.assert_allclose(times, [105,111,194,199])
    assert w.channel[0] == 1


@pytest.mark.parametrize("p,c", [([float('nan'),0],0), ([float('inf'),0],0), ([2000001,0],0),
                                ([0,0],1.5), ([0,0],-1), ([0,0],20)])
def test_invalid_actions_atomic(p, c):
    w = fixture_world()
    with pytest.raises(ValueError):
        w.step([p], [c], [False])
    assert w.steps[0] == 0 and w.time[0] == 0
    np.testing.assert_array_equal(w.position, [[0,0]])


def test_error_repeat_round_wrap():
    w = WorldBatch(Config(num_envs=32), 9)
    p = np.tile([10., -20.], (32,1))
    channels = np.arange(32) % 20
    a = w.angle_error(p, channels)
    assert np.max(np.abs(a)) <= 1
    np.testing.assert_array_equal(a, w.angle_error(p, channels))
    assert not np.array_equal(a, w.angle_error(p + .01, channels))
    w = fixture_world()
    w.xy[0,0] = [100., -.001]
    _, angle, _ = w.step([[0,0]], [0], [False])
    assert angle[0] == 0


def test_vector_physics_against_scalar_reference():
    rng = np.random.default_rng(781)
    w = WorldBatch(Config(num_envs=16, error_mode="zero"), 177)
    for _ in range(100):
        channels = rng.integers(0,20,16)
        is_clear = rng.random(16) < .4
        p = rng.uniform(-1800,1800,(16,2))
        for i in range(8):
            p[i] = w.xy[i,channels[i]] + [rng.choice([0,5,20,1000,1500]), 0]
        previous = w.position.copy()
        old_channel = w.channel.copy()
        old_cleared = w.cleared.copy()
        result, angle, dt = w.step(p,channels,is_clear)
        for i in range(16):
            c = channels[i]
            distance = float(np.linalg.norm(w.xy[i,c]-p[i]))
            alive = w.present[i,c] and not old_cleared[i,c]
            if is_clear[i]:
                expected = CLEARED if alive and distance <= 20 else CLEAR_FAILED
                expected_time = (5 if expected == CLEARED else 3)
                assert w.channel[i] == old_channel[i]
            else:
                expected = NO_SIGNAL if not alive or distance > w.radius[i,c] else NEAR if distance <= 5 else DIRECTION
                expected_time = 5 + int(c != old_channel[i])
            assert result[i] == expected
            assert dt[i] == pytest.approx(round(expected_time+np.linalg.norm(p[i]-previous[i])/5,6),abs=1e-6)


def test_safe_outer_region_contains_truth_and_clear_certificate():
    n = 128
    w = WorldBatch(Config(num_envs=n), 7100)
    w.present[:,0] = True
    b = BeliefBatch(n,20)
    rng = np.random.default_rng(124)
    for radius in [800, 400, 100, 10, 0]:
        theta = rng.uniform(0,2*np.pi,n)
        p = w.xy[:,0] + radius*np.column_stack((np.cos(theta),np.sin(theta)))
        result, angle, _ = w.step(p,np.zeros(n,np.int64),np.zeros(n,bool))
        b.update(p,np.zeros(n,np.int64),result,angle)
        for i in range(n):
            poly = b.polys[i,0,:b.sizes[i,0]]
            nxt = np.roll(poly,-1,axis=0)
            edge, v = nxt-poly, w.xy[i,0]-poly
            cross = edge[:,0]*v[:,1]-edge[:,1]*v[:,0]
            assert cross.min() > -1e-5
            if b.radii[i,0] <= 20:
                assert np.linalg.norm(b.centers[i,0]-w.xy[i,0]) <= 20+1e-6


def test_coverage_absence_certificate():
    b = BeliefBatch(1,20)
    for k, p in enumerate(STATIONS):
        b.update(p[None], np.array([0]), np.array([NO_SIGNAL]), np.array([np.nan]))
        assert b.status[0,0] == (3 if k==6 else 0)
    theta = np.linspace(0,2*np.pi,1000)
    points = np.concatenate([r*np.column_stack((np.cos(theta),np.sin(theta))) for r in [0,1000,1800]])
    assert np.linalg.norm(points[:,None]-STATIONS[None],axis=2).min(1).max() <= 1000+1e-6


def test_observation_has_no_hidden_truth_dependency():
    env = RadioEnv(Config(num_envs=2), 991)
    before = {k:v.copy() for k,v in env.observe().items()}
    env.world.xy[:] *= -.7
    env.world.radius[:] = 1000
    env.world.present[:] = ~env.world.present
    after = env.observe()
    for k in before:
        np.testing.assert_array_equal(before[k],after[k])


def test_no_oracle_early_termination():
    env = RadioEnv(Config(num_envs=1), 3)
    env.world.present[0] = False
    env.world.present[0,:10] = True
    env.world.cleared[0,:10] = True
    env.belief.status[0,:10] = 2
    env.obs = env.observe()
    _, _, done, _ = env.step(np.array([10*4]))
    assert not done[0]  # 10 clears does not prove there are no more sources.


def test_gae_terminal_and_rollout_bootstrap():
    rewards = torch.tensor([[1.],[2.],[3.]])
    values = torch.zeros_like(rewards)
    done = torch.tensor([[0.],[1.],[0.]])
    adv, ret = gae(rewards,values,done,torch.tensor([4.]),lam=1)
    torch.testing.assert_close(ret, torch.tensor([[3.],[2.],[7.]]))


def test_potential_telescopes_at_absorbing_terminal():
    env = RadioEnv(Config(num_envs=1,channels=1,min_sources=1,max_sources=1,max_steps=20), 9)
    total_reward = 0
    for _ in range(20):
        _, reward, done, episodes = env.step(heuristic_actions(env.obs))
        total_reward += float(reward[0])
        if done[0]:
            e = episodes[0]
            expected = -e['virtual_s']/env.config.time_scale
            if not e['success']:
                expected -= env.config.failure_penalty
            assert total_reward == pytest.approx(expected, abs=1e-4)
            break
    else:
        pytest.fail("No episode end")


def test_network_channel_permutation_and_mask():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    env = RadioEnv(Config(num_envs=2))
    model = ActorCritic(32)
    ob = tensor_obs(env.obs,'cpu')
    dist,value = model(*ob)
    permutation = np.arange(20)[::-1].copy()
    indices = (permutation[:,None]*4 + np.arange(4)).ravel()
    other,v2 = model(ob[0][:,indices],ob[1],ob[2][:,indices])
    torch.testing.assert_close(value,v2,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(dist.probs[:,indices],other.probs,atol=1e-6,rtol=1e-5)
    assert torch.all(dist.probs[~ob[2]] == 0)
