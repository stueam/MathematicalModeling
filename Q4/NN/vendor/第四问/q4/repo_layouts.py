"""Frozen external coverage layouts, evaluated with the existing ProbePolicy.

Only coordinates are transferred. Routing, posterior, channel evidence and the
finite square-grid fallback remain those of this project. Source revisions are
pinned below; no external solver or runtime file is imported.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np

from .core import DOMAIN
from .coverage import certifies
from .sector import ProbePolicy


SOURCES = {
    'tong22': 'https://github.com/tong2324919503jl/26-/blob/'
              '8aefbd78df015bf8d2034257a07353176c35798d/problem4/search_layout.py',
    's21': 'https://github.com/3371879035-lang/shxjm-B-RL/blob/'
           'b15bfaae4bc67e9a4bcb0974a89de3d5e4a2610c/brl/coverage.py',
    's25': 'https://github.com/3371879035-lang/shxjm-B-RL/blob/'
           'b15bfaae4bc67e9a4bcb0974a89de3d5e4a2610c/brl/coverage.py',
    'tri25': 'https://github.com/feiniao87968492/CCCC/blob/'
             '71077da3f0d2b2ea47d530f19f5a68b70126c821/B题/src/q4_cover.py',
}
# tong22 is retained for geometry diagnosis but fails our current conservative
# polygon certificate. Do not advertise it as an executable local policy.
POLICIES = ('probes-s21', 'probes-s25', 'probes-tri25')

# Preserve the published nonuniform coordinates and the exact integer S21.
TONG22 = (
    (0.0, 0.0),
    (964.5515517477214, -2.5141093794892644),
    (604.2485387606617, 754.5477667140619),
    (-159.5260349827816, 924.9346196887958),
    (-914.1877328941633, 509.18399584126234),
    (-856.1934991432322, -432.59901140364667),
    (-213.50027740515807, -935.5760263970167),
    (594.9066003037709, -756.7604700722686),
    (1849.2346331352699, 1.8477590650225735),
    (1659.0142310264234, 810.4630919605346),
    (1144.9244448947306, 1449.5079701083196),
    (399.6637278191817, 1803.6166375363737),
    (-417.831619638129, 1796.9196774699847),
    (-1141.2857152918975, 1449.3583270101292),
    (-1656.2771869163237, 821.0709009719499),
    (-1850.3826834323652, -1.92387953251106),
    (-1663.6686851876796, -804.7262455989857),
    (-1144.2173381135444, -1450.2150768895058),
    (-400.51148688420454, -1802.8512706716435),
    (415.98386057310455, -1797.6850443347153),
    (1148.4599788006628, -1445.9724362023871),
    (1660.678046388523, -807.4462863480685),
)
S21 = (
    (0, 0), (998, 0), (706, 706), (0, 998), (-706, 706),
    (-998, 0), (-706, -706), (0, -998), (706, -706),
    (1866, 0), (1616, 933), (933, 1616), (0, 1866),
    (-933, 1616), (-1616, 933), (-1866, 0), (-1616, -933),
    (-933, -1616), (0, -1866), (933, -1616), (1616, -933),
)


@lru_cache(maxsize=4)
def layout_points(name):
    """Return frozen coordinates; independently certify before policy use."""
    if name == 'tong22':
        points = TONG22
    elif name == 's21':
        points = S21
    elif name == 's25':
        k = np.arange(12, dtype=float)
        inner_angle = np.deg2rad(15.0+30.0*k)
        outer_angle = np.deg2rad(30.0*k)
        points = np.vstack((np.zeros((1, 2)),
            970.0*np.column_stack((np.cos(inner_angle), np.sin(inner_angle))),
            1880.0*np.column_stack((np.cos(outer_angle), np.sin(outer_angle)))))
    elif name == 'tri25':
        # Published TRI25 at the pinned revision has radius 1950, not 1900.
        points = [(0., 0.)]+[(920.*(q+r/2.), 920.*np.sqrt(3.)/2.*r)
            for q in range(-2, 3) for r in range(-2, 3)
            if (q, r) != (0, 0) and max(abs(q), abs(r), abs(q+r)) <= 2]
        angles = np.pi/6+np.arange(6)*np.pi/3
        points = np.concatenate((points,
            1950.*np.column_stack((np.cos(angles), np.sin(angles)))))
    else:
        raise ValueError(f'Unknown frozen layout: {name}')
    return tuple(tuple(map(float, p)) for p in points)


@lru_cache(maxsize=4)
def certified_layout(name):
    points = layout_points(name)
    if not certifies(DOMAIN, points):
        raise ValueError(f'{name} lacks a full conservative Q4 certificate')
    return points


def layout_manifest(name):
    points = layout_points(name)
    serialized = json.dumps(points, separators=(',', ':'), allow_nan=False)
    return {'name': name, 'source': SOURCES[name], 'points': points,
            'coordinate_sha256': hashlib.sha256(serialized.encode()).hexdigest()}


class RepoLayoutPolicy(ProbePolicy):
    def __init__(self, name, config=None):
        points = certified_layout(name)
        super().__init__(config)
        self.points = {-i-1: p for i, p in enumerate(points)}
        self.implementation = f'q4_github_layout_{name}_probes_v1'
        self.counters['initial_search_stations'] = len(points)
        self.pending_cover_audit.append({
            'operation': 'initialize_external_layout', **layout_manifest(name),
            'full_certificate_valid': True, 'probability_used_for_proof': False,
            'planning_only': True,
        })
