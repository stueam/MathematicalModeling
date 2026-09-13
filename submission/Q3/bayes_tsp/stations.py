"""Coverage-preserving station optimization used by the Q3 paper algorithm."""

import numpy as np
from scipy.optimize import minimize
from .repair import repair_segment


def covers_vertices(vertices, position, radius=997.0):
    return bool(np.linalg.norm(vertices - position, axis=1).max() <= radius)


def adjust_station(vertices, current, previous, following, radius=997.0):
    """Constrained two-edge improvement, accepted only after full vertex check."""
    current, previous = (np.asarray(current, dtype=float), np.asarray(previous, dtype=float))
    if not covers_vertices(vertices, current, radius):
        raise ValueError('Station optimizer needs an initially feasible point')

    def objective(x):
        return float(
            np.linalg.norm(x - previous) + (np.linalg.norm(x - following) if following is not None else 0.0)
        )

    def jac(x):
        d = x - previous
        out = d / max(float(np.linalg.norm(d)), 1e-12)
        if following is not None:
            d = x - following
            out += d / max(float(np.linalg.norm(d)), 1e-12)
        return out

    def constraint(x):
        return radius - np.linalg.norm(vertices - x, axis=1)

    def constraint_jac(x):
        d = vertices - x
        return d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)

    result = minimize(
        objective,
        current,
        jac=jac,
        method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}],
        options={'maxiter': 30, 'ftol': 1e-05},
    )
    accepted = (
        result.success
        and np.isfinite(result.x).all()
        and covers_vertices(vertices, result.x, radius)
        and (objective(result.x) < objective(current) - 1e-06)
    )
    point = result.x if accepted else current
    return (
        tuple(map(float, point)),
        {
            'accepted': bool(accepted),
            'optimizer_success': bool(result.success),
            'saved_leg_m': objective(current) - objective(point),
            'iterations': int(result.nit),
        },
    )


def repaired_station(vertices, current, previous, following, radius):
    current, previous = (np.asarray(current), np.asarray(previous))

    def objective(x):
        return float(
            np.linalg.norm(x - previous) + (np.linalg.norm(x - following) if following is not None else 0.0)
        )

    def jac(x):
        d = x - previous
        g = d / max(float(np.linalg.norm(d)), 1e-12)
        if following is not None:
            d = x - following
            g += d / max(float(np.linalg.norm(d)), 1e-12)
        return g

    def constraint(x):
        return radius - np.linalg.norm(vertices - x, axis=1)

    def constraint_jac(x):
        d = vertices - x
        return d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)

    result = minimize(
        objective,
        current,
        jac=jac,
        method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}],
        options={'maxiter': 30, 'ftol': 1e-05},
    )
    point = repair_segment(current, result.x, constraint) if result.success else None
    accepted = (
        point is not None
        and covers_vertices(vertices, point, radius)
        and (objective(point) < objective(current) - 1e-06)
    )
    return (
        tuple(map(float, point if accepted else current)),
        {
            'accepted': bool(accepted),
            'optimizer_success': bool(result.success),
            'raw_margin_m': float(np.min(constraint(result.x))),
            'saved_leg_m': objective(current) - objective(point) if accepted else 0.0,
        },
    )
