"""Independent, evaluation-only replay of saved Q4 physical trajectories.

This module reads hidden world manifests solely to audit completed logs. It
must not be imported by a policy, observation encoder, or action generator.
Costs below are simulated physical seconds, not Python execution time.
"""

from collections import Counter
from fractions import Fraction
import math
import sys
from typing import Any, Iterable, Mapping

from shapely.geometry import Point

from .bridge import Action, Belief, DOMAIN, distance, point_key
from q4.coverage import full_exclusion
from q4.shared import disk
from q4.simulator import LocalSimulator, Source, World


CLOCK_TOLERANCE_S = 1.000001e-6
GEOMETRY_TOLERANCE_M = 1e-7
BEARING_TOLERANCE_DEG = 1.0050001
_TIMESTAMPS = {"real_timestamp", "real_timestamp_ms"}

# Compared directly with the provided problem and attachments, not inferred
# from prior local benchmark scores. These are scope facts, not validations of
# the official simulator's unknown random world generator.
PROTOCOL_COMPARISON = {
    "sources": ["B题.pdf, pages 2–4, appendices 1/2/4",
                "附件/附件1.docx, sections 1–2",
                "附件/附件2.docx, sections 5/7/8/10/12"],
    "matched_physics": [
        "Straight-line motion at 5 m/s; initially (0,0), receiver channel 1",
        "Measurement takes 5 s and a changed receiver channel adds 1 s",
        "Clear takes 3 s when unsuccessful, 5 s when successful; receiver unchanged",
        "Directional reception includes the two 90-degree half-plane boundaries",
        "Fixed radius in [1000,1500] m; near <=5 m still requires illumination",
        "Clear succeeds within 20 m independently of emitter heading",
        "Fixed same-point bearing error <=1 degree; displayed angle has two decimals",
        "Accepted retry IDs are idempotent; rejected zero clocks do not reset state",
    ],
    "local_assumptions": [
        "Source positions, count, channel subset, radii, type mix and headings are generated from local priors; the problem specifies bounds, not their probability laws",
        "iid uses a seed/channel/exact-position hash error; extreme uses +/-1 degree; correlated uses a bounded spatial sine field; none is an official error generator",
        "Per-action movement and total virtual time are rounded to 6 decimals by the local simulator; the attachments do not prescribe that internal rounding rule",
        "Pure omnidirectional/directional worlds are useful stress extensions; the written Q4 premise describes both source types",
    ],
    "not_validated": [
        "Official case distribution or official practice/formal-test performance",
        "HTTP transport, online login, server-time synchronization, or encrypted submission logs",
        "The 25-minute window and actual remaining /enter runtime (at most 20 minutes)",
    ],
}


class AuditError(ValueError):
    """A saved event contradicts the physical protocol or public certificate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _without_timestamp(response: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in response.items() if key not in _TIMESTAMPS}


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} is not numeric") from exc
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _compare_response(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> float:
    actual, expected = _without_timestamp(actual), _without_timestamp(expected)
    _require(actual.keys() == expected.keys(), "Response fields differ from frozen simulator replay")
    error = 0.0
    for key, expected_value in expected.items():
        if key == "virtual_time_s":
            error = abs(_number(actual[key], key) - expected_value)
            _require(error <= CLOCK_TOLERANCE_S, "Virtual clock differs from simulator replay")
        else:
            _require(actual[key] == expected_value, f"Simulator replay mismatch: {key}")
    return error


def _physical_feedback(action: Action, response: Mapping[str, Any], source: Source | None,
                       active: bool) -> tuple[str, int, float | None]:
    """Check half-plane/radius/bearing physics without World.feedback."""
    separation = distance(action.position, source.position) if active else math.inf
    if action.kind == "clear":
        success = active and separation <= 20.0
        result = "success" if success else "no_target_in_range"
        _require(response.get("clear_result") == result, "Clear must use the 20 m disk irrespective of heading")
        return result, 5 if success else 3, None

    illuminated = True
    if active and source.heading_deg is not None:
        heading = math.radians(source.heading_deg)
        dx, dy = action.position[0] - source.position[0], action.position[1] - source.position[1]
        projection = math.cos(heading) * dx + math.sin(heading) * dy
        # Closed half-plane, including co-location; only floating-point
        # roundoff at the tangent boundary receives a numerical tolerance.
        tolerance = 8 * sys.float_info.epsilon * max(1.0, separation)
        illuminated = projection >= -tolerance
    signal = active and illuminated and separation <= source.radius
    result = "near" if signal and separation <= 5.0 else "direction" if signal else "no_signal"
    _require(response.get("measure_result") == result, "Measurement violates fixed-radius / closed-half-plane physics")
    bearing_error = None
    if result == "direction":
        bearing = _number(response.get("svd_deg"), "svd_deg")
        _require(0 <= bearing < 360, "Bearing must be in [0, 360)")
        true_bearing = math.degrees(math.atan2(source.position[1] - action.position[1],
                                             source.position[0] - action.position[0]))
        bearing_error = abs((bearing - true_bearing + 180) % 360 - 180)
        _require(bearing_error <= BEARING_TOLERANCE_DEG, "Direction error exceeds 1 degree plus 0.005 degree rounding")
    return result, 5, bearing_error


def _exact_surround_certificate(support, negatives):
    """Prove residual components impossible with exact rational arithmetic.

    All vertices of a component must lie strictly inside the convex hull of
    actual negative observation stations that each lie within 1000 m of all
    those vertices. Convexity then gives both properties for every point in
    the entire component. Any closed directional half-plane through an
    interior point contains a station: simultaneous no_signal is impossible.
    This is a geometric proof for a region of any size, never an area cutoff.
    """
    def pieces(geometry):
        if hasattr(geometry, 'geoms'):
            for child in geometry.geoms:
                yield from pieces(child)
        elif not geometry.is_empty:
            yield geometry

    def cross(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def convex_hull(points):
        lower, upper = [], []
        for chain, sequence in ((lower, points), (upper, reversed(points))):
            for point in sequence:
                while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                    chain.pop()
                chain.append(point)
        return lower[:-1] + upper[:-1]

    rational_stations = [tuple(Fraction(float(x)) for x in point) for point in sorted(set(negatives))]
    result = []
    for component in pieces(support):
        coordinates = component.exterior.coords if component.geom_type == 'Polygon' else component.coords
        vertices = sorted(set(tuple(Fraction(float(x)) for x in point) for point in coordinates))
        nearby = [station for station in rational_stations if all(
            (station[0] - vertex[0]) ** 2 + (station[1] - vertex[1]) ** 2 <= 1000000
            for vertex in vertices)]
        hull = convex_hull(nearby)
        if len(hull) < 3:
            return None
        margins = [cross(hull[i], hull[(i + 1) % len(hull)], vertex)
                   for i in range(len(hull)) for vertex in vertices]
        if min(margins) <= 0:
            return None
        radius_slack = min(1000000 - (station[0] - vertex[0]) ** 2 - (station[1] - vertex[1]) ** 2
                           for station in hull for vertex in vertices)
        result.append({'negative_station_hull': [[float(x) for x in point] for point in hull],
                       'residual_vertices': [[float(x) for x in point] for point in vertices],
                       'minimum_strict_hull_cross_m2': float(min(margins)),
                       'minimum_radius_squared_slack_m2': float(radius_slack),
                       'arithmetic': 'exact Fraction of stored binary floating-point coordinates'})
    return result


def _check_absence_certificates(belief: Belief) -> tuple[int, list]:
    """Rebuild absent support from actual same-channel negative evidence."""
    checked = 0
    exact_witnesses = []
    for channel, state in belief.channels.items():
        if state.status != "absent_certified":
            continue
        _require(state.region.is_empty, f"Channel {channel} has a nonempty absence region")
        _require(all(obs.result in ("no_signal", "no_target_in_range") for obs in state.history),
                 f"Channel {channel} absence conflicts with positive evidence")
        negatives = tuple(sorted(set(tuple(obs.action.position) for obs in state.history
                                     if obs.result == "no_signal")))
        support = DOMAIN.difference(full_exclusion(negatives))
        for obs in state.history:
            if obs.result == "no_target_in_range":
                support = support.difference(disk(obs.action.position, 20))
        # Never substitute small area, posterior mass, or sample disappearance
        # for an empty conservative geometric support.
        if not support.is_empty:
            witnesses = _exact_surround_certificate(support, negatives)
            _require(witnesses is not None, f"Channel {channel} lacks a complete geometric absence certificate")
            exact_witnesses.append({'channel': channel, 'union_residual_area_m2': support.area,
                                    'components': witnesses})
        checked += 1
    return checked, exact_witnesses


def audit_episode(world_manifest: Mapping[str, Any],
                  events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a full or truncated local episode, returning JSON-compatible data.

    ``events`` contain request_id, action=asdict(Action), response, and optional
    move_m / executor. Rejected requests, including a returned clock of zero,
    do not advance the simulator or public state. Repeated accepted IDs are
    replayed from the original response and never charged twice. Only wall
    timestamps are ignored when checking saved responses. A duplicate's
    optional move_m may be zero or repeat the original action annotation.
    Rejections do not reserve IDs: the attachment permits corrected requests
    to reuse an ID when an earlier request was not accepted.

    An invalid event stops the audit with ``ok=False`` and an indexed error.
    A valid but unfinished prefix has ``ok=True, complete=False``. In either
    case ``seconds_per_source`` and ``completed_virtual_s`` are None, so a
    truncated mission cannot be mistaken for a low-cost completed mission.
    The spent cost and checked-prefix length remain available for diagnostics.
    """
    belief = Belief()
    sources: dict[int, Source] = {}
    cleared: set[int] = set()
    errors: list[dict[str, Any]] = []
    accepted_requests: dict[str, tuple[Action, dict[str, Any]]] = {}
    accepted_movements: dict[str, float] = {}
    fixed_readings: dict[tuple[Any, ...], tuple[str, Any]] = {}
    costs = {"move_s": 0.0, "switch_s": 0.0, "measure_s": 0.0,
             "clear_success_s": 0.0, "clear_failure_s": 0.0}
    executors: Counter[str] = Counter()
    clock, total_move = 0.0, 0.0
    max_clock_error, max_bearing_error = 0.0, 0.0
    events_seen = accepted_steps = rejected = duplicates = geometry_checks = fixed_checks = 0
    absent_checked = 0
    exact_witnesses = []
    event_index: int | None = None
    request_id: str | None = None
    try:
        source_list = [Source(row["channel"], tuple(row["position"]), row["radius"],
                              row.get("heading_deg")) for row in world_manifest["sources"]]
        _require(len(source_list) <= 16, "Q4 cannot contain more than 16 sources")
        world = World(source_list, seed=world_manifest.get("seed", 0),
                      error_mode=world_manifest.get("error_mode", "iid"))
        simulator = LocalSimulator(world)
        sources = {source.channel: source for source in source_list}
        for event_index, event in enumerate(events):
            events_seen += 1
            request_id = event.get("request_id")
            _require(isinstance(request_id, str) and bool(request_id), "request_id must be a nonempty string")
            raw = event["action"]
            action = Action(raw["kind"], tuple(raw["position"]), raw["channel"])
            response = event["response"]
            _require(isinstance(response, Mapping), "response must be an object")
            _require(response.get("accepted") is True or response.get("accepted") is False,
                     "accepted must be an explicit boolean")
            if response["accepted"] is False:
                before = (belief.steps, belief.position, belief.receiver, belief.virtual_time)
                _require(not belief.apply(action, dict(response), request_id), "Rejected response was applied")
                _require(before == (belief.steps, belief.position, belief.receiver, belief.virtual_time),
                         "Rejected request changed public state")
                if "move_m" in event:
                    _require(abs(_number(event["move_m"], "move_m")) <= 1e-6,
                             "Rejected request must not record executed movement")
                rejected += 1
                continue

            if request_id in accepted_requests:
                old_action, old_response = accepted_requests[request_id]
                _require(action == old_action, "Request ID reused with changed action")
                error = _compare_response(response, old_response)
                replayed = simulator.execute(action, request_id)
                error = max(error, _compare_response(response, replayed))
                max_clock_error = max(max_clock_error, error)
                _require(not belief.apply(old_action, old_response, request_id), "Duplicate response was applied twice")
                if "move_m" in event:
                    recorded_move = _number(event["move_m"], "move_m")
                    _require(min(abs(recorded_move), abs(recorded_move - accepted_movements[request_id])) <= 1e-5,
                             "Duplicate move_m differs from zero and the original action annotation")
                duplicates += 1
                continue

            _require(not belief.done(), "New physical action was executed after public completion")
            source = sources.get(action.channel)
            active = source is not None and action.channel not in cleared
            result, operation, bearing_error = _physical_feedback(action, response, source, active)
            move = distance(belief.position, action.position)
            if "move_m" in event:
                _require(abs(_number(event["move_m"], "move_m") - move) <= 1e-5,
                         "Recorded move_m differs from executed physical distance")
            movement_seconds = round(move / 5.0, 6)
            switch = int(action.kind == "measure" and action.channel != belief.receiver)
            expected_clock = round(clock + movement_seconds + switch + operation, 6)
            clock_error = abs(expected_clock - _number(response.get("virtual_time_s"), "virtual_time_s"))
            _require(clock_error <= CLOCK_TOLERANCE_S, "Independent 5 m/s movement / operation clock mismatch")
            replayed = simulator.execute(action, request_id)
            clock_error = max(clock_error, _compare_response(response, replayed))

            if action.kind == "measure":
                key = (action.channel, point_key(action.position), active)
                observed = result, response.get("svd_deg") if result == "direction" else None
                if key in fixed_readings:
                    _require(fixed_readings[key] == observed, "Same channel and position changed fixed feedback")
                    fixed_checks += 1
                fixed_readings[key] = observed

            public_response = _without_timestamp(response)
            _require(belief.apply(action, public_response, request_id), "New accepted request was not applied")
            if result == "success":
                cleared.add(action.channel)
            for channel, truth in sources.items():
                state = belief.channels[channel]
                _require(state.status != "absent_certified", f"Real source {channel} was certified absent")
                if channel not in cleared:
                    gap = state.region.distance(Point(truth.position))
                    _require(not state.region.is_empty and gap <= GEOMETRY_TOLERANCE_M,
                             f"True uncleared source {channel} left the conservative public support")
                    geometry_checks += 1
            _require(belief.cleared == cleared, "Public cleared channels disagree with successful clear feedback")
            _require(belief.position == simulator.position and belief.receiver == simulator.receiver,
                     "Public robot position / receiver disagrees with simulator")
            accepted_requests[request_id] = action, public_response
            accepted_movements[request_id] = move
            accepted_steps += 1
            executors[str(event.get("executor", "unspecified"))] += 1
            clock, total_move = expected_clock, total_move + move
            max_clock_error = max(max_clock_error, clock_error)
            if bearing_error is not None:
                max_bearing_error = max(max_bearing_error, bearing_error)
            costs["move_s"] += movement_seconds
            costs["switch_s"] += switch
            if action.kind == "measure":
                costs["measure_s"] += operation
            else:
                costs["clear_success_s" if result == "success" else "clear_failure_s"] += operation

        absent_checked, exact_witnesses = _check_absence_certificates(belief)
        public_complete = len(cleared) == 16 or all(
            state.status in ("cleared", "absent_certified") for state in belief.channels.values())
        _require(belief.done() == public_complete, "Completion does not match the public geometric/count certificate")
        _require(not public_complete or cleared == set(sources), "Public certificate ended with real sources uncleared")
        _require(abs(sum(costs.values()) - clock) <= CLOCK_TOLERANCE_S,
                 "Independent physical cost components do not sum to the total clock")
    except Exception as exc:
        errors.append({"event_index": event_index, "request_id": request_id,
                       "type": type(exc).__name__, "message": str(exc)})

    ok = not errors
    all_cleared = cleared == set(sources)
    certified_complete = belief.done()
    complete = ok and all_cleared and certified_complete
    source_count = len(sources)
    return {"ok": ok, "complete": complete, "status": "complete" if complete else "incomplete" if ok else "invalid",
            "audit_scope": "frozen_local_q4_simulator", "official_validation": False,
            "world_distribution_is_local_assumption": True,
            "certified_complete": certified_complete, "all_cleared": all_cleared,
            "source_count": source_count, "problem_source_count_valid": 10 <= source_count <= 16,
            "cleared_count": len(cleared), "events_seen": events_seen, "accepted_steps": accepted_steps,
            "rejected_requests": rejected, "duplicate_requests": duplicates,
            "spent_virtual_s": clock, "completed_virtual_s": clock if complete else None,
            "seconds_per_source": clock / source_count if complete and source_count else None,
            "move_m": total_move, "costs": costs, "executor_steps": dict(executors),
            "max_clock_error_s": max_clock_error, "max_bearing_error_deg": max_bearing_error,
            "truth_support_checks": geometry_checks, "fixed_point_checks": fixed_checks,
            "absence_certificates_checked": absent_checked,
            "exact_surround_certificates": exact_witnesses,
            "receiver": belief.receiver, "position": list(belief.position),
            "channel_statuses": {str(channel): state.status for channel, state in belief.channels.items()},
            "errors": errors}
