"""Complete paired hypothetical continuations from public training histories.

This module never reads an evaluator's world file. It reconstructs the public
belief and planner state, samples compatible hypothetical worlds, and retains a
state only when every selected candidate finishes every shared sampled world.
Costs are finite-prior estimates, not true-world observations or optimal Q.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np  # File interchange and the frozen WorldSampler RNG boundary.
import torch

from .bridge import Action, Belief, Config, Model, point_key
from .experiment import dump
from .network import PolicyNetwork
from .state import MenuPlanner
from q4.rollout import continue_world
from q4.sampling import WorldSampler


STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
_NETWORKS: dict[str, PolicyNetwork] = {}


def _action(value: Mapping[str, Any]) -> Action:
    return Action(value["kind"], tuple(map(float, value["position"])), int(value["channel"]))


def _action_key(action: Action) -> tuple:
    return action.kind, action.channel, point_key(action.position)


def _read_events(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    if not events:
        raise ValueError(f"Empty public action history: {path}")
    return events


def _stage(step: int, count: int) -> int:
    return min(4, int(5 * step / max(1, count - 1)))


def stratified_decisions(records: Sequence[Mapping[str, Any]], states: int,
                         seed: int) -> tuple[dict[int, list[int]], dict]:
    """Allocate across the full action timeline, explicitly including its tail."""
    buckets: list[list[tuple[int, int]]] = [[] for _ in STAGES]
    for record_index, record in enumerate(records):
        events = _read_events(record["public_path"])
        for step, event in enumerate(events):
            # Paid channels inside an already selected survey are not a fresh
            # learned decision. Fallback decisions remain eligible in the tail.
            if event.get("executor") != "macro_continuation":
                buckets[_stage(step, len(events))].append((record_index, step))
    generator = torch.Generator().manual_seed(seed)
    queues = [[bucket[index] for index in torch.randperm(len(bucket), generator=generator).tolist()]
              for bucket in buckets]
    selected: dict[int, list[int]] = {}
    counts = [0] * 5
    # Even a two-state pilot covers an early and a late stage where available.
    order = (0, 4, 2, 1, 3)
    for _ in range(min(states, sum(map(len, queues)))):
        available = [stage for stage in order if queues[stage]]
        stage = min(available, key=lambda index: counts[index])
        record_index, step = queues[stage].pop()
        selected.setdefault(record_index, []).append(step)
        counts[stage] += 1
    for values in selected.values():
        values.sort()
    return selected, {"available_by_stage": dict(zip(STAGES, map(len, buckets))),
                      "selected_by_stage": dict(zip(STAGES, counts)),
                      "available_states": sum(map(len, buckets)),
                      "selected_states": sum(counts)}


def replay_snapshots(record: Mapping[str, Any], selected_steps: Sequence[int],
                     planner_config: Mapping[str, Any] | None = None, *,
                     snapshot_features: str | None = None) -> tuple[list[dict], dict]:
    """Reproduce actual macro commits, then verify the entire public certificate."""
    events = _read_events(record["public_path"])
    selected = set(selected_steps)
    if not selected or min(selected) < 0 or max(selected) >= len(events):
        raise ValueError("Selected public steps must lie inside the complete trajectory")
    source_config = dict(record.get("planner_config", planner_config or {}))
    planner_options = {"menu": record.get("menu", "full"), "features": record.get("features", "v1")}
    if "layout" in record:
        planner_options["layout"] = record["layout"]
    planner = MenuPlanner(Config(**source_config), **planner_options)
    belief = Belief()
    belief.deadline = math.inf
    snapshots = []
    reconstructed = 0
    last_selected = max(selected)
    for step, event in enumerate(events):
        actual = _action(event["action"])
        if belief.done():
            raise ValueError("Public log contains an action after certified completion")
        if step <= last_selected:
            executor = event.get("executor", "teacher")
            if executor == "macro_continuation":
                expected = planner.pending_action(belief)
                if expected is None or _action_key(expected) != _action_key(actual):
                    raise ValueError(f"Survey continuation diverges at public step {step}")
                if step in selected:
                    raise ValueError("Survey continuation cannot be selected as a macro state")
            else:
                # NeuralPolicy asks pending_action before frame, except after
                # its macro/time guard switches to the ProbePolicy fallback.
                if executor in ("teacher", "network") and planner.pending_action(belief) is not None:
                    raise ValueError(f"Missing recorded survey continuation at public step {step}")
                frame = planner.frame(belief)
                reconstructed += 1
                key = _action_key(actual)
                matches = [index for index, action in enumerate(frame.actions) if _action_key(action) == key]
                if not matches:
                    raise ValueError(f"Executed action is absent from reconstructed menu at public step {step}")
                index = matches[0]
                if "survey_macro" in event and bool(event["survey_macro"]) != bool(frame.survey[index]):
                    raise ValueError(f"Survey macro annotation diverges at public step {step}")
                if step in selected:
                    snapshot_frame = frame
                    if snapshot_features == 'v2' and planner.features == 'v1':
                        from .features_v2 import augment_frame
                        snapshot_frame = augment_frame(frame, belief, planner)
                    elif snapshot_features not in (None, planner.features):
                        raise ValueError('Unsupported snapshot feature conversion')
                    snapshots.append({"belief": belief.clone(), "frame": snapshot_frame,
                                      "source_frame": frame,
                                      "points": dict(planner.points),
                                      "shared_checked": planner.shared_checked,
                                      "completion_mode": planner.completion_mode,
                                      "config": planner.config, "step": step,
                                      "progress": step / max(1, len(events) - 1),
                                      "stage": STAGES[_stage(step, len(events))],
                                      "actual_index": index,
                                      "known": len(belief.known), "cleared": len(belief.cleared),
                                      "detected": sum(channel.status == "detected" for channel in belief.channels.values()),
                                      "executor": executor})
                # The reference's survey flag must not affect the actual path.
                planner.commit(actual, frame.survey[index])
        belief.apply(actual, event["response"], event["request_id"])
    if not belief.done():
        raise ValueError("Public history does not provide a complete mission certificate")
    if len(snapshots) != len(selected):
        raise ValueError("Not all selected public states could be reconstructed")
    return snapshots, {"public_complete": True, "actions": len(events),
                       "reconstructed_macro_states": reconstructed,
                       "selected_steps": sorted(selected), "final_virtual_time_s": belief.virtual_time}


def _network(checkpoint: str | None) -> PolicyNetwork | None:
    if checkpoint is None:
        return None
    if checkpoint not in _NETWORKS:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("network_config", payload.get("config"))
        weights = payload.get("model", payload.get("model_state_dict"))
        net = PolicyNetwork.from_config(config)
        net.load_state_dict(weights)
        _NETWORKS[checkpoint] = net.eval()
    return _NETWORKS[checkpoint]


@torch.inference_mode()
def select_candidates(frame, limit: int, generator: torch.Generator,
                      network: PolicyNetwork | None = None) -> tuple[list[int], dict]:
    """Retain the teacher, a learned alternative, and local/survey diversity."""
    limit = min(limit, len(frame.actions))
    if limit < 1:
        raise ValueError("At least one candidate is required")
    chosen, reasons = [frame.target], {frame.target: "teacher"}

    def add(index, reason):
        if index is not None and index not in chosen and len(chosen) < limit:
            chosen.append(index)
            reasons[index] = reason

    if network is not None and len(chosen) < limit:
        arrays = frame.arrays()
        outputs = network(*(torch.as_tensor(arrays[key]).unsqueeze(0) for key in
                            ("nodes", "node_mask", "candidates", "candidate_mask", "global_features")))
        order = outputs["scores"][0].argsort(descending=True).tolist()
        add(next((index for index in order if index not in chosen), None), "network_high_score")
    for survey in (False, True):
        pool = [index for index, value in enumerate(frame.survey) if bool(value) == survey and index not in chosen]
        if pool:
            pick = int(torch.randint(len(pool), (1,), generator=generator))
            add(pool[pick], "survey_diversity" if survey else "local_diversity")
    # Fill the remaining budget with widely separated public candidate sites.
    xy = torch.tensor([action.position for action in frame.actions], dtype=torch.float64)
    while len(chosen) < limit:
        distances = torch.cdist(xy, xy[chosen]).amin(1)
        distances[chosen] = -1
        add(int(distances.argmax()), "spatial_diversity")
    coverage = {"menu_candidates": len(frame.actions), "selected_candidates": len(chosen),
                "indices": chosen, "selection_reasons": {str(index): reasons[index] for index in chosen},
                "teacher_index": frame.target,
                "selected_local": sum(not frame.survey[index] for index in chosen),
                "selected_survey": sum(bool(frame.survey[index]) for index in chosen)}
    if getattr(frame, "families", None) is not None:
        coverage["selected_families"] = [frame.families[index] for index in chosen]
    return chosen, coverage


def make_rollout_jobs(snapshot: Mapping[str, Any], worlds: Sequence[Any],
                      indices: Sequence[int], max_steps: int, wall_limit_s: float) -> list[dict]:
    """The first action's committed survey batch is candidate specific."""
    frame = snapshot["frame"]
    return [{"belief": snapshot["belief"], "world": world, "config": snapshot["config"],
             "points": dict(snapshot["points"]), "action": frame.actions[index],
             "batch": frame.actions[index].position if frame.survey[index] else None,
             "shared_checked": snapshot["shared_checked"], "completion_mode": snapshot["completion_mode"],
             "world_index": world_index, "candidate_index": index,
             "max_steps": max_steps, "wall_limit_s": wall_limit_s}
            for world_index, world in enumerate(worlds) for index in indices]


def paired_costs(results: Sequence[Mapping[str, Any]], worlds: int,
                 indices: Sequence[int]) -> torch.Tensor | None:
    """Reject the entire state if even one shared-world continuation is missing."""
    expected = {(world, candidate) for world in range(worlds) for candidate in indices}
    observed = [(result["world_index"], result["candidate_index"]) for result in results]
    if len(results) != len(expected) or len(set(observed)) != len(observed) or set(observed) != expected:
        return None
    if any(result.get("complete") is not True or result.get("cost_s") is None
           or not math.isfinite(result["cost_s"]) or result["cost_s"] < 0 for result in results):
        return None
    lookup = {(result["world_index"], result["candidate_index"]): result["cost_s"] for result in results}
    return torch.tensor([[lookup[world, candidate] for candidate in indices] for world in range(worlds)],
                        dtype=torch.float64)


def _label_snapshot(snapshot: dict, settings: Mapping[str, Any], generator: torch.Generator,
                    network: PolicyNetwork | None) -> tuple[dict | None, dict]:
    indices, coverage = select_candidates(snapshot["frame"], settings["candidates"], generator, network)
    diagnostics = {key: snapshot[key] for key in ("step", "progress", "stage", "known", "cleared", "detected", "executor")}
    diagnostics.update(coverage=coverage, source="public_belief_hypothetical_worlds", sampling=[], rollouts=[])
    config = snapshot["config"]
    sampler = WorldSampler(Model(config.resolution, config.directional_prior, config.existence_prior))
    worlds = []
    for world_index in range(settings["worlds"]):
        accepted = False
        for attempt in range(settings["sampling_attempts"]):
            sample_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=generator, dtype=torch.int64))
            row = {"world_index": world_index, "attempt": attempt, "seed": sample_seed}
            try:
                # The frozen sampling implementation alone still accepts a
                # NumPy RNG; all allocation/statistics in this module use torch.
                world = sampler.sample(snapshot["belief"], np.random.default_rng(sample_seed))
                worlds.append(world)
                row["accepted"] = True
                row["hypothetical_world"] = world.manifest()
                accepted = True
            except Exception as exc:
                row.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
            diagnostics["sampling"].append(row)
            if accepted:
                break
        if not accepted:
            diagnostics.update(status="sampling_failed", retained=False)
            return None, diagnostics
    jobs = make_rollout_jobs(snapshot, worlds, indices, settings["max_steps"], settings["wall_limit_s"])
    results = [continue_world(job) for job in jobs]
    diagnostics["rollouts"] = results
    costs = paired_costs(results, settings["worlds"], indices)
    if costs is None:
        diagnostics.update(status="incomplete_paired_continuation", retained=False)
        return None, diagnostics
    means = costs.mean(0)
    teacher_column = indices.index(snapshot["frame"].target)
    winner = int(means.argmin())
    differences = costs - costs[:, teacher_column, None]
    diagnostics.update(status="complete", retained=True, candidate_mean_cost_s=means.tolist(),
                       candidate_mean_difference_to_teacher_s=differences.mean(0).tolist(),
                       paired_difference_standard_error_s=(differences.std(0, correction=1) / math.sqrt(len(worlds))).tolist()
                       if len(worlds) > 1 else None,
                       target=indices[winner], teacher_mean_cost_s=float(means[teacher_column]))
    return {"frame": snapshot["frame"], "indices": list(indices), "q_costs": means.float(),
            "target": indices[winner], "remaining": float(means[teacher_column]), "step": snapshot["step"]}, diagnostics


def _pack_labels(labels: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = sorted(labels, key=lambda row: row["step"])
    count = len(labels)
    n = max(len(row["frame"].nodes) for row in labels)
    k = max(len(row["frame"].actions) for row in labels)
    first = labels[0]['frame']
    tensors = {"nodes": torch.zeros(count, n, first.nodes.shape[-1]), "node_mask": torch.zeros(count, n, dtype=torch.bool),
               "candidates": torch.zeros(count, k, first.candidates.shape[-1]), "candidate_mask": torch.zeros(count, k, dtype=torch.bool),
               "global_features": torch.zeros(count, first.global_features.shape[-1]), "target": torch.empty(count, dtype=torch.int64),
               "remaining": torch.empty(count), "q_costs": torch.zeros(count, k),
               "q_mask": torch.zeros(count, k, dtype=torch.bool)}
    for index, row in enumerate(labels):
        frame = row["frame"]
        tensors["nodes"][index, :len(frame.nodes)] = torch.as_tensor(frame.nodes)
        tensors["node_mask"][index, :len(frame.nodes)] = True
        tensors["candidates"][index, :len(frame.actions)] = torch.as_tensor(frame.candidates)
        tensors["candidate_mask"][index, :len(frame.actions)] = True
        tensors["global_features"][index] = torch.as_tensor(frame.global_features)
        tensors["target"][index], tensors["remaining"][index] = row["target"], row["remaining"]
        tensors["q_costs"][index, row["indices"]] = row["q_costs"]
        tensors["q_mask"][index, row["indices"]] = True
    np.savez_compressed(path, **{key: value.numpy() for key, value in tensors.items()})


def _label_episode(job: Mapping[str, Any]) -> dict:
    torch.set_num_threads(1)
    record, settings, index = job["record"], job["settings"], job["index"]
    folder = Path(job["out"]) / f"episode-{index:05d}"
    folder.mkdir(parents=True, exist_ok=False)
    diagnostics = {"input_record": record, "selected_steps": job["steps"], "states": [], "retained_states": 0}
    started = time.monotonic()
    try:
        snapshots, replay = replay_snapshots(record, job["steps"], settings["planner_config"])
        diagnostics["replay"] = replay
        network = _network(settings["checkpoint"])
        labels = []
        for snapshot in snapshots:
            digest = hashlib.sha256(f"{settings['seed']}:{record['source_group']}:{snapshot['step']}".encode()).digest()
            state_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
            generator = torch.Generator().manual_seed(state_seed)
            label, detail = _label_snapshot(snapshot, settings, generator, network)
            detail["state_seed"] = state_seed
            diagnostics["states"].append(detail)
            if label is not None:
                labels.append(label)
            dump(folder / "diagnostics.json", diagnostics)
        if labels:
            path = folder / "episode.npz"
            _pack_labels(labels, path)
            result_record = {key: record[key] for key in ("split", "seed", "source_group")}
            result_record.update({key: record[key] for key in ("menu", "layout", "features", "planner_config") if key in record})
            result_record.update(path=str(path.resolve()), complete=True,
                                 label_kind="complete_paired_hypothetical_continuations",
                                 original_public_path=record["public_path"])
            diagnostics["record"] = result_record
        diagnostics.update(status="complete", retained_states=len(labels))
    except Exception as exc:
        diagnostics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    diagnostics["real_time_s"] = time.monotonic() - started
    dump(folder / "diagnostics.json", diagnostics)
    return diagnostics


def collect_cost_labels(episode_records: Sequence[Mapping[str, Any]] | str | Path, out: str | Path,
                        states: int = 128, worlds: int = 2, candidates: int = 4,
                        workers: int = 16, seed: int = 42, checkpoint: str | Path | None = None, *,
                        planner_config: Mapping[str, Any] | None = None,
                        max_steps: int = 6000, wall_limit_s: float = 45.0,
                        sampling_attempts: int = 3) -> Path:
    """Collect finite-prior cost labels without reading or using real world truth.

    A manifest path is filtered to train; an explicit record sequence must contain
    train records only. ``out`` must be a fresh output directory. Parallelism is
    across selected episodes, with one numerical thread per worker. Every failed
    selected state is recorded and is not replaced by an easier sampled state.
    """
    if min(states, worlds, candidates, workers, max_steps, sampling_attempts) < 1 or wall_limit_s <= 0:
        raise ValueError("Collection sizes, limits and worker count must be positive")
    if isinstance(episode_records, (str, Path)):
        manifest = Path(episode_records).resolve()
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
        records = [dict(record) for record in json.loads(manifest.read_text())["records"] if record["split"] == "train"]
        for record in records:
            record["path"] = str((manifest.parent / record["path"]).resolve())
    else:
        records = [dict(record) for record in episode_records]
    if not records:
        raise ValueError("At least one complete training record is required")
    for record in records:
        if record.get("split") != "train" or record.get("complete") is not True:
            raise ValueError("Only complete training worlds may supply cost labels")
        if not isinstance(record.get("source_group"), str) or not record["source_group"]:
            raise ValueError("Records must preserve the original world source_group")
        if "seed" not in record:
            raise ValueError("Records must preserve the original seed")
        path = Path(record["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        record["path"] = str(path)
        record["public_path"] = str(path.parent / "public.jsonl.gz")
        if not Path(record["public_path"]).is_file():
            raise FileNotFoundError(record["public_path"])
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(states=states, worlds=worlds, candidates=candidates, workers=workers, seed=seed,
                    checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                    planner_config=dict(planner_config or {}), max_steps=max_steps,
                    wall_limit_s=wall_limit_s, sampling_attempts=sampling_attempts,
                    prior="frozen Q4 quadrature model; compatible hypothetical worlds only",
                    pair_rejection="Reject entire selected state if any selected candidate/world is incomplete")
    selected, sampling = stratified_decisions(records, states, seed)
    if not selected:
        raise ValueError("Training public histories contain no eligible macro decisions")
    dump(output / "config.json", settings)
    dump(output / "state-selection.json", sampling)
    dump(output / "input-records.json", records)
    rows = []

    def record_result(row):
        rows.append(row)
        valid = [value["record"] for value in rows if value.get("record")]
        details = [state for value in rows for state in value.get("states", [])]
        summary = {"status": "running", "requested_states": states,
                   "selected_states": sampling["selected_states"], "episodes_planned": len(selected),
                   "episodes_finished": len(rows), "episodes_failed": sum(value["status"] == "failed" for value in rows),
                   "states_processed": len(details), "retained_states": sum(value["retained_states"] for value in rows),
                   "state_status": dict(Counter(value["status"] for value in details)),
                   "retained_by_stage": dict(Counter(value["stage"] for value in details if value["retained"])),
                   "hypothetical_rollouts": sum(len(value["rollouts"]) for value in details),
                   "complete_hypothetical_rollouts": sum(result["complete"] for value in details for result in value["rollouts"]),
                   "sampling_failures": sum(not result["accepted"] for value in details for result in value["sampling"]),
                   "real_world_files_read": False, "test_worlds_used": False}
        dump(output / "manifest.json", {"records": sorted(valid, key=lambda value: value["path"])})
        dump(output / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("episodes_finished", "episodes_planned", "states_processed", "retained_states")}), flush=True)

    jobs = [dict(index=index, record=records[index], steps=steps, settings=settings, out=str(output))
            for index, steps in sorted(selected.items())]
    if workers == 1:
        for job in jobs:
            record_result(_label_episode(job))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(_label_episode, job) for job in jobs]):
                record_result(future.result())
    summary = json.loads((output / "summary.json").read_text())
    summary["status"] = "complete"
    summary["all_selected_states_labeled"] = summary["retained_states"] == sampling["selected_states"]
    dump(output / "summary.json", summary)
    return output
