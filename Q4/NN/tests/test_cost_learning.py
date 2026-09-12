"""Public replay, survey semantics, late coverage and complete cost labels."""
from dataclasses import asdict
import gzip
import json
import math

import numpy as np
import pytest
import torch

from nnq4.bridge import Action, Belief, Config
from nnq4.cost_learning import (collect_cost_labels, make_rollout_jobs, paired_costs,
                               replay_snapshots, select_candidates, stratified_decisions,
                               _label_snapshot, _pack_labels)
from nnq4.state import Frame, MenuPlanner
from nnq4.training import EpisodeDataset
from q4.simulator import LocalSimulator, Source, World


def small_frame():
    actions = [Action("measure", (0.0, 0.0), 1), Action("clear", (10.0, 0.0), 2),
               Action("measure", (20.0, 0.0), 3), Action("measure", (100.0, 0.0), 4)]
    return Frame(torch.zeros(2, 32), torch.zeros(4, 32), torch.zeros(16),
                 actions, [True, False, False, True], 0, "test")


def write_events(path, events):
    with gzip.open(path, "wt") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")


@pytest.fixture(scope="module")
def public_mission(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cost_public")
    # This fixture constructs its own world; collection receives only its public
    # log. Co-located omnidirectional sources make terminal continuation cheap.
    environment = LocalSimulator(World([Source(channel, (0.0, 0.0), 1200.0) for channel in range(1, 17)], seed=919))
    belief, planner, events, decision_steps = Belief(), MenuPlanner(), [], []
    belief.deadline = math.inf
    snapshots = {}
    for step in range(100):
        if belief.done():
            break
        action = planner.pending_action(belief)
        if action is None:
            frame = planner.frame(belief)
            selected = frame.target
            action = frame.actions[selected]
            snapshots[step] = {"points": dict(planner.points), "shared_checked": planner.shared_checked,
                               "completion_mode": planner.completion_mode, "frame": frame}
            planner.commit(action, frame.survey[selected])
            event = {"executor": "teacher", "survey_macro": frame.survey[selected], "selected": selected}
            decision_steps.append(step)
        else:
            event = {"executor": "macro_continuation"}
        response = environment.execute(action, f"synthetic-{step}")
        event.update(action=asdict(action), response=response, request_id=f"synthetic-{step}")
        belief.apply(action, response, event["request_id"])
        events.append(event)
    assert belief.done() and environment._world.score()["all_cleared"]
    write_events(directory / "public.jsonl.gz", events)
    # Collector only uses the original NPZ's location; no truth file is valid.
    np.savez(directory / "episode.npz", marker=np.array([1]))
    (directory / "world.json").write_bytes(b"\xffDO NOT READ EVALUATOR TRUTH")
    record = {"path": str(directory / "episode.npz"), "public_path": str(directory / "public.jsonl.gz"),
              "complete": True, "split": "train", "seed": 919, "source_group": "synthetic-public-919"}
    return record, events, decision_steps, snapshots


def test_stratification_includes_first_and_last_fifths(tmp_path):
    log = tmp_path / "public.jsonl.gz"
    write_events(log, [{"executor": "teacher"} for _ in range(101)])
    selected, diagnostics = stratified_decisions([{"public_path": str(log)}], 10, 22)
    assert len(selected[0]) == 10
    assert list(diagnostics["selected_by_stage"].values()) == [2, 2, 2, 2, 2]
    assert min(selected[0]) < 20
    assert max(selected[0]) >= 80
    assert stratified_decisions([{"public_path": str(log)}], 10, 22) == (selected, diagnostics)


def test_candidate_selection_keeps_teacher_and_local_survey_diversity():
    frame = small_frame()

    class FixedNetwork:
        def __call__(self, *inputs):
            return {"scores": torch.tensor([[0.0, 1.0, 3.0, 2.0]])}

    indices, diagnostics = select_candidates(frame, 4, torch.Generator().manual_seed(2), FixedNetwork())
    assert indices[:2] == [0, 2]
    assert len(set(indices)) == 4
    assert diagnostics["selected_local"] == diagnostics["selected_survey"] == 2


def test_rollout_batch_is_committed_for_each_candidate_and_world_is_shared():
    frame = small_frame()
    snapshot = {"frame": frame, "belief": object(), "config": Config(), "points": {1: (1.0, 2.0)},
                "shared_checked": 3, "completion_mode": False}
    worlds = [object(), object()]
    jobs = make_rollout_jobs(snapshot, worlds, [0, 1], max_steps=50, wall_limit_s=2)
    assert jobs[0]["batch"] == frame.actions[0].position
    assert jobs[1]["batch"] is None
    assert jobs[0]["world"] is jobs[1]["world"] is worlds[0]
    assert jobs[2]["world"] is jobs[3]["world"] is worlds[1]
    assert jobs[0]["points"] == snapshot["points"]
    assert jobs[0]["points"] is not snapshot["points"]


def test_one_incomplete_or_missing_continuation_rejects_whole_state():
    results = [{"world_index": world, "candidate_index": candidate, "complete": True, "cost_s": 100 + world + candidate}
               for world in range(2) for candidate in (3, 5)]
    assert paired_costs(results, 2, [3, 5]).tolist() == [[103, 105], [104, 106]]
    assert paired_costs(results[:-1], 2, [3, 5]) is None
    assert paired_costs(results + [results[0]], 2, [3, 5]) is None
    results[-1] = {**results[-1], "complete": False, "cost_s": 0.0}
    assert paired_costs(results, 2, [3, 5]) is None


def test_public_replay_restores_macro_planner_and_proves_completion(public_mission):
    record, events, decisions, expected = public_mission
    selected = [decisions[0], decisions[-1]]
    snapshots, diagnostics = replay_snapshots(record, selected)
    assert diagnostics["public_complete"]
    assert diagnostics["actions"] == len(events)
    assert snapshots[-1]["progress"] == 1.0
    assert snapshots[-1]["cleared"] == 15
    for snapshot in snapshots:
        original = expected[snapshot["step"]]
        assert snapshot["points"] == original["points"]
        assert snapshot["shared_checked"] == original["shared_checked"]
        assert snapshot["completion_mode"] == original["completion_mode"]
        assert torch.equal(torch.as_tensor(snapshot["frame"].candidates), torch.as_tensor(original["frame"].candidates))


def test_real_terminal_paired_continuations_and_training_npz(public_mission, tmp_path):
    record, _, decisions, _ = public_mission
    snapshots, _ = replay_snapshots(record, [decisions[-1]])
    settings = {"candidates": 4, "worlds": 2, "sampling_attempts": 2,
                "max_steps": 10, "wall_limit_s": 5.0}
    label, diagnostics = _label_snapshot(snapshots[0], settings, torch.Generator().manual_seed(777), None)
    assert label is not None
    assert diagnostics["retained"] and diagnostics["status"] == "complete"
    assert len(diagnostics["rollouts"]) == 2
    assert all(result["complete"] and result["steps"] == 1 for result in diagnostics["rollouts"])
    assert all("hypothetical_world" in sample for sample in diagnostics["sampling"] if sample["accepted"])
    path = tmp_path / "costs.npz"
    _pack_labels([label], path)
    dataset = EpisodeDataset([{**record, "path": str(path)}])
    state = dataset[0]
    assert state["q_mask"].sum() == 1
    assert state["remaining"].item() == pytest.approx(5.0)
    assert state["q_costs"][state["target"]].item() == pytest.approx(5.0)


def test_collector_rejects_validation_and_truncated_public_data(public_mission, tmp_path):
    record, events, decisions, _ = public_mission
    with pytest.raises(ValueError, match="complete training"):
        collect_cost_labels([{**record, "split": "validation"}], tmp_path / "forbidden", workers=1)
    log = tmp_path / "truncated.jsonl.gz"
    write_events(log, events[:-1])
    with pytest.raises(ValueError, match="complete mission certificate"):
        replay_snapshots({**record, "public_path": str(log)}, [decisions[0]])


def test_collector_writes_complete_replay_manifest(public_mission, tmp_path, monkeypatch):
    from nnq4 import cost_learning
    record, _, decisions, _ = public_mission
    # Restrict this integration check to one real terminal public state; the
    # separate stratification test verifies selection across the whole mission.
    monkeypatch.setattr(cost_learning, "stratified_decisions", lambda *args: (
        {0: [decisions[-1]]}, {"selected_states": 1, "selected_by_stage": {"80-100%": 1}}))
    output = collect_cost_labels([record], tmp_path / "labeled", states=1, worlds=2,
                                 candidates=4, workers=1, max_steps=10, wall_limit_s=5)
    document = json.loads((output / "manifest.json").read_text())
    assert len(document["records"]) == 1
    saved = document["records"][0]
    assert saved["split"] == "train" and saved["complete"] is True
    assert saved["source_group"] == record["source_group"]
    assert len(EpisodeDataset([saved])) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["retained_states"] == 1
    assert summary["all_selected_states_labeled"] is True
    assert summary["real_world_files_read"] is False
    assert summary["complete_hypothetical_rollouts"] == 2
