"""Data provenance, annotation masking, and reproducible training checks."""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from nnq4.network import PolicyNetwork
from nnq4.training import EpisodeDataset, collate_states, evaluate, objective, read_records, train


def episode(path, *, length=3, nodes=2, candidates=4, seed=0, q=False, dims=(32, 32, 16)):
    rng = np.random.default_rng(seed)
    data = {
        "nodes": rng.normal(size=(length, nodes, dims[0])).astype(np.float32),
        "node_mask": np.ones((length, nodes), dtype=bool),
        "candidates": rng.normal(size=(length, candidates, dims[1])).astype(np.float32),
        "candidate_mask": np.ones((length, candidates), dtype=bool),
        "global_features": rng.normal(size=(length, dims[2])).astype(np.float32),
        "target": np.arange(length, dtype=np.int64) % candidates,
        "remaining": np.linspace(2000, 100, length).astype(np.float32),
    }
    if q:
        data["q_costs"] = np.full((length, candidates), np.nan, dtype=np.float32)
        data["q_mask"] = np.zeros((length, candidates), dtype=bool)
        data["q_costs"][:, :2] = [1000, 1200]
        data["q_mask"][:, :2] = True
    np.savez_compressed(path, **data)
    return data


def manifest(directory, records):
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"records": records}))
    return directory


def record(path, split, seed, group=None, complete=True):
    return {"path": str(path), "split": split, "seed": seed,
            "source_group": group or f"world-{seed}", "complete": complete}


def dataset_manifest(tmp_path):
    episode(tmp_path / "train.npz", length=5, seed=1)
    episode(tmp_path / "valid.npz", length=3, candidates=6, seed=2)
    return manifest(tmp_path, [record("train.npz", "train", 1), record("valid.npz", "validation", 2)])


def test_replay_checks_world_leakage_and_deduplicates(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    dataset_manifest(base)
    extra = tmp_path / "replay"
    extra.mkdir()
    episode(extra / "again.npz")
    replay = manifest(extra, [record("again.npz", "train", 200, group="world-2")])
    with pytest.raises(ValueError, match="split leakage"):
        read_records(base, [replay])
    manifest(extra, [record("again.npz", "train", 200, group="world-1")])
    grouped = read_records(base, [replay, replay])
    assert len(grouped["train"]) == 2
    assert len(grouped["validation"]) == 1
    assert grouped["validation"][0]["seed"] == 2


def test_rejects_incomplete_trajectory(tmp_path):
    dataset_manifest(tmp_path)
    content = json.loads((tmp_path / "manifest.json").read_text())
    content["records"][0]["complete"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Incomplete trajectory"):
        read_records(tmp_path)


def test_dynamic_padding_preserves_label_indices_and_unknown_q(tmp_path):
    left = episode(tmp_path / "left.npz", length=2, nodes=3, candidates=5, q=True)
    left["candidate_mask"][:, 2:4] = False
    left["target"][:] = 4
    np.savez(tmp_path / "left.npz", **left)
    episode(tmp_path / "right.npz", length=4, nodes=1, candidates=2)
    data = EpisodeDataset([record(tmp_path / "left.npz", "train", 0),
                           record(tmp_path / "right.npz", "train", 1)])
    assert len(data) == 6
    batch = collate_states([data[0], data[5]])
    assert batch["nodes"].shape == (2, 3, 32)
    assert batch["candidates"].shape == (2, 5, 32)
    assert batch["target"].tolist() == [4, 1]
    assert batch["candidate_mask"].tolist() == [[True, True, False, False, True], [True, True, False, False, False]]
    assert batch["q_mask"].sum().item() == 2
    assert torch.isfinite(batch["q_costs"]).all()
    assert not batch["q_mask"][1].any()
    # A short-only minibatch shrinks independently of the largest episode.
    assert collate_states([data[5]])["candidates"].shape[1] == 2


def test_rejects_labels_and_q_annotations_on_padding(tmp_path):
    arrays = episode(tmp_path / "bad.npz", q=True)
    arrays["candidate_mask"][:, 0] = False
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="label points to padding"):
        EpisodeDataset([record(tmp_path / "bad.npz", "train", 0)])
    arrays["target"][:] = -1
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="Q annotation refers to padded"):
        EpisodeDataset([record(tmp_path / "bad.npz", "train", 0)])


def test_q_loss_ignores_all_unannotated_candidates_and_negative_bc_labels():
    scores = torch.tensor([[0.0, 1.0, 100.0, 999.0]], requires_grad=True)
    q = torch.tensor([[2.0, 1.0, -1e20, float("nan")]], requires_grad=True)
    value = torch.tensor([0.5], requires_grad=True)
    batch = {"target": torch.tensor([-1]), "remaining": torch.tensor([500.0]),
             "candidate_mask": torch.tensor([[True, True, True, False]]),
             "q_costs": torch.tensor([[1000.0, 2000.0, float("nan"), float("nan")]]),
             "q_mask": torch.tensor([[True, True, False, False]])}
    loss, parts = objective((scores, q, value), batch)
    assert parts["bc_loss"].item() == 0
    assert parts["q_loss"].item() == pytest.approx(0.5)
    assert parts["pair_loss"].item() == pytest.approx(1.5)
    assert loss.item() == pytest.approx(0.2)
    loss.backward()
    assert q.grad[0, 2:].tolist() == [0.0, 0.0]
    assert torch.isfinite(q.grad).all()


def test_cost_regret_reports_missing_greedy_annotation(tmp_path):
    arrays = episode(tmp_path / "costs.npz", length=1, candidates=4, q=True)
    arrays["candidates"][:, :, 0] = [1, 2, 3, 100]  # Last action is padding.
    arrays["candidates"][:, :, 1] = [1, 2, -100, -100]
    arrays["candidate_mask"][:, 3] = False
    arrays["target"][:] = 2
    np.savez(tmp_path / "costs.npz", **arrays)
    data = EpisodeDataset([record(tmp_path / "costs.npz", "validation", 0)])

    class FixedNetwork(torch.nn.Module):
        def forward(self, nodes, node_mask, candidates, candidate_mask, global_features):
            return {"scores": candidates[:, :, 0], "q": candidates[:, :, 1], "value": global_features[:, 0]}

    metrics = evaluate(FixedNetwork(), DataLoader(data, batch_size=1, collate_fn=collate_states))
    assert metrics["candidate_top1"] == 1.0
    assert metrics["candidate_top5"] == 1.0
    assert metrics["q_restricted_cost_regret_s"] == 0
    assert metrics["policy_restricted_cost_regret_s"] == 200
    assert metrics["policy_greedy_cost_coverage"] == 0
    assert metrics["policy_greedy_observed_cost_regret_s"] is None


def test_cost_policy_supervision_updates_only_paired_candidates():
    scores = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    batch = {"target": torch.tensor([-1]), "remaining": torch.tensor([0.0]),
             "candidate_mask": torch.ones(1, 3, dtype=torch.bool),
             "q_costs": torch.tensor([[1000.0, 2000.0, float("nan")]]),
             "q_mask": torch.tensor([[True, True, False]])}
    outputs = {"scores": scores, "q": torch.zeros(1, 3), "value": torch.zeros(1)}
    loss, _ = objective(outputs, batch, value_weight=0, q_weight=0, pair_weight=0,
                        cost_policy_weight=1)
    loss.backward()
    assert scores.grad[0, 0] < 0  # Gradient descent favors the cheaper annotated action.
    assert scores.grad[0, 1] > 0
    assert scores.grad[0, 2] == 0


def test_training_is_reproducible_and_preserves_best_and_last(tmp_path):
    data_path = dataset_manifest(tmp_path)
    network_config = {"d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32, "dropout": 0.1}
    common = dict(epochs=2, batch_size=3, seed=19, device="cpu", threads=1,
                  network_config=network_config)
    first = train(data_path, tmp_path / "first", **common)
    second = train(data_path, tmp_path / "second", **common)
    one = torch.load(first / "last.pt", weights_only=True)
    two = torch.load(second / "last.pt", weights_only=True)
    assert one["epoch"] == two["epoch"] == 2
    assert one["validation"] == two["validation"]
    assert all(torch.equal(value, two["model"][key]) for key, value in one["model"].items())
    restored = PolicyNetwork(**one["config"])
    restored.load_state_dict(one["model"])
    for name in ("best.pt", "last.pt", "best-offline-validation.json", "last-offline-validation.json", "config.json"):
        assert (first / name).is_file()
    summary = json.loads((first / "training-summary.json").read_text())
    assert summary["test_evaluated"] is False
    assert summary["epochs_completed"] == 2
    assert summary["train_states"] == 5
    with pytest.raises(FileExistsError, match="overwrite"):
        train(data_path, first, **common)


def test_v2_dimensions_are_preserved_and_mixed_schemas_rejected(tmp_path):
    arrays = episode(tmp_path / "v2.npz", dims=(40, 48, 20), nodes=3, candidates=5)
    episode(tmp_path / "v1.npz")
    new_record, old_record = record(tmp_path / "v2.npz", "train", 2), record(tmp_path / "v1.npz", "train", 1)
    new, old = EpisodeDataset([new_record]), EpisodeDataset([old_record])
    assert new.feature_dims == {"node_dim": 40, "candidate_dim": 48, "global_dim": 20}
    batch = collate_states([new[0], new[2]])
    assert batch["nodes"].shape == (2, 3, 40)
    assert batch["candidates"].shape == (2, 5, 48)
    assert batch["global_features"].shape == (2, 20)
    assert torch.equal(batch["candidates"][1, :, -1], torch.from_numpy(arrays["candidates"][2, :, -1]))
    with pytest.raises(ValueError, match="Mixed feature schemas"):
        EpisodeDataset([old_record, new_record])
    with pytest.raises(ValueError, match="Mixed feature schemas in minibatch"):
        collate_states([old[0], new[0]])
    with pytest.raises(ValueError, match="Mixed feature schema versions"):
        EpisodeDataset([{**new_record, "feature_version": "version-a"}, {**new_record, "feature_version": "version-b"}])


def test_cost_and_replay_weights_are_selective_and_reproducible(tmp_path):
    episode(tmp_path / "base.npz", length=8)
    costs = episode(tmp_path / "cost.npz", length=2, q=True)
    costs["q_mask"][1] = False
    np.savez(tmp_path / "cost.npz", **costs)
    records = [record(tmp_path / "base.npz", "train", 1),
               {**record(tmp_path / "cost.npz", "train", 2), "is_replay": True}]
    data = EpisodeDataset(records)
    weights, config = data.sampling_weights(replay_weight=3, cost_sample_weight=16)
    assert weights.tolist() == [1.0] * 8 + [48.0, 3.0]
    assert config["cost_labeled_states"] == 1 and config["replay_states"] == 2
    assert config["expected_cost_fraction"] == pytest.approx(48 / 59)
    assert config["sampler"] == "weighted_with_replacement"
    first = list(WeightedRandomSampler(weights, 1000, generator=torch.Generator().manual_seed(91)))
    second = list(WeightedRandomSampler(weights, 1000, generator=torch.Generator().manual_seed(91)))
    assert first == second
    assert 750 < first.count(8) < 870
    default, default_config = data.sampling_weights()
    assert torch.equal(default, torch.ones(10, dtype=torch.float64))
    assert default_config["sampler"] == "shuffle_without_replacement"
    with pytest.raises(ValueError, match="finite and positive"):
        data.sampling_weights(cost_sample_weight=float("nan"))


def test_bc_coefficient_scales_policy_supervision_without_changing_other_losses():
    scores = torch.tensor([[0.0, 1.0]], requires_grad=True)
    outputs = scores, torch.zeros(1, 2), torch.tensor([0.5])
    batch = {"target": torch.tensor([0]), "remaining": torch.tensor([500.0]),
             "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
             "q_mask": torch.zeros(1, 2, dtype=torch.bool), "q_costs": torch.zeros(1, 2)}
    original, _ = objective(outputs, batch)
    finetune, _ = objective(outputs, batch, bc_weight=0.2)
    assert finetune.item() == pytest.approx(original.item() * 0.2)


def test_train_checks_validation_and_replay_schema_before_model_creation(tmp_path):
    episode(tmp_path / "train.npz", dims=(40, 48, 20))
    episode(tmp_path / "valid.npz")
    manifest(tmp_path, [record("train.npz", "train", 1), record("valid.npz", "validation", 2)])
    with pytest.raises(ValueError, match="Train/validation feature schemas differ"):
        train(tmp_path, tmp_path / "invalid", epochs=1, device="cpu")
    episode(tmp_path / "valid.npz", dims=(40, 48, 20))
    extra = tmp_path / "replay"
    extra.mkdir()
    episode(extra / "v1.npz")
    manifest(extra, [record("v1.npz", "train", 3)])
    with pytest.raises(ValueError, match="Mixed feature schemas"):
        train(tmp_path, tmp_path / "invalid_replay", epochs=1, device="cpu", replay_data=[extra])


def test_train_infers_v2_dimensions_and_rejects_incompatible_initial_weights(tmp_path):
    episode(tmp_path / "train.npz", length=3, dims=(40, 48, 20), q=True)
    episode(tmp_path / "valid.npz", length=2, dims=(40, 48, 20))
    manifest(tmp_path, [record("train.npz", "train", 1), record("valid.npz", "validation", 2)])
    architecture = {"d_model": 16, "n_heads": 2, "n_layers": 1, "ffn_dim": 32}
    # One tiny CPU epoch verifies automatic construction and saved settings;
    # it is a unit check and does not train on any project experiment data.
    output = train(tmp_path, tmp_path / "v2_run", epochs=1, batch_size=2, device="cpu", threads=1,
                   network_config=architecture, bc_weight=0.2, cost_sample_weight=8)
    saved = torch.load(output / "last.pt", weights_only=True)
    assert [saved["config"][key] for key in ("node_dim", "candidate_dim", "global_dim")] == [40, 48, 20]
    assert saved["training_config"]["loss"]["bc_weight"] == 0.2
    assert saved["training_config"]["sampling"]["cost_sample_weight"] == 8
    old = PolicyNetwork(**architecture, node_dim=32, candidate_dim=32, global_dim=16)
    initial = tmp_path / "v1.pt"
    torch.save({"config": old.get_config(), "model": old.state_dict()}, initial)
    with pytest.raises(ValueError, match="Initial checkpoint feature schema mismatch"):
        train(tmp_path, tmp_path / "initial_mismatch", epochs=1, device="cpu", initial=initial,
              network_config={"node_dim": 40, "candidate_dim": 48, "global_dim": 20})
    with pytest.raises(ValueError, match="Network feature schema mismatch"):
        train(tmp_path, tmp_path / "configured_mismatch", epochs=1, device="cpu",
              network_config={"candidate_dim": 32})
