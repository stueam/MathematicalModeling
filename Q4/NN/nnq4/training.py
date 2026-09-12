"""Reproducible candidate-policy training on complete, episode-grouped data.

Each manifest contains ``records`` with path, split, seed, source_group and
complete fields. Paths are relative to the manifest. ``source_group`` identifies
the underlying world, including across replay collections; all its trajectories
must stay in one split. Validation chooses a checkpoint, never a simulation score.
"""
from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .network import PolicyNetwork


SPLITS = {"train", "validation", "test"}
REQUIRED = ("nodes", "node_mask", "candidates", "candidate_mask",
            "global_features", "target", "remaining")


def _dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_records(data_path: str | Path,
                 replay_data: Sequence[str | Path] = ()) -> dict[str, list[dict]]:
    """Validate world-level splits globally; append replay training records only."""
    grouped: dict[str, list[dict]] = {split: [] for split in SPLITS}
    world_splits: dict[str, str] = {}
    path_splits: dict[Path, str] = {}
    selected: set[Path] = set()
    for index, location in enumerate((data_path, *replay_data)):
        manifest = Path(location).resolve()
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
        document = json.loads(manifest.read_text())
        records = document.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Manifest requires a records list: {manifest}")
        for raw in records:
            missing = {"path", "split", "seed", "source_group", "complete"} - raw.keys()
            if missing:
                raise ValueError(f"Missing record fields {sorted(missing)}: {manifest}")
            record = dict(raw)
            split = record["split"]
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r}: {manifest}")
            if record["complete"] is not True:
                raise ValueError(f"Incomplete trajectory cannot supply remaining cost: {record['path']}")
            world = record["source_group"]
            if not isinstance(world, str) or not world:
                raise ValueError("source_group must be a nonempty world identifier")
            path = (manifest.parent / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if world in world_splits and world_splits[world] != split:
                raise ValueError(f"Episode split leakage for source_group {world!r}")
            if path in path_splits and path_splits[path] != split:
                raise ValueError(f"Episode split leakage for file {path}")
            world_splits[world] = path_splits[path] = split
            if (index == 0 or split == "train") and path not in selected:
                record.update(path=str(path), manifest=str(manifest), is_replay=index > 0)
                grouped[split].append(record)
                selected.add(path)
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("Separate, nonempty train and validation episode splits are required")
    return grouped


class EpisodeDataset(Dataset):
    """Keep episodes unpadded in host memory; pad only the requested minibatch."""

    def __init__(self, records: Sequence[Mapping[str, Any]]):
        self.records = [dict(record) for record in records]
        self.episodes: list[dict[str, Tensor]] = []
        self.feature_dims: dict[str, int] | None = None
        self.feature_version: str | None = None
        lengths = []
        for record in self.records:
            if record.get("complete") is not True:
                raise ValueError("Only complete trajectories may enter training")
            with np.load(record["path"], allow_pickle=False) as archive:
                missing = set(REQUIRED) - set(archive.files)
                if missing:
                    raise ValueError(f"Missing arrays {sorted(missing)}: {record['path']}")
                episode = {key: torch.from_numpy(archive[key]) for key in REQUIRED}
                q_keys = {"q_costs", "q_mask"}.intersection(archive.files)
                if q_keys and len(q_keys) != 2:
                    raise ValueError("q_costs and q_mask must be supplied together")
                if q_keys:
                    episode.update({key: torch.from_numpy(archive[key]) for key in q_keys})
            self._validate(episode, record["path"])
            dims = {"node_dim": episode["nodes"].shape[-1],
                    "candidate_dim": episode["candidates"].shape[-1],
                    "global_dim": episode["global_features"].shape[-1]}
            if self.feature_dims is not None and self.feature_dims != dims:
                raise ValueError(f"Mixed feature schemas in episode dataset: {self.feature_dims} versus {dims}: {record['path']}")
            self.feature_dims = dims
            version = record.get("feature_version")
            if version is not None:
                if self.feature_version is not None and self.feature_version != version:
                    raise ValueError(f"Mixed feature schema versions: {self.feature_version!r} versus {version!r}")
                self.feature_version = version
            self.episodes.append(episode)
            lengths.append(len(episode["target"]))
        self.ends = list(accumulate(lengths))

    @staticmethod
    def _validate(episode: dict[str, Tensor], path: str) -> None:
        nodes, candidates = episode["nodes"], episode["candidates"]
        if nodes.ndim != 3 or nodes.shape[-1] < 1:
            raise ValueError(f"nodes must have shape (T, N, D_node) with positive feature dimension: {path}")
        if candidates.ndim != 3 or candidates.shape[-1] < 1:
            raise ValueError(f"candidates must have shape (T, K, D_candidate) with positive feature dimension: {path}")
        globals_ = episode["global_features"]
        if globals_.ndim != 2 or globals_.shape[-1] < 1:
            raise ValueError(f"global_features must have shape (T, D_global) with positive feature dimension: {path}")
        length = nodes.shape[0]
        if length < 1 or any(value.ndim == 0 or len(value) != length for value in episode.values()):
            raise ValueError(f"Nonempty arrays must agree on trajectory length: {path}")
        shapes = {"node_mask": nodes.shape[:2], "candidate_mask": candidates.shape[:2],
                  "global_features": (length, globals_.shape[-1]), "target": (length,), "remaining": (length,)}
        for key, shape in shapes.items():
            if episode[key].shape != shape:
                raise ValueError(f"Invalid shape for {key}: {path}")
        for key in ("node_mask", "candidate_mask"):
            if not ((episode[key] == 0) | (episode[key] == 1)).all():
                raise ValueError(f"{key} must be boolean: {path}")
            episode[key] = episode[key].bool()
            if not episode[key].any(axis=1).all():
                raise ValueError(f"Every state requires a valid {key}: {path}")
        for key in ("nodes", "candidates", "global_features", "remaining"):
            if not torch.isfinite(episode[key]).all():
                raise ValueError(f"Nonfinite {key}: {path}")
            episode[key] = episode[key].float()
        if (episode["remaining"] < 0).any():
            raise ValueError(f"Remaining seconds cannot be negative: {path}")
        target = episode["target"]
        if target.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"Candidate labels must be integers: {path}")
        labeled = target >= 0
        if (target[labeled] >= candidates.shape[1]).any():
            raise ValueError(f"Candidate label out of range: {path}")
        if not episode["candidate_mask"][torch.nonzero(labeled, as_tuple=True)[0], target[labeled].long()].all():
            raise ValueError(f"Candidate label points to padding: {path}")
        episode["target"] = target.long()
        if "q_costs" in episode:
            if episode["q_costs"].shape != candidates.shape[:2] or episode["q_mask"].shape != candidates.shape[:2]:
                raise ValueError(f"Q annotations must have shape (T, K): {path}")
            if not ((episode["q_mask"] == 0) | (episode["q_mask"] == 1)).all():
                raise ValueError(f"q_mask must be boolean: {path}")
            q_mask = episode["q_mask"].bool()
            if (q_mask & ~episode["candidate_mask"]).any():
                raise ValueError(f"Q annotation refers to padded candidate: {path}")
            observed = episode["q_costs"][q_mask]
            if not torch.isfinite(observed).all() or (observed < 0).any():
                raise ValueError(f"Q annotations must be finite complete remaining seconds: {path}")
            # Unannotated costs may be NaN in the collector; never propagate them.
            episode["q_costs"] = episode["q_costs"].masked_fill(~q_mask, 0).float()
            episode["q_mask"] = q_mask

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_id = bisect_right(self.ends, index)
        local = index - (int(self.ends[episode_id - 1]) if episode_id else 0)
        return {key: value[local] for key, value in self.episodes[episode_id].items()}

    def sampling_weights(self, replay_weight: float = 1.0,
                         cost_sample_weight: float = 1.0) -> tuple[Tensor, dict[str, Any]]:
        """Weight actual cost annotations; keep the epoch's number of draws fixed.

        Replay and cost factors multiply when a state belongs to both groups.
        Only rows with at least one annotated Q cost receive the cost factor.
        """
        if any(not math.isfinite(value) or value <= 0 for value in (replay_weight, cost_sample_weight)):
            raise ValueError("Sampling weights must be finite and positive")
        all_weights, costs, replays = [], [], []
        for record, episode in zip(self.records, self.episodes):
            count = len(episode["target"])
            cost = episode["q_mask"].any(1) if "q_mask" in episode else torch.zeros(count, dtype=torch.bool)
            replay = torch.full((count,), bool(record.get("is_replay", False)), dtype=torch.bool)
            weight = torch.ones(count, dtype=torch.float64)
            weight[cost] *= cost_sample_weight
            weight[replay] *= replay_weight
            all_weights.append(weight)
            costs.append(cost)
            replays.append(replay)
        weights = torch.cat(all_weights) if all_weights else torch.empty(0, dtype=torch.float64)
        if not len(weights):
            raise ValueError("Cannot sample an empty dataset")
        cost_rows, replay_rows = torch.cat(costs), torch.cat(replays)
        weighted = bool((weights != weights[0]).any())
        return weights, {"replay_weight": replay_weight, "cost_sample_weight": cost_sample_weight,
                         "sampler": "weighted_with_replacement" if weighted else "shuffle_without_replacement",
                         "draws_per_epoch": len(self), "cost_labeled_states": int(cost_rows.sum()),
                         "replay_states": int(replay_rows.sum()),
                         "expected_cost_fraction": float(weights[cost_rows].sum() / weights.sum()),
                         "expected_replay_fraction": float(weights[replay_rows].sum() / weights.sum())}


def collate_states(states: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    """Trim to the last valid index, preserving noncontiguous candidate indices."""
    size = len(states)
    if not size:
        raise ValueError("Cannot collate an empty batch")
    schemas = {(state["nodes"].shape[-1], state["candidates"].shape[-1], state["global_features"].shape[-1])
               for state in states}
    if len(schemas) != 1:
        raise ValueError(f"Mixed feature schemas in minibatch: {sorted(schemas)}")
    node_dim, candidate_dim, global_dim = next(iter(schemas))
    n = max(int(torch.nonzero(state["node_mask"], as_tuple=True)[0][-1]) + 1 for state in states)
    k = max(int(torch.nonzero(state["candidate_mask"], as_tuple=True)[0][-1]) + 1 for state in states)
    batch = {"nodes": torch.zeros(size, n, node_dim), "node_mask": torch.zeros(size, n, dtype=torch.bool),
             "candidates": torch.zeros(size, k, candidate_dim), "candidate_mask": torch.zeros(size, k, dtype=torch.bool),
             "global_features": torch.zeros(size, global_dim), "target": torch.empty(size, dtype=torch.long),
             "remaining": torch.empty(size), "q_costs": torch.zeros(size, k),
             "q_mask": torch.zeros(size, k, dtype=torch.bool)}
    for row, state in enumerate(states):
        for key in ("nodes", "node_mask", "candidates", "candidate_mask", "q_costs", "q_mask"):
            if key in state:
                count = min(len(state[key]), batch[key].shape[1])
                batch[key][row, :count] = torch.as_tensor(state[key][:count])
        for key in ("global_features", "target", "remaining"):
            batch[key][row] = torch.as_tensor(state[key])
    return batch


def objective(outputs: tuple[Tensor, Tensor, Tensor] | Mapping[str, Tensor], batch: Mapping[str, Tensor], *,
              remaining_scale: float = 1000.0, bc_weight: float = 1.0, value_weight: float = 0.1,
              q_weight: float = 0.1, pair_weight: float = 0.1,
              cost_policy_weight: float = 0.0,
              cost_temperature: float = 100.0) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked BC and optional complete-rollout cost regression/difference learning.

    Q predicts costs in units of remaining_scale (lower is better). Pair loss
    compares only jointly annotated actions from the same public state; neither
    padding nor unknown costs supplies a target. Cost policy supervision likewise
    normalizes exclusively over annotated candidates and requires at least two.
    """
    if isinstance(outputs, Mapping):
        scores, q, value = outputs["scores"], outputs["q"], outputs["value"]
    else:
        scores, q, value = outputs
    value = value.reshape(-1)
    zero = value.sum() * 0
    scores = scores.masked_fill(~batch["candidate_mask"], -torch.inf)
    labeled = batch["target"] >= 0
    bc = F.cross_entropy(scores[labeled], batch["target"][labeled]) if labeled.any() else zero
    value_loss = F.smooth_l1_loss(value, batch["remaining"] / remaining_scale)
    observed = batch["q_mask"] & batch["candidate_mask"]
    q_loss = F.smooth_l1_loss(q[observed], batch["q_costs"][observed] / remaining_scale) if observed.any() else zero
    paired_rows = observed.sum(1) >= 2
    pair_loss = policy_loss = zero
    if paired_rows.any():
        known = observed[paired_rows]
        costs = batch["q_costs"][paired_rows].masked_fill(~known, 0)
        prediction = q[paired_rows].masked_fill(~known, 0)
        left, right = torch.triu_indices(q.shape[1], q.shape[1], offset=1, device=q.device)
        pairs = known[:, left] & known[:, right]
        differences = F.smooth_l1_loss(prediction[:, left] - prediction[:, right],
                                     (costs[:, left] - costs[:, right]) / remaining_scale, reduction="none")
        pair_loss = ((differences * pairs).sum(1) / pairs.sum(1)).mean()
        if cost_policy_weight:
            soft_target = torch.softmax((-costs / cost_temperature).masked_fill(~known, -torch.inf), dim=1)
            log_policy = F.log_softmax(scores[paired_rows].masked_fill(~known, -torch.inf), dim=1)
            policy_loss = -(soft_target * log_policy.masked_fill(~known, 0)).sum(1).mean()
    parts = {"bc_loss": bc, "value_loss": value_loss, "q_loss": q_loss,
             "pair_loss": pair_loss, "cost_policy_loss": policy_loss}
    loss = bc_weight * bc + value_weight * value_loss + q_weight * q_loss + pair_weight * pair_loss + cost_policy_weight * policy_loss
    return loss, parts


def _forward(net: PolicyNetwork, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    output = net(batch["nodes"], batch["node_mask"], batch["candidates"],
                 batch["candidate_mask"], batch["global_features"])
    if isinstance(output, Mapping):
        return output["scores"], output["q"], output["value"]
    return output


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


@torch.inference_mode()
def evaluate(net: PolicyNetwork, loader: DataLoader, device: str | torch.device = "cpu",
             **loss_options: float) -> dict[str, Any]:
    net.eval()
    device = torch.device(device)
    total = labeled_total = top1 = top5 = cost_states = covered_states = 0
    value_error = cost_regret = restricted_regret = q_regret = q_error = 0.0
    q_labels = 0
    part_sums: dict[str, float] = {}
    part_counts: dict[str, int] = {}
    for raw in loader:
        batch = _to_device(raw, device)
        outputs = _forward(net, batch)
        _, parts = objective(outputs, batch, **loss_options)
        scores, q, value = outputs
        scores = scores.masked_fill(~batch["candidate_mask"], -torch.inf)
        count = len(batch["target"])
        total += count
        observed = batch["q_mask"] & batch["candidate_mask"]
        paired_count = int((observed.sum(1) >= 2).sum())
        counts = {"bc_loss": int((batch["target"] >= 0).sum()), "value_loss": count,
                  "q_loss": int(observed.sum()), "pair_loss": paired_count,
                  "cost_policy_loss": paired_count}
        for key, part in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + float(part) * counts[key]
            part_counts[key] = part_counts.get(key, 0) + counts[key]
        chosen = scores.argmax(1)
        labeled = batch["target"] >= 0
        labeled_total += int(labeled.sum())
        top1 += int((chosen[labeled] == batch["target"][labeled]).sum())
        top = scores.topk(min(5, scores.shape[1]), dim=1).indices
        top5 += int((top[labeled] == batch["target"][labeled, None]).any(1).sum())
        scale = loss_options.get("remaining_scale", 1000.0)
        value_error += float((value.reshape(-1) * scale - batch["remaining"]).abs().sum())
        q_labels += int(observed.sum())
        q_error += float((q[observed] * scale - batch["q_costs"][observed]).abs().sum())
        paired_rows = observed.sum(1) >= 2
        if paired_rows.any():
            known, costs = observed[paired_rows], batch["q_costs"][paired_rows]
            best = costs.masked_fill(~known, torch.inf).amin(1)
            policy_choice = scores[paired_rows].masked_fill(~known, -torch.inf).argmax(1, keepdim=True)
            q_choice = q[paired_rows].masked_fill(~known, torch.inf).argmin(1, keepdim=True)
            cost_states += int(paired_rows.sum())
            restricted_regret += float((costs.gather(1, policy_choice).squeeze(1) - best).sum())
            q_regret += float((costs.gather(1, q_choice).squeeze(1) - best).sum())
            greedy_choice = chosen[paired_rows, None]
            covered = known.gather(1, greedy_choice).squeeze(1)
            covered_states += int(covered.sum())
            cost_regret += float((costs.gather(1, greedy_choice).squeeze(1) - best)[covered].sum())
    if not total:
        raise ValueError("Cannot evaluate an empty split")
    means = {key: value / max(1, part_counts[key]) for key, value in part_sums.items()}
    loss_mean = (loss_options.get("bc_weight", 1.0) * means["bc_loss"] + loss_options.get("value_weight", 0.1) * means["value_loss"]
                 + loss_options.get("q_weight", 0.1) * means["q_loss"]
                 + loss_options.get("pair_weight", 0.1) * means["pair_loss"]
                 + loss_options.get("cost_policy_weight", 0.0) * means["cost_policy_loss"])
    return {"states": total, "labeled_states": labeled_total,
            "loss": loss_mean, **means,
            "candidate_top1": top1 / labeled_total if labeled_total else None,
            "candidate_top5": top5 / labeled_total if labeled_total else None,
            "remaining_time_mae_s": value_error / total,
            "q_labeled_candidates": q_labels, "q_cost_mae_s": q_error / q_labels if q_labels else None,
            "paired_cost_states": cost_states,
            "q_restricted_cost_regret_s": q_regret / cost_states if cost_states else None,
            "policy_restricted_cost_regret_s": restricted_regret / cost_states if cost_states else None,
            "policy_greedy_cost_coverage": covered_states / cost_states if cost_states else None,
            "policy_greedy_observed_cost_regret_s": cost_regret / covered_states if covered_states else None}


def train(data_path: str | Path, out: str | Path, epochs: int = 12, batch_size: int = 64,
          seed: int = 42, initial: str | Path | None = None, device: str = "cuda", *,
          replay_data: Sequence[str | Path] = (), network_config: Mapping[str, Any] | None = None,
          learning_rate: float = 3e-4, weight_decay: float = 1e-4, threads: int = 4,
          remaining_scale: float = 1000.0, bc_weight: float = 1.0, value_weight: float = 0.1,
          q_weight: float = 0.1, pair_weight: float = 0.1, cost_policy_weight: float = 0.0,
          cost_temperature: float = 100.0, gradient_clip: float = 1.0,
          replay_weight: float = 1.0, cost_sample_weight: float = 1.0) -> Path:
    """Train for every requested epoch; retain last and best-validation weights.

    ``initial`` initializes weights, not optimizer state. Replay manifests append
    their train trajectories; the original validation/test split stays fixed.
    Input feature dimensions come from the dataset and must match every loaded
    episode and any initial checkpoint. Cost/replay weights change sampling;
    cost_policy_weight trains policy logits and bc_weight scales imitation.
    GPU unavailability raises explicitly: no silent CPU fallback or early stop.
    """
    if epochs < 1 or batch_size < 1 or threads < 1:
        raise ValueError("epochs, batch_size and threads must be positive")
    if remaining_scale <= 0 or cost_temperature <= 0 or learning_rate <= 0 or gradient_clip <= 0:
        raise ValueError("Scales, learning rate and gradient clip must be positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (bc_weight, value_weight, q_weight, pair_weight, cost_policy_weight, weight_decay)):
        raise ValueError("Loss weights and weight decay must be finite and nonnegative")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run with GPU access or explicitly select device='cpu'")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    records = read_records(data_path, replay_data)
    training, validation = EpisodeDataset(records["train"]), EpisodeDataset(records["validation"])
    if training.feature_dims != validation.feature_dims:
        raise ValueError(f"Train/validation feature schemas differ: {training.feature_dims} versus {validation.feature_dims}")
    if (training.feature_version is not None and validation.feature_version is not None
            and training.feature_version != validation.feature_version):
        raise ValueError("Train/validation feature schema versions differ")
    weights, sampling_config = training.sampling_weights(replay_weight, cost_sample_weight)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("config.json", "history.json", "best.pt", "last.pt")):
        raise FileExistsError(f"Refusing to overwrite a previous training run: {output}")
    checkpoint = torch.load(initial, map_location="cpu", weights_only=True) if initial else None
    net_config = dict(checkpoint.get("network_config", checkpoint.get("config", {})) if checkpoint else {})
    if checkpoint:
        checkpoint_weights = checkpoint.get("model", checkpoint.get("model_state_dict"))
        if not isinstance(checkpoint_weights, Mapping):
            raise ValueError("Initial checkpoint must contain a model state dictionary")
        for key, projection in (("node_dim", "node_embedding.weight"),
                                ("candidate_dim", "candidate_embedding.weight"),
                                ("global_dim", "global_embedding.weight")):
            # Read actual tensor dimensions too: an architecture override must
            # never disguise an incompatible saved input projection.
            if projection in checkpoint_weights and checkpoint_weights[projection].shape[1] != training.feature_dims[key]:
                raise ValueError(f"Initial checkpoint feature schema mismatch for {key}: "
                                 f"{checkpoint_weights[projection].shape[1]} versus {training.feature_dims[key]}")
    net_config.update(dict(network_config or {}))
    for key, size in training.feature_dims.items():
        if key in net_config and net_config[key] != size:
            raise ValueError(f"Network feature schema mismatch for {key}: {net_config[key]} versus {size}")
        net_config[key] = size
    net = PolicyNetwork(**net_config).to(requested_device)
    if checkpoint:
        net.load_state_dict(checkpoint_weights, strict=True)
    # Network exports canonical defaults, so saved weights are self-describing.
    if hasattr(net, "get_config"):
        net_config = dict(net.get_config())
    elif hasattr(net, "config"):
        config_value = net.config
        net_config = dict(config_value() if callable(config_value) else config_value)
    if net_config.get("cost_scale", remaining_scale) != remaining_scale:
        raise ValueError("Network cost_scale must match training remaining_scale")
    generator = torch.Generator().manual_seed(seed)
    loader_options = dict(batch_size=batch_size, collate_fn=collate_states, num_workers=0,
                          pin_memory=requested_device.type == "cuda")
    sampler = (WeightedRandomSampler(weights, num_samples=len(training), replacement=True, generator=generator)
               if sampling_config["sampler"] == "weighted_with_replacement" else None)
    train_loader = DataLoader(training, shuffle=sampler is None, sampler=sampler, generator=generator, **loader_options)
    valid_loader = DataLoader(validation, shuffle=False, **loader_options)
    loss_options = dict(remaining_scale=remaining_scale, bc_weight=bc_weight, value_weight=value_weight,
                        q_weight=q_weight, pair_weight=pair_weight,
                        cost_policy_weight=cost_policy_weight, cost_temperature=cost_temperature)
    config = {"data_path": str(Path(data_path).resolve()), "replay_data": [str(Path(p).resolve()) for p in replay_data],
              "network": net_config, "epochs": epochs, "batch_size": batch_size, "seed": seed,
              "initial": str(Path(initial).resolve()) if initial else None, "device": str(requested_device),
              "threads": threads, "learning_rate": learning_rate, "weight_decay": weight_decay,
              "gradient_clip": gradient_clip, "loss": loss_options,
              "sampling": sampling_config, "feature_dims": training.feature_dims,
              "feature_version": training.feature_version or validation.feature_version,
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "deterministic_algorithms": True, "validation_selection": "offline_loss",
              "checkpoint_note": "Select complete-mission performance separately on development worlds; test is untouched."}
    _dump(output / "config.json", config)
    _dump(output / "data-records.json", records)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history, best = [], float("inf")
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        net.train()
        running, samples = 0.0, 0
        for raw in train_loader:
            batch = _to_device(raw, requested_device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(_forward(net, batch), batch, **loss_options)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            size = len(batch["target"])
            running += float(loss.detach()) * size
            samples += size
        metrics = evaluate(net, valid_loader, requested_device, **loss_options)
        row = {"epoch": epoch, "train_loss": running / samples,
               "validation": metrics, "elapsed_s": time.monotonic() - started}
        history.append(row)
        _dump(output / "history.json", history)
        payload = {"model": net.state_dict(), "config": net_config, "training_config": config,
                   "epoch": epoch, "validation": metrics, "optimizer": optimizer.state_dict(),
                   "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": generator.get_state()}
        torch.save(payload, output / "last.pt")
        if metrics["loss"] < best:
            best, best_epoch = metrics["loss"], epoch
            torch.save(payload, output / "best.pt")
            _dump(output / "best-offline-validation.json", metrics)
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    _dump(output / "last-offline-validation.json", history[-1]["validation"])
    _dump(output / "training-summary.json", {
        "train_episodes": len(training.records), "validation_episodes": len(validation.records),
        "train_states": len(training), "validation_states": len(validation),
        "parameters": sum(parameter.numel() for parameter in net.parameters()),
        "epochs_completed": epochs, "best_validation_epoch": best_epoch,
        "best_validation_loss": best, "training_wall_s": time.monotonic() - started,
        "test_evaluated": False, "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt")})
    return output
