"""Candidate-conditioned attention policy for the Q4 public belief state.

Inspired by the graph/node decoders in CAtNIPP and ARiADNE. Candidate
positions and channel identity are supplied as features, never as sequence
positions. Both node and candidate order are therefore arbitrary.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NetworkConfig:
    node_dim: int = 32
    candidate_dim: int = 32
    global_dim: int = 16
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 192
    dropout: float = 0.0
    candidate_attention: bool = False
    logit_clip: float = 10.0
    cost_scale: float = 1000.0

    def __post_init__(self) -> None:
        dimensions = (self.node_dim, self.candidate_dim, self.global_dim,
                      self.d_model, self.n_heads, self.n_layers, self.ffn_dim)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError("Network dimensions, heads and layers must be positive integers")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.logit_clip <= 0 or self.cost_scale <= 0:
            raise ValueError("logit_clip and cost_scale must be positive")


class _AttentionBlock(nn.Module):
    """Pre-normalized self-attention with explicit valid-token masking."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        width = config.d_model
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, config.n_heads, dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, config.ffn_dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        update, _ = self.attention(normalized, normalized, normalized,
                                   key_padding_mask=~mask, need_weights=False)
        tokens = tokens + self.dropout(update)
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens.masked_fill(~mask.unsqueeze(-1), 0.0)


class PolicyNetwork(nn.Module):
    """Score a variable number of actions against a variable public node set.

    Inputs are ``nodes[B,N,32]``, ``node_mask[B,N]``,
    ``candidates[B,K,32]``, ``candidate_mask[B,K]`` and
    ``global_features[B,16]`` by default. Masks are bool with True = valid.
    ``scores[B,K]`` are policy logits (higher is better); ``q[B,K]`` and
    ``value[B]`` predict remaining cost / ``config.cost_scale`` (lower is
    better). The Q estimate includes the candidate's immediate cost.

    Invalid policy logits are -10000 and invalid Q entries are zero; always
    apply candidate_mask to a Q loss or an argmin. A global token keeps the
    memory nonempty even when there are no valid public nodes. A state with
    no valid candidate is an error: terminal states must not call the policy.
    """

    checkpoint_version = 1

    def __init__(self, config: NetworkConfig | Mapping[str, Any] | None = None,
                 **kwargs: Any) -> None:
        super().__init__()
        if isinstance(config, NetworkConfig):
            settings = asdict(config)
        else:
            settings = dict(config or {})
        settings.update(kwargs)
        self.config = NetworkConfig(**settings)
        cfg = self.config
        width = cfg.d_model
        self.node_embedding = nn.Linear(cfg.node_dim, width)
        self.candidate_embedding = nn.Linear(cfg.candidate_dim, width)
        self.global_embedding = nn.Linear(cfg.global_dim, width)
        self.node_encoder = nn.ModuleList(_AttentionBlock(cfg) for _ in range(cfg.n_layers))
        self.memory_norm = nn.LayerNorm(width)
        self.candidate_norm = nn.LayerNorm(width)
        self.candidate_cross_attention = nn.MultiheadAttention(
            width, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.candidate_ffn_norm = nn.LayerNorm(width)
        self.candidate_ffn = nn.Sequential(
            nn.Linear(width, cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim, width))
        self.candidate_encoder = _AttentionBlock(cfg) if cfg.candidate_attention else None
        self.global_query_norm = nn.LayerNorm(width)
        self.global_cross_attention = nn.MultiheadAttention(
            width, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.global_context_norm = nn.LayerNorm(width)
        self.candidate_context_norm = nn.LayerNorm(width)
        self.pointer_query = nn.Linear(width, width, bias=False)
        self.pointer_key = nn.Linear(width, width, bias=False)
        # Each Q/logit head receives this candidate's attended evidence directly.
        self.rank_head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.q_head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.dropout = nn.Dropout(cfg.dropout)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        # Nonzero small heads let BC and cost losses reach cross-attention at
        # the first update, while initial costs and logits stay well behaved.
        for head in (self.rank_head, self.q_head, self.value_head):
            nn.init.normal_(head[-1].weight, std=0.01)
        nn.init.normal_(self.pointer_query.weight, std=0.02)
        nn.init.normal_(self.pointer_key.weight, std=0.02)

    def get_config(self) -> dict[str, Any]:
        """Return JSON-serializable architectural settings for a checkpoint."""
        return asdict(self.config)

    @classmethod
    def from_config(cls, config: NetworkConfig | Mapping[str, Any]) -> "PolicyNetwork":
        return cls(config)

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        """Build a torch.save payload; optimizer/data provenance may be metadata."""
        return {"network_version": self.checkpoint_version,
                "network_config": self.get_config(),
                "model_state_dict": self.state_dict(), "metadata": metadata}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *, strict: bool = True) -> "PolicyNetwork":
        if checkpoint.get("network_version", 1) != cls.checkpoint_version:
            raise ValueError("Unsupported Q4 network checkpoint version")
        model = cls.from_config(checkpoint["network_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return model

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> "PolicyNetwork":
        """Load a tensor-only checkpoint on the requested device, in eval mode."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls.from_checkpoint(checkpoint)
        # load_state_dict copies into this fresh model's CPU parameters.
        if isinstance(map_location, (str, torch.device)):
            model.to(map_location)
        return model.eval()

    def _validate(self, nodes: Tensor, node_mask: Tensor, candidates: Tensor,
                  candidate_mask: Tensor, global_features: Tensor) -> None:
        cfg = self.config
        if nodes.ndim != 3 or nodes.shape[-1] != cfg.node_dim:
            raise ValueError(f"nodes must have shape [B,N,{cfg.node_dim}]")
        if candidates.ndim != 3 or candidates.shape[-1] != cfg.candidate_dim:
            raise ValueError(f"candidates must have shape [B,K,{cfg.candidate_dim}]")
        if candidates.shape[0] != nodes.shape[0]:
            raise ValueError("nodes and candidates must have the same batch size")
        if global_features.shape != (nodes.shape[0], cfg.global_dim):
            raise ValueError(f"global_features must have shape [B,{cfg.global_dim}]")
        if node_mask.shape != nodes.shape[:2] or candidate_mask.shape != candidates.shape[:2]:
            raise ValueError("Masks must match the batch and token dimensions")
        if node_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
            raise TypeError("Masks must be bool with True for valid entries")
        if candidates.shape[1] == 0 or not bool(candidate_mask.any(dim=1).all()):
            raise ValueError("Every state must contain at least one valid candidate")

    def encode(self, nodes: Tensor, node_mask: Tensor, candidates: Tensor,
               candidate_mask: Tensor, global_features: Tensor) -> dict[str, Tensor]:
        """Expose the attention representation without evaluating any cost head."""
        self._validate(nodes, node_mask, candidates, candidate_mask, global_features)
        # Remove padding before projection: even NaN/Inf padding must not leak
        # through an attention matmul or its backward pass.
        clean_nodes = nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        clean_candidates = candidates.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        global_token = self.global_embedding(global_features).unsqueeze(1)
        memory = torch.cat((self.node_embedding(clean_nodes), global_token), dim=1)
        memory_mask = torch.cat((node_mask, torch.ones_like(node_mask[:, :1])), dim=1)
        if nodes.shape[1] == 0:
            memory_mask = torch.ones((nodes.shape[0], 1), dtype=torch.bool, device=nodes.device)
        for block in self.node_encoder:
            memory = block(memory, memory_mask)
        memory = self.memory_norm(memory)

        candidate_tokens = self.candidate_embedding(clean_candidates)
        candidate_update, _ = self.candidate_cross_attention(
            self.candidate_norm(candidate_tokens), memory, memory,
            key_padding_mask=~memory_mask, need_weights=False)
        candidate_tokens = candidate_tokens + self.dropout(candidate_update)
        candidate_tokens = candidate_tokens + self.dropout(
            self.candidate_ffn(self.candidate_ffn_norm(candidate_tokens)))
        candidate_tokens = candidate_tokens.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        if self.candidate_encoder is not None:
            candidate_tokens = self.candidate_encoder(candidate_tokens, candidate_mask)
        candidate_context = self.candidate_context_norm(candidate_tokens)

        global_update, _ = self.global_cross_attention(
            self.global_query_norm(global_token), memory, memory,
            key_padding_mask=~memory_mask, need_weights=False)
        global_context = self.global_context_norm(global_token + self.dropout(global_update)).squeeze(1)
        joint = torch.cat((candidate_context, global_context.unsqueeze(1).expand(
            -1, candidates.shape[1], -1)), dim=-1)
        return {"joint": joint, "candidate_context": candidate_context,
                "global_context": global_context}

    def policy_scores(self, encoded: dict[str, Tensor], candidate_mask: Tensor) -> Tensor:
        candidate_context, global_context = encoded["candidate_context"], encoded["global_context"]
        joint = encoded["joint"]
        pointer = (self.pointer_key(candidate_context) *
                   self.pointer_query(global_context).unsqueeze(1)).sum(dim=-1)
        pointer = pointer / (self.config.d_model ** 0.5)
        scores = self.config.logit_clip * torch.tanh(pointer) + self.rank_head(joint).squeeze(-1)
        return scores.masked_fill(~candidate_mask, -10000.0)

    def forward(self, nodes: Tensor, node_mask: Tensor, candidates: Tensor,
                candidate_mask: Tensor, global_features: Tensor) -> dict[str, Tensor]:
        encoded = self.encode(nodes, node_mask, candidates, candidate_mask, global_features)
        q = F.softplus(self.q_head(encoded["joint"]).squeeze(-1))
        value = F.softplus(self.value_head(encoded["global_context"]).squeeze(-1))
        return {"scores": self.policy_scores(encoded, candidate_mask),
                "q": q.masked_fill(~candidate_mask, 0.0), "value": value}
