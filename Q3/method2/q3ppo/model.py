import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared candidate encoder + permutation-invariant global context.

    No channel-index embedding, so arbitrary channel labels do not get memorized.
    The current-channel indicator is explicitly part of observable features.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(20, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(hidden + 8, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1)

    def forward(self, candidates, global_features, mask):
        encoded = self.encoder(candidates)
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        context = self.context(torch.cat((pooled, global_features), dim=-1))
        joined = torch.cat((encoded, context[:, None].expand(-1, encoded.shape[1], -1)), dim=-1)
        logits = self.actor(joined).squeeze(-1).masked_fill(~mask, -1e9)
        return Categorical(logits=logits), self.critic(context).squeeze(-1)


def tensor_obs(obs, device):
    return (torch.as_tensor(obs["candidates"], device=device),
            torch.as_tensor(obs["global"], device=device),
            torch.as_tensor(obs["mask"], device=device))


def gae(rewards, values, dones, last_value, gamma=1., lam=.95):
    """dones[t] describes transition t. Rollout cutoffs bootstrap, resets do not."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards)-1 else values[t+1]
        live = 1. - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        carry = delta + gamma * lam * live * carry
        advantages[t] = carry
    return advantages, advantages + values
