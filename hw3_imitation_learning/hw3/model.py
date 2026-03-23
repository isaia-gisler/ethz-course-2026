"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(self, state_dim=10, action_dim=4, chunk_size=8, d_model=480, depth=4):
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model=d_model
        self.depth=depth
        
        layers = [nn.Linear(self.state_dim, d_model)]

        for _ in range(depth):
            layers.extend([nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model,d_model)])
        layers.append(nn.Linear(d_model, action_dim*chunk_size))

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        
        output = self.model(state)
        return output.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        
        pred_chunk = self(state)
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        
        return self(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(self, state_dim=19, action_dim=4, chunk_size=8, d_model=512, depth=5):
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model=d_model
        self.depth=depth
        
        layers = [nn.Linear(self.state_dim, d_model)]

        for _ in range(depth):
            layers.extend([nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model,d_model)])
        layers.append(nn.Linear(d_model, action_dim*chunk_size))

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        
        output = self.model(state)
        return output.reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        
        pred_chunk = self(state)
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        
        return self(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    **kwargs,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            **kwargs,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size
        )
    raise ValueError(f"Unknown policy type: {policy_type}")