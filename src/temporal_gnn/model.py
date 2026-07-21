"""Temporal Graph Network components."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv


class TemporalGraphAttention(nn.Module):
    """Attend over recent neighbors using messages and relative-time encoding."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int,
                 time_encoder: nn.Module, use_time_encoding: bool = True) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("embedding_dim must be even for two attention heads")
        self.time_encoder = time_encoder
        self.use_time_encoding = use_time_encoding
        edge_dim = msg_dim + time_encoder.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, memory: Tensor, last_update: Tensor, edge_index: Tensor,
                edge_t: Tensor, edge_msg: Tensor) -> Tensor:
        relative_t = last_update[edge_index[0]] - edge_t
        time_features = self.time_encoder(relative_t.to(memory.dtype))
        if not self.use_time_encoding:
            time_features = torch.zeros_like(time_features)
        edge_features = torch.cat([time_features, edge_msg], dim=-1)
        return self.conv(memory, edge_index, edge_features)


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.src = nn.Linear(embedding_dim, embedding_dim)
        self.dst = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, source: Tensor, destination: Tensor) -> Tensor:
        return self.output(torch.relu(self.src(source) + self.dst(destination)))
