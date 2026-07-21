"""Temporal Graph Network components."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv


class LearnedMessage(nn.Module):
    """Compress raw interactions and temporal context before memory update."""

    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int) -> None:
        super().__init__()
        input_dim = raw_msg_dim + 2 * memory_dim + time_dim
        self.out_channels = memory_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, source: Tensor, destination: Tensor, raw_message: Tensor,
                time_encoding: Tensor) -> Tensor:
        features = torch.cat(
            [source, destination, raw_message, time_encoding], dim=-1
        )
        return self.network(features)


class TemporalGraphAttention(nn.Module):
    """Attend over recent neighbors using messages and relative-time encoding."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int,
                 time_encoder: nn.Module) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("embedding_dim must be even for two attention heads")
        self.time_encoder = time_encoder
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
        edge_features = torch.cat([time_features, edge_msg], dim=-1)
        return self.conv(memory, edge_index, edge_features)


class GatedTemporalGraphAttention(TemporalGraphAttention):
    """Adaptively fuse long-term memory with the recent temporal neighborhood."""

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int,
                 time_encoder: nn.Module) -> None:
        super().__init__(in_channels, out_channels, msg_dim, time_encoder)
        self.memory_projection = nn.Linear(in_channels, out_channels)
        self.gate = nn.Linear(2 * out_channels, out_channels)
        self.normalization = nn.LayerNorm(out_channels)

    def forward(self, memory: Tensor, last_update: Tensor, edge_index: Tensor,
                edge_t: Tensor, edge_msg: Tensor) -> Tensor:
        neighborhood = super().forward(
            memory, last_update, edge_index, edge_t, edge_msg
        )
        long_term = self.memory_projection(memory)
        gate = torch.sigmoid(self.gate(torch.cat([long_term, neighborhood], dim=-1)))
        fused = gate * neighborhood + (1.0 - gate) * long_term
        return self.normalization(fused)


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.src = nn.Linear(embedding_dim, embedding_dim)
        self.dst = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, source: Tensor, destination: Tensor) -> Tensor:
        return self.output(torch.relu(self.src(source) + self.dst(destination)))
