"""Training and causal evaluation for TGN temporal link prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from .config import ExperimentConfig
from .metrics import (
    binary_metrics,
    chronological_performance,
    inactivity_gap_performance,
)
from .model import (
    GatedTemporalGraphAttention,
    LearnedMessage,
    LinkPredictor,
    TemporalGraphAttention,
)


@dataclass
class Modules:
    memory: TGNMemory
    embedding: TemporalGraphAttention
    predictor: LinkPredictor
    neighbors: LastNeighborLoader
    optimizer: torch.optim.Optimizer


class TGNExperiment:
    def __init__(self, config: ExperimentConfig, data: Any, device: torch.device):
        self.config = config
        self.data = data.to(device)
        self.device = device
        if config.model_variant == "baseline":
            message_module = IdentityMessage(
                data.msg.size(-1), config.memory_dim, config.time_dim
            )
            embedding_type = TemporalGraphAttention
        else:
            message_module = LearnedMessage(
                data.msg.size(-1), config.memory_dim, config.time_dim
            )
            embedding_type = GatedTemporalGraphAttention

        memory = TGNMemory(
            num_nodes=data.num_nodes,
            raw_msg_dim=data.msg.size(-1),
            memory_dim=config.memory_dim,
            time_dim=config.time_dim,
            message_module=message_module,
            aggregator_module=LastAggregator(),
        ).to(device)
        embedding = embedding_type(
            config.memory_dim,
            config.embedding_dim,
            data.msg.size(-1),
            memory.time_enc,
        ).to(device)
        predictor = LinkPredictor(config.embedding_dim).to(device)
        parameters = (
            set(memory.parameters())
            | set(embedding.parameters())
            | set(predictor.parameters())
        )
        self.modules = Modules(
            memory=memory,
            embedding=embedding,
            predictor=predictor,
            neighbors=LastNeighborLoader(
                data.num_nodes, size=config.neighbor_size, device=device
            ),
            optimizer=torch.optim.Adam(parameters, lr=config.learning_rate),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    def loader(self, split: Any, shuffle: bool = False) -> TemporalDataLoader:
        if shuffle:
            raise ValueError("temporal event loaders must never be shuffled")
        return TemporalDataLoader(
            split,
            batch_size=self.config.batch_size,
            neg_sampling_ratio=1.0,
        )

    def reset_state(self) -> None:
        self.modules.memory.reset_state()
        self.modules.neighbors.reset_state()

    def trainable_parameter_count(self) -> int:
        unique = {
            id(parameter): parameter
            for module in (
                self.modules.memory,
                self.modules.embedding,
                self.modules.predictor,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        return sum(parameter.numel() for parameter in unique.values())

    def _embed(self, batch: Any):
        n_id, edge_index, e_id = self.modules.neighbors(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        memory, last_update = self.modules.memory(n_id)
        embeddings = self.modules.embedding(
            memory,
            last_update,
            edge_index,
            self.data.t[e_id],
            self.data.msg[e_id],
        )
        return embeddings, last_update

    def _observe(self, batch: Any) -> None:
        self.modules.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        self.modules.neighbors.insert(batch.src, batch.dst)

    def train_epoch(self, train_data: Any) -> float:
        self.modules.memory.train()
        self.modules.embedding.train()
        self.modules.predictor.train()
        self.reset_state()
        total_loss = 0.0
        for batch in self.loader(train_data):
            batch = batch.to(self.device)
            self.modules.optimizer.zero_grad()
            embeddings, _ = self._embed(batch)
            positive = self.modules.predictor(
                embeddings[self.assoc[batch.src]], embeddings[self.assoc[batch.dst]]
            )
            negative = self.modules.predictor(
                embeddings[self.assoc[batch.src]],
                embeddings[self.assoc[batch.neg_dst]],
            )
            loss = self.loss_fn(positive, torch.ones_like(positive))
            loss += self.loss_fn(negative, torch.zeros_like(negative))
            self._observe(batch)
            loss.backward()
            self.modules.optimizer.step()
            self.modules.memory.detach()
            total_loss += float(loss) * batch.num_events
        return total_loss / train_data.num_events

    @torch.no_grad()
    def replay(self, split: Any) -> None:
        self.modules.memory.eval()
        for batch in self.loader(split):
            self._observe(batch.to(self.device))

    @torch.no_grad()
    def evaluate(self, split: Any, seed: int) -> dict[str, object]:
        self.modules.memory.eval()
        self.modules.embedding.eval()
        self.modules.predictor.eval()
        torch.manual_seed(seed)
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        inactivity_gaps: list[float] = []
        for batch in self.loader(split):
            batch = batch.to(self.device)
            embeddings, last_update = self._embed(batch)
            source_update = last_update[self.assoc[batch.src]]
            destination_update = last_update[self.assoc[batch.dst]]
            last_endpoint_activity = torch.maximum(
                source_update, destination_update
            )
            inactivity_gaps.extend(
                (batch.t - last_endpoint_activity).clamp_min(0).cpu().tolist()
            )
            positive = self.modules.predictor(
                embeddings[self.assoc[batch.src]], embeddings[self.assoc[batch.dst]]
            ).sigmoid().view(-1)
            negative = self.modules.predictor(
                embeddings[self.assoc[batch.src]],
                embeddings[self.assoc[batch.neg_dst]],
            ).sigmoid().view(-1)
            positive_scores.extend(positive.cpu().tolist())
            negative_scores.extend(negative.cpu().tolist())
            self._observe(batch)
        overall = binary_metrics(
            [1] * len(positive_scores) + [0] * len(negative_scores),
            positive_scores + negative_scores,
        )
        return {
            "overall": overall,
            "chronological_bins": chronological_performance(
                positive_scores, negative_scores
            ),
            "inactivity_gap_bins": inactivity_gap_performance(
                positive_scores, negative_scores, inactivity_gaps
            ),
        }

    def snapshot_parameters(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "memory": copy.deepcopy(self.modules.memory.state_dict()),
            "embedding": copy.deepcopy(self.modules.embedding.state_dict()),
            "predictor": copy.deepcopy(self.modules.predictor.state_dict()),
        }

    def load_parameters(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.modules.memory.load_state_dict(state["memory"])
        self.modules.embedding.load_state_dict(state["embedding"])
        self.modules.predictor.load_state_dict(state["predictor"])
