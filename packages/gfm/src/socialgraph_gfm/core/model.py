"""Compact shared core GraphSAGE encoder and task heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv

from .adapters import HIDDEN_DIM


class ResidualGraphSAGE(nn.Module):
    def __init__(self, *, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SAGEConv(HIDDEN_DIM, HIDDEN_DIM) for _ in range(3)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(HIDDEN_DIM) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, edge_index: Tensor) -> Tensor:
        hidden = inputs
        for layer, norm in zip(self.layers, self.norms, strict=True):
            update = layer(hidden, edge_index)
            hidden = norm(hidden + self.dropout(torch.relu(update)))
        return hidden


class EdgeHead(nn.Module):
    """Order-sensitive edge head for directed relations."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, encoded: Tensor, pairs: Tensor) -> Tensor:
        endpoints = torch.cat((encoded[pairs[:, 0]], encoded[pairs[:, 1]]), dim=-1)
        return self.network(endpoints).squeeze(-1)


class SymmetricEdgeHead(nn.Module):
    """Endpoint-order invariant head for undirected relation completion."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, encoded: Tensor, pairs: Tensor) -> Tensor:
        left = encoded[pairs[:, 0]]
        right = encoded[pairs[:, 1]]
        invariant = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        return self.network(invariant).squeeze(-1)


class ResilienceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, encoded: Tensor) -> Tensor:
        return self.projection(encoded).squeeze(-1)


class CoreGFM(nn.Module):
    def __init__(self, *, node_classes: int) -> None:
        super().__init__()
        if node_classes < 1:
            raise ValueError("node_classes must be positive")
        self.encoder = ResidualGraphSAGE()
        self.node_head = nn.Linear(HIDDEN_DIM, node_classes)
        self.binary_link_head = EdgeHead()
        self.signed_edge_head = EdgeHead()
        self.resilience_head = ResilienceHead()
        self.field_decoder = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.decoder_remask_token = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.field_decoder_layer = SAGEConv(HIDDEN_DIM, HIDDEN_DIM)

    def encode(self, inputs: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(inputs, edge_index)

    def decode_fields(self, encoded: Tensor, edge_index: Tensor, field_mask: Tensor) -> Tensor:
        if field_mask.ndim != 2 or field_mask.shape[0] != encoded.shape[0]:
            raise ValueError("field mask must have shape [num_nodes, num_fields]")
        decoded: list[Tensor] = []
        for field_index in range(field_mask.shape[1]):
            selected = field_mask[:, field_index].unsqueeze(1)
            remasked = torch.where(selected, self.decoder_remask_token.unsqueeze(0), encoded)
            decoded.append(torch.relu(self.field_decoder_layer(remasked, edge_index)))
        return torch.stack(decoded, dim=1)


class ResearchCoreGFM(CoreGFM):
    """SocialGraph-FM Research shared encoder with domain routes and four explicit task heads.

    ``CoreGFM`` remains byte-for-byte compatible with the formal core
    checkpoint path.  This subclass is intentionally published through the
    separate research registry and never satisfies formal readiness gates.
    """

    CONTENT_POLICY_TASK = "research.content_policy_review"
    ACCOUNT_RISK_TASK = "research.account_risk_review"
    SIGNED_RELATION_TASK = "research.signed_relation_review"
    COLLABORATION_TASK = "core.collaboration_completion"

    def __init__(self, *, domains: tuple[str, ...]) -> None:
        if not domains or len(domains) != len(set(domains)) or tuple(sorted(domains)) != domains:
            raise ValueError("research domains must be a nonempty sorted unique tuple")
        super().__init__(node_classes=2)
        self.research_domains = domains
        self._domain_keys = {domain: f"domain_{index}" for index, domain in enumerate(domains)}
        self.domain_prompts = nn.ParameterDict(
            {
                self._domain_keys[domain]: nn.Parameter(torch.zeros(HIDDEN_DIM))
                for domain in domains
            }
        )
        self.target_adapters = nn.ModuleDict(
            {
                self._domain_keys[domain]: nn.Sequential(
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                )
                for domain in domains
            }
        )
        self.target_gates = nn.ParameterDict(
            {
                self._domain_keys[domain]: nn.Parameter(torch.zeros(()))
                for domain in domains
            }
        )
        self.content_policy_head = nn.Linear(HIDDEN_DIM, 2)
        self.account_risk_head = nn.Linear(HIDDEN_DIM, 2)
        self.collaboration_head = SymmetricEdgeHead()

    def encode_domain(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        domain: str | None,
        *,
        use_target_route: bool = True,
    ) -> Tensor:
        """Encode through the shared/null route or one explicit target route."""

        if domain is None:
            return super().encode(inputs, edge_index)
        try:
            key = self._domain_keys[domain]
        except KeyError as error:
            raise ValueError(f"unknown research domain: {domain}") from error
        shared = super().encode(inputs + self.domain_prompts[key].unsqueeze(0), edge_index)
        if not use_target_route:
            return shared
        gate = torch.sigmoid(self.target_gates[key])
        return shared + gate * self.target_adapters[key](shared)

    def edge_reconstruction_logits(
        self, encoded: Tensor, pairs: Tensor, *, directed: bool
    ) -> Tensor:
        if directed:
            return self.signed_edge_head(encoded, pairs)
        return self.collaboration_head(encoded, pairs)

    def task_logits(self, task_id: str, encoded: Tensor, locators: Tensor) -> Tensor:
        if task_id == self.CONTENT_POLICY_TASK:
            return self.content_policy_head(encoded[locators])
        if task_id == self.ACCOUNT_RISK_TASK:
            return self.account_risk_head(encoded[locators])
        if task_id == self.SIGNED_RELATION_TASK:
            return self.signed_edge_head(encoded, locators)
        if task_id == self.COLLABORATION_TASK:
            return self.collaboration_head(encoded, locators)
        raise ValueError(f"unsupported research task: {task_id}")


__all__ = [
    "EdgeHead",
    "ResearchCoreGFM",
    "ResidualGraphSAGE",
    "ResilienceHead",
    "CoreGFM",
    "SymmetricEdgeHead",
]
