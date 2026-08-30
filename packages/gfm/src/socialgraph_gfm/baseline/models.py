"""Feature MLP and two-layer COO GraphSAGE baseline models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


class HadamardLinkPredictor(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return self.layers(source * target).view(-1)


class FeatureMLP(nn.Module):
    """Feature-only link model; it never receives the message graph."""

    def __init__(self, feature_channels: int, *, hidden_channels: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.predictor = HadamardLinkPredictor(
            feature_channels, hidden_channels=hidden_channels, dropout=dropout
        )

    def forward(self, x: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.predictor(x[edge_label_index[0]], x[edge_label_index[1]])


class GraphSAGEEncoder(nn.Module):
    """Two mean-aggregation layers using PyG COO tensors (no torch_sparse)."""

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_channels: int = 128,
        output_channels: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(feature_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, output_channels, aggr="mean")
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        hidden = self.conv1(x, edge_index).relu()
        hidden = nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index)

    def inference(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        device: str,
        batch_size: int,
    ) -> Tensor:
        """Exact layer-wise inference with CPU-resident full-node embeddings.

        Each layer samples all one-hop neighbors for a bounded seed batch.  It
        uses PyG COO/pyg_lib and never constructs a ``torch_sparse`` tensor.
        """

        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        try:
            import pyg_lib  # noqa: F401
        except ImportError as error:
            raise RuntimeError("GraphSAGE neighbor inference requires pyg_lib") from error
        if batch_size <= 0:
            raise ValueError("inference batch size must be positive")
        current = x.detach().cpu()
        cpu_edge_index = edge_index.detach().cpu()
        layers = (self.conv1, self.conv2)
        self.eval()
        for index, layer in enumerate(layers):
            data = Data(x=current, edge_index=cpu_edge_index)
            loader = NeighborLoader(
                data,
                input_nodes=None,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            output_channels = layer.out_channels
            output = torch.empty((current.shape[0], output_channels), dtype=current.dtype)
            with torch.no_grad():
                for batch in loader:
                    seed_count = int(batch.batch_size)
                    seed_ids = batch.n_id[:seed_count]
                    values = layer(batch.x.to(device), batch.edge_index.to(device))[:seed_count]
                    if index == 0:
                        values = values.relu()
                    output[seed_ids] = values.detach().cpu()
            current = output
        return current


class GraphSAGELinkModel(nn.Module):
    def __init__(
        self, feature_channels: int, *, hidden_channels: int = 128, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            feature_channels,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            dropout=dropout,
        )
        self.predictor = HadamardLinkPredictor(
            hidden_channels, hidden_channels=hidden_channels, dropout=dropout
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, embeddings: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.predictor(
            embeddings[edge_label_index[0]], embeddings[edge_label_index[1]]
        )

    def inference(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        device: str,
        batch_size: int,
    ) -> Tensor:
        return self.encoder.inference(
            x, edge_index, device=device, batch_size=batch_size
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.decode(self.encode(x, edge_index), edge_label_index)
