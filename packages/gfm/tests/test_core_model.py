from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from socialgraph_gfm.core.adapters import (
    AdapterSchema,
    BundleInputAdapter,
    CategoricalAdapter,
    NumericAdapter,
    SparseMultiHotAdapter,
    StructureViewAdapter,
    fit_adapter_schema,
)
from socialgraph_gfm.core.bundle import calculate_graph_version_hash, CoreGraphBundle
from socialgraph_gfm.core.model import CoreGFM


def test_schema_adapters_emit_128_rows_and_multihot_stays_sparse() -> None:
    numeric = NumericAdapter()
    categorical = CategoricalAdapter(cardinality=7)
    multihot = SparseMultiHotAdapter(bucket_count=32)
    structure = StructureViewAdapter(input_width=3)

    assert numeric(torch.tensor([[1.0], [2.0]])).shape == (2, 128)
    assert categorical(torch.tensor([1, 6])).shape == (2, 128)
    indices = torch.tensor([1, 999_999, 3], dtype=torch.long)
    offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    output = multihot(indices=indices, offsets=offsets)
    assert output.shape == (2, 128)
    assert multihot.embedding.num_embeddings == 32
    assert structure(torch.ones(2, 3)).shape == (2, 128)
    exact = CategoricalAdapter(cardinality=300)
    assert exact.embedding.num_embeddings == 300
    with torch.no_grad():
        exact.embedding.weight.zero_()
        exact.embedding.weight[1].fill_(1.0)
        exact.embedding.weight[257].fill_(2.0)
    assert not torch.equal(exact(torch.tensor([1])), exact(torch.tensor([257])))
    with pytest.raises(ValueError, match="bounded categorical capacity"):
        CategoricalAdapter(cardinality=4_097)


def test_core_gfm_has_shared_three_layer_residual_encoder_and_four_heads() -> None:
    model = CoreGFM(node_classes=3)
    model.eval()
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    encoded = model.encode(torch.randn(4, 128), edge_index)

    assert encoded.shape == (4, 128)
    assert len(model.encoder.layers) == 3
    assert model.node_head(encoded).shape == (4, 3)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    assert model.binary_link_head(encoded, pairs).shape == (2,)
    assert model.signed_edge_head(encoded, pairs).shape == (2,)
    assert model.resilience_head(encoded).shape == (4,)
    assert (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        < 5_000_000
    )


def test_bundle_adapter_maps_every_schema_field_without_dense_expansion() -> None:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}, {"id": "b", "index": 1}],
        "edges": [{"sourceId": "a", "targetId": "b", "edgeType": "knows"}],
        "nodeFeatures": [
            {"kind": "numeric", "name": "score", "values": [1.0, 2.0]},
            {"kind": "categorical", "name": "role", "values": ["admin", None]},
            {
                "kind": "multiHot",
                "name": "sparse-tags",
                "rowOffsets": [0, 2, 3],
                "values": ["1", "999999", "4"],
            },
        ],
        "structuralFeatures": {
            "names": ["degree", "pagerank"],
            "values": [[1.0, 0.5], [1.0, 0.5]],
        },
        "source": {"sourceName": "fixture", "sourceSha256": "f" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    bundle = CoreGraphBundle.model_validate(payload)
    adapter = BundleInputAdapter(bundle, multi_hot_buckets=64, mode="training")

    assert adapter().shape == (2, 128)
    assert adapter.field_names == ("score", "role", "sparse-tags", "structure-view")
    assert max(buffer.numel() for _, buffer in adapter.named_buffers()) < 1_000_000
    model = CoreGFM(node_classes=2)
    total = sum(
        parameter.numel()
        for module in (adapter, model)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    assert total < 5_000_000


def _portable_bundle(
    *,
    nodes: list[str],
    numeric: list[float],
    categories: list[str | None],
    train_nodes: tuple[str, ...] | None = None,
) -> CoreGraphBundle:
    selected_train = set(nodes if train_nodes is None else train_nodes)
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": identifier, "index": index} for index, identifier in enumerate(nodes)],
        "edges": [
            {"sourceId": nodes[index], "targetId": nodes[index + 1], "edgeType": "knows"}
            for index in range(len(nodes) - 1)
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "score", "values": numeric},
            {"kind": "categorical", "name": "role", "values": categories},
            {
                "kind": "multiHot",
                "name": "tags",
                "rowOffsets": list(range(len(nodes) + 1)),
                "values": [f"tag-{identifier}" for identifier in nodes],
            },
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "portable", "sourceSha256": "8" * 64},
        "splitManifest": {
            "strategy": "official",
            "assignments": [
                {
                    "entityId": identifier,
                    "role": "train" if identifier in selected_train else "validation",
                }
                for identifier in nodes
            ],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def test_large_train_only_categorical_vocabulary_is_exact_and_unseen_remains_oov() -> None:
    train_categories = [f"category-{index:04d}" for index in range(300)]
    source = _portable_bundle(
        nodes=[f"train-{index:04d}" for index in range(300)] + ["validation"],
        numeric=[float(index) for index in range(301)],
        categories=train_categories + ["validation-only"],
        train_nodes=tuple(f"train-{index:04d}" for index in range(300)),
    )

    schema = fit_adapter_schema(source, multi_hot_buckets=512)
    categorical = schema.fields[1]
    assert categorical.kind == "categorical"
    assert len(categorical.vocabulary) == 300
    assert all("validation-only" not in entry.token for entry in categorical.vocabulary)

    target = _portable_bundle(
        nodes=["known", "unseen"],
        numeric=[0.0, 1.0],
        categories=["category-0000", "cross-graph-unseen"],
    )
    adapter = BundleInputAdapter(target, schema=schema, mode="inference")
    encoded_categories = getattr(adapter, "_field_1_values")
    assert encoded_categories[0].item() != 0
    assert encoded_categories[1].item() == 0
    categorical_adapter = adapter.adapters["field_1"]
    assert isinstance(categorical_adapter, CategoricalAdapter)
    assert categorical_adapter.embedding.num_embeddings == 301

    tampered = json.loads(schema.model_dump_json(by_alias=True))
    tampered["fields"][1]["vocabulary"][-1]["token"] = '{"type":"string","value":"category-9999"}'
    with pytest.raises(ValueError, match="adapterSchemaHash"):
        AdapterSchema.model_validate_json(json.dumps(tampered))


def test_portable_schema_fits_explicit_train_rows_and_target_rows_never_enter_state() -> None:
    source = _portable_bundle(
        nodes=["a", "b", "c"],
        numeric=[10.0, 1_000.0, -1_000.0],
        categories=["1", "held-out", None],
        train_nodes=("a",),
    )
    schema = fit_adapter_schema(source, train_row_ids=("a",), multi_hot_buckets=32)

    assert isinstance(schema, AdapterSchema)
    assert schema.fit_row_count == 1
    assert schema.fields[0].model_dump(mode="json", by_alias=True) == {
        "kind": "numeric",
        "name": "score",
        "mean": 10.0,
        "scale": 1.0,
    }
    categorical = schema.fields[1]
    assert categorical.model_dump(mode="json", by_alias=True)["vocabulary"] == [
        {"index": 1, "token": '{"type":"string","value":"1"}'}
    ]

    target = _portable_bundle(
        nodes=["w", "x", "y", "z"],
        numeric=[12.0, 20.0, -5.0, 10.0],
        categories=["1", "unseen", None, "held-out"],
    )
    adapter = BundleInputAdapter(target, schema=schema, mode="inference")
    assert adapter.num_nodes == 4
    assert getattr(adapter, "_field_0_values").flatten().tolist() == [2.0, 10.0, -15.0, 0.0]
    assert getattr(adapter, "_field_1_values").tolist() == [1, 0, 0, 0]
    assert all(not key.startswith("_field_") for key in adapter.state_dict())


def test_loading_learned_adapter_state_preserves_different_target_rows() -> None:
    source = _portable_bundle(nodes=["a", "b"], numeric=[2.0, 4.0], categories=["known", "known"])
    schema = fit_adapter_schema(source, train_row_ids=("a", "b"), multi_hot_buckets=16)
    learned = BundleInputAdapter(source, schema=schema, mode="training").state_dict()
    target = _portable_bundle(
        nodes=["a", "b"], numeric=[20.0, 40.0], categories=["unknown", "known"]
    )
    restored = BundleInputAdapter(target, schema=schema, mode="inference")
    before_numeric = getattr(restored, "_field_0_values").clone()
    before_categories = getattr(restored, "_field_1_values").clone()

    restored.load_state_dict(learned)

    assert torch.equal(getattr(restored, "_field_0_values"), before_numeric)
    assert torch.equal(getattr(restored, "_field_1_values"), before_categories)
    assert before_categories.tolist() == [0, 1]


def test_fresh_process_keeps_same_size_target_values_and_oov_categories(
    tmp_path: Path,
) -> None:
    source = _portable_bundle(nodes=["a", "b"], numeric=[2.0, 4.0], categories=["known", "known"])
    schema = fit_adapter_schema(source, train_row_ids=("a", "b"), multi_hot_buckets=16)
    target = _portable_bundle(
        nodes=["a", "b"], numeric=[20.0, 40.0], categories=["unknown", "known"]
    )
    schema_path = tmp_path / "schema.json"
    bundle_path = tmp_path / "target.json"
    state_path = tmp_path / "learned.pt"
    schema_path.write_text(schema.model_dump_json(by_alias=True), encoding="utf-8")
    bundle_path.write_text(target.model_dump_json(by_alias=True), encoding="utf-8")
    torch.save(BundleInputAdapter(source, schema=schema, mode="training").state_dict(), state_path)
    code = """
import json,sys,torch
from pathlib import Path
from socialgraph_gfm.core.adapters import AdapterSchema,BundleInputAdapter
from socialgraph_gfm.core.bundle import load_core_graph_bundle_json
schema=AdapterSchema.model_validate_json(Path(sys.argv[1]).read_bytes())
bundle=load_core_graph_bundle_json(Path(sys.argv[2]).read_bytes())
adapter=BundleInputAdapter(bundle,schema=schema,mode="inference")
adapter.load_state_dict(torch.load(sys.argv[3],weights_only=True,map_location='cpu'))
print(json.dumps({'numeric':adapter._field_0_values.flatten().tolist(),'categories':adapter._field_1_values.tolist()}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(schema_path), str(bundle_path), str(state_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert json.loads(completed.stdout.strip()) == {
        "numeric": [17.0, 37.0],
        "categories": [0, 1],
    }


def test_structure_rows_are_recomputed_from_explicit_train_visible_topology() -> None:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
            {"id": "c", "index": 2},
        ],
        "edges": [
            {"sourceId": "a", "targetId": "b", "edgeType": "knows"},
            {"sourceId": "b", "targetId": "c", "edgeType": "knows"},
        ],
        "nodeFeatures": [],
        "structuralFeatures": {
            "names": ["degree"],
            "values": [[999.0], [999.0], [999.0]],
        },
        "source": {"sourceName": "leak-trap", "sourceSha256": "9" * 64},
        "splitManifest": {
            "strategy": "official",
            "assignments": [
                {"entityId": "a", "role": "train"},
                {"entityId": "b", "role": "train"},
                {"entityId": "c", "role": "validation"},
            ],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    bundle = CoreGraphBundle.model_validate(payload)
    schema = fit_adapter_schema(
        bundle,
        train_row_ids=("a", "b"),
        visible_edge_indices=(0,),
        multi_hot_buckets=16,
    )
    adapter = BundleInputAdapter(bundle, schema=schema, mode="training", visible_edge_indices=(0,))

    assert schema.fields[0].model_dump(mode="json", by_alias=True) == {
        "kind": "structure",
        "names": ["degree"],
        "means": [1.0],
        "scales": [1.0],
        "algorithmVersion": "socialgraph-fm.core-visible-topology-structure/1.0",
    }
    assert getattr(adapter, "_field_0_values").flatten().tolist() == [0.0, 0.0, -1.0]


def test_decoder_remask_removes_selected_node_latent_before_field_decoding() -> None:
    model = CoreGFM(node_classes=2)
    model.eval()
    encoded = torch.randn(3, 128)
    changed = encoded.clone()
    changed[1] = 10_000.0
    edge_index = torch.empty((2, 0), dtype=torch.long)
    field_mask = torch.tensor([[False, False], [True, False], [False, False]])

    first = model.decode_fields(encoded, edge_index, field_mask)
    second = model.decode_fields(changed, edge_index, field_mask)
    assert torch.equal(first[1, 0], second[1, 0])
    assert not torch.equal(first[1, 1], second[1, 1])


def test_multihot_negative_buckets_are_unique_per_row_complements() -> None:
    adapter = SparseMultiHotAdapter(bucket_count=4)
    positive_rows = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    positive_ids = torch.tensor([0, 0, 2, 0, 1, 2, 3, 3])
    negative_rows, negative_ids = adapter.sample_negative_buckets(
        positive_rows=positive_rows,
        positive_ids=positive_ids,
        selected_rows=torch.tensor([0, 1]),
        budget_per_row=3,
        generator=torch.Generator().manual_seed(19),
    )

    row_zero_ids = negative_ids[negative_rows == 0]
    assert set(row_zero_ids.tolist()) == {1, 3}
    assert row_zero_ids.unique().numel() == row_zero_ids.numel()
    assert not torch.any(negative_rows == 1)
    positive_keys = positive_rows * 4 + positive_ids
    negative_keys = negative_rows * 4 + negative_ids
    assert not torch.isin(negative_keys, positive_keys).any()
    latent = torch.randn(2, 128, requires_grad=True)
    loss = adapter.reconstruction_loss(
        latent=latent,
        positive_rows=positive_rows,
        positive_ids=positive_ids,
        selected_rows=torch.tensor([0, 1]),
        budget_per_row=3,
        generator=torch.Generator().manual_seed(19),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
