import math

from socialgraph_gfm.core.bundle import (
    CategoricalFeature,
    MultiHotFeature,
    NumericFeature,
)
from socialgraph_gfm.core.transforms import (
    apply_numeric_normalization,
    build_training_structure_view,
    encode_sparse_multi_hot,
    fit_train_only_transforms,
)


def _features():
    return (
        NumericFeature(kind="numeric", name="activity", values=(1.0, 3.0, 100.0)),
        CategoricalFeature(
            kind="categorical", name="team", values=("red", "blue", "test-only")
        ),
        MultiHotFeature(
            kind="multiHot",
            name="skills",
            rowOffsets=(0, 2, 3, 5),
            values=("a", "b", "b", "test-only", "c"),
        ),
    )


def test_transform_fitting_uses_only_declared_training_rows():
    metadata = fit_train_only_transforms(_features(), train_node_indices=(0, 1))

    numeric = metadata.numeric[0]
    assert (numeric.name, numeric.mean, numeric.scale) == ("activity", 2.0, 1.0)
    assert apply_numeric_normalization((1.0, 3.0, 100.0), numeric) == (-1.0, 1.0, 98.0)

    categorical = metadata.categorical[0]
    assert categorical.vocabulary == ("blue", "red")
    assert "test-only" not in categorical.vocabulary

    multi_hot = metadata.multi_hot[0]
    assert multi_hot.vocabulary == ("a", "b")
    assert "test-only" not in multi_hot.vocabulary
    assert metadata.fitted_role == "train"
    assert not hasattr(metadata, "target_labels")


def test_constant_numeric_feature_gets_finite_unit_scale():
    feature = NumericFeature(kind="numeric", name="constant", values=(4.0, 4.0, 99.0))
    metadata = fit_train_only_transforms((feature,), train_node_indices=(0, 1))
    assert metadata.numeric[0].scale == 1.0
    assert all(
        math.isfinite(value)
        for value in apply_numeric_normalization(feature.values, metadata.numeric[0])
    )


def test_multi_hot_encoding_remains_sparse_and_maps_unseen_values_to_unknown():
    feature = _features()[2]
    metadata = fit_train_only_transforms((feature,), train_node_indices=(0, 1)).multi_hot[0]
    encoded = encode_sparse_multi_hot(feature, metadata)
    assert encoded.row_offsets == (0, 2, 3, 5)
    assert encoded.column_indices == (1, 2, 2, 0, 0)
    assert len(encoded.column_indices) == len(feature.values)
    assert not hasattr(encoded, "dense")


def test_structure_view_is_hash_bound_to_training_topology_only():
    first = build_training_structure_view(
        graph_version_hash="a" * 64,
        num_nodes=4,
        edges=((1, 0), (2, 1), (0, 1)),
        directed=False,
    )
    reordered = build_training_structure_view(
        graph_version_hash="a" * 64,
        num_nodes=4,
        edges=((0, 1), (1, 2)),
        directed=False,
    )
    assert first.edges == ((0, 1), (1, 2))
    assert first.topology_hash == reordered.topology_hash
    assert first.role == "train"
    assert set(first.node_indices) == {0, 1, 2, 3}
