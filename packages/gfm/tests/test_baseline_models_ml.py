# ruff: noqa: E402

import copy
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from socialgraph_gfm.baseline.models import FeatureMLP, GraphSAGELinkModel
from socialgraph_gfm.baseline.protocols import build_protocol
from socialgraph_gfm.baseline.trainer import probe_cuda_batch_size, train_learning_run
from socialgraph_gfm.baseline.types import CorpusArrays, RunSpec

PYG_LIB_AVAILABLE = importlib.util.find_spec("pyg_lib") is not None


def _tiny_corpus() -> CorpusArrays:
    rng = np.random.default_rng(4)
    return CorpusArrays.from_mapping(
        {
            "x": rng.normal(size=(8, 4)).astype(np.float32),
            "edge_index": np.asarray(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
                ]
            ),
            "edge_timestamp": np.asarray(
                [2015, 2015, 2016, 2016, 2017, 2017, 2017, 2017, 2016, 2016, 2015, 2015]
            ),
            "variant_train_positive": np.asarray([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]),
            "variant_validation_positive": np.asarray([[0, 1], [3, 4]]),
            "variant_validation_negative": np.asarray([[0, 0, 1, 2], [4, 5, 6, 7]]),
            "variant_test_positive": np.asarray([[0, 1, 2], [4, 5, 7]]),
            "variant_test_negative": np.asarray([[0, 1, 2, 3], [6, 7, 5, 7]]),
        }
    )


@pytest.mark.parametrize("kind", ["mlp", "graphsage"])
def test_models_have_finite_forward_backward_and_update(kind):
    x = torch.randn(7, 4)
    message = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    labels = torch.tensor([[0, 1, 2, 5], [2, 3, 4, 6]], dtype=torch.long)
    model = FeatureMLP(4, hidden_channels=8) if kind == "mlp" else GraphSAGELinkModel(
        4, hidden_channels=8
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logits = model(x, labels) if kind == "mlp" else model(x, message, labels)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 1.0, 0.0, 0.0])
    )
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_cpu_batch_probe_selects_first_fixed_candidate():
    seen = []
    selected, peak = probe_cuda_batch_size(
        seen.append, device="cpu", candidates=(4096, 2048, 1024)
    )
    assert selected == 4096
    assert peak == 0.0
    assert seen == [4096]


@pytest.mark.skipif(not PYG_LIB_AVAILABLE, reason="layer-wise sampling requires pyg_lib")
def test_graphsage_layerwise_inference_is_finite_without_torch_sparse():
    model = GraphSAGELinkModel(4, hidden_channels=8, dropout=0.0)
    x = torch.randn(7, 4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    embeddings = model.inference(x, edge_index, device="cpu", batch_size=3)
    assert embeddings.shape == (7, 8)
    assert torch.isfinite(embeddings).all()


def test_dev_mlp_never_reads_test_and_formal_reads_it_once_after_selection():
    corpus = _tiny_corpus()
    protocol = build_protocol(corpus, "ogb_official")
    config = {
        "hiddenChannels": 8,
        "dropout": 0.0,
        "learningRate": 0.01,
        "devEpochs": 2,
        "devPositiveLimit": 4,
        "formalMaxEpochs": 2,
        "formalMinEpochs": 1,
        "evalEvery": 1,
        "patience": 2,
        "trainPositiveLimit": 4,
        "candidateBatchSizes": [8, 4, 2],
        "scoreBatchSize": 16,
    }

    calls = []

    def evaluator(positive, negative):
        calls.append((len(positive), len(negative)))
        return {"hits@10": 0.5, "hits@50": 0.5, "hits@100": 0.5}

    dev = train_learning_run(
        RunSpec("exp-dev", "dev", "dev", "ogb_official", "mlp", 7),
        corpus=corpus,
        protocol=protocol,
        config=config,
        device="cpu",
        evaluator=evaluator,
    )
    assert dev.test_metrics is None
    assert dev.test_read_after_selection is False
    assert all(positive_count != 3 for positive_count, _ in calls)

    calls.clear()
    formal = train_learning_run(
        RunSpec("exp-formal", "formal", "formal", "ogb_official", "mlp", 8),
        corpus=corpus,
        protocol=protocol,
        config=config,
        device="cpu",
        evaluator=evaluator,
    )
    assert formal.test_metrics == {"hits@10": 0.5, "hits@50": 0.5, "hits@100": 0.5}
    assert formal.test_read_after_selection is True
    assert [count for count, _ in calls].count(3) == 1


def test_resume_restores_current_best_optimizer_and_independent_sampler_rng():
    corpus = _tiny_corpus()
    protocol = build_protocol(corpus, "ogb_official")
    config = {
        "hiddenChannels": 8,
        "dropout": 0.0,
        "learningRate": 0.01,
        "formalMaxEpochs": 3,
        "formalMinEpochs": 1,
        "evalEvery": 1,
        "patience": 10,
        "trainPositiveLimit": 4,
        "candidateBatchSizes": [8, 4, 2],
        "scoreBatchSize": 16,
    }

    def evaluator(_positive, _negative):
        return {"hits@10": 0.5, "hits@50": 0.5, "hits@100": 0.5}

    class InjectedInterruption(RuntimeError):
        pass

    interrupted = {}

    def stop_after_first_latest(kind, payload):
        if kind == "latest":
            interrupted.update(copy.deepcopy(payload))
            raise InjectedInterruption

    spec = RunSpec("resume-exp", "resume-run", "formal", "ogb_official", "mlp", 42)
    with pytest.raises(InjectedInterruption):
        train_learning_run(
            spec,
            corpus=corpus,
            protocol=protocol,
            config=config,
            device="cpu",
            evaluator=evaluator,
            checkpoint_sink=stop_after_first_latest,
        )
    assert interrupted["epoch"] == 1
    assert interrupted["terminal"] is False
    assert interrupted["best_epoch"] == 1

    completed = {}

    def capture_latest(kind, payload):
        if kind == "latest":
            completed.clear()
            completed.update(copy.deepcopy(payload))

    result = train_learning_run(
        spec,
        corpus=corpus,
        protocol=protocol,
        config=config,
        device="cpu",
        evaluator=evaluator,
        checkpoint_sink=capture_latest,
        resume_state=interrupted,
    )
    assert [row["epoch"] for row in result.history] == [1.0, 2.0, 3.0]
    assert completed["epoch"] == 3
    assert completed["terminal"] is True
    assert completed["best_epoch"] == 1
    with pytest.raises(ValueError, match="terminal"):
        train_learning_run(
            spec,
            corpus=corpus,
            protocol=protocol,
            config=config,
            device="cpu",
            evaluator=evaluator,
            resume_state=completed,
        )
