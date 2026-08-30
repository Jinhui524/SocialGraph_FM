"""Fresh-process fixed-output verification for baseline checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baseline.protocols import build_protocol
from .baseline.types import CorpusArrays
from .canonical import canonical_sha256
from .checkpoint import load_baseline_checkpoint, read_baseline_manifest
from .corpus import check_ogbl_collab_corpus, load_ogbl_collab_arrays
from .runtime import require_ml_runtime, set_seed
from .tensor_digest import canonical_tensor_digest


def verification_digest(
    manifest_path: str | Path,
    *,
    root: str | Path,
    device: str,
) -> dict[str, Any]:
    torch, _ = require_ml_runtime(device)
    manifest = read_baseline_manifest(manifest_path)
    payload = load_baseline_checkpoint(manifest, map_location="cpu")
    if manifest.model != "graphsage":
        raise ValueError("fresh-process verification is defined for GraphSAGE checkpoints")
    corpus_manifest = check_ogbl_collab_corpus(root)
    if corpus_manifest["logicalHash"] != manifest.corpus_hash:
        raise ValueError("checkpoint corpus hash differs from the checked formal corpus")
    corpus = CorpusArrays.from_mapping(
        load_ogbl_collab_arrays(root),
        corpus_hash=str(corpus_manifest["logicalHash"]),
        expected_num_nodes=235_868,
        expected_feature_dim=128,
    )
    config = payload["config"]
    hidden = int(config["hiddenChannels"])
    dropout = float(config["dropout"])
    inference_batch_size = int(config["inferenceBatchSize"])
    from .baseline.models import GraphSAGELinkModel

    set_seed(20260820, device)
    model = GraphSAGELinkModel(
        corpus.node_features.shape[1], hidden_channels=hidden, dropout=dropout
    ).to(device)
    model.encoder.load_state_dict(payload["best_model_state"])
    model.predictor.load_state_dict(payload["best_predictor_state"])
    model.eval()
    stage = build_protocol(corpus, manifest.track).validation
    x = torch.as_tensor(corpus.node_features, dtype=torch.float32)
    message = torch.as_tensor(stage.message_edges, dtype=torch.long).t().contiguous()
    embeddings = model.inference(
        x,
        message,
        device=device,
        batch_size=inference_batch_size,
    )
    fixed_positive = stage.positive_edges[:64]
    if stage.negative_edges is None:
        raise ValueError("verification stage has no fixed negatives")
    fixed_negative = stage.negative_edges[:64]
    fixed_pairs = torch.as_tensor(
        __import__("numpy").concatenate((fixed_positive, fixed_negative), axis=0),
        dtype=torch.long,
    ).t()
    scores = []
    with torch.no_grad():
        for start in range(0, fixed_pairs.shape[1], 64):
            selected = fixed_pairs[:, start : start + 64]
            source = embeddings[selected[0]].to(device)
            target = embeddings[selected[1]].to(device)
            scores.append(model.predictor(source, target).detach().cpu())
    score_tensor = torch.cat(scores).to(dtype=torch.float32)
    digest = canonical_sha256(
        {
            "schemaVersion": "gfm.baseline-verification/1.0",
            "runId": manifest.run_id,
            "corpusHash": manifest.corpus_hash,
            "track": manifest.track,
            "model": manifest.model,
            "positiveCount": 64,
            "negativeCount": 64,
            "scores": canonical_tensor_digest(score_tensor),
        }
    )
    return {
        "ok": True,
        "schemaVersion": "gfm.baseline-verification/1.0",
        "runId": manifest.run_id,
        "verificationDigest": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                verification_digest(
                    args.manifest, root=args.root, device=args.device
                ),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {"code": "GFM_BASELINE_VERIFICATION_FAILED", "message": str(error)},
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
