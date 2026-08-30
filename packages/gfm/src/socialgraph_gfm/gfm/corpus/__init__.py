"""Safe acquisition and preparation adapters for the three GFM domain families."""

from .domains import check_all_gfm_corpora, load_domain, load_domain_view
from .openalex import (
    OpenAlexConfig,
    check_openalex_newcomers,
    fetch_historical_newcomers,
    fetch_openalex,
    load_openalex_newcomers,
    load_openalex_newcomers_view,
    newcomer_overlay_status,
    parse_topic_selector,
    prepare_openalex,
    verify_openalex_newcomers,
)
from .text_embeddings import (
    EmbeddingConfig,
    EmbeddingShard,
    VerifiedEmbeddingArtifact,
    build_bge_m3_embeddings,
    iter_embedding_shards,
    load_embedding_shard,
    lookup_embedding_rows,
    open_embedding_artifact,
    open_embedding_artifact_view,
    verify_embedding_artifact,
)
from .thgl_software import (
    fetch_thgl_software,
    load_thgl_software_splits,
    prepare_thgl_software,
)
from .wikimedia import fetch_wikimedia, prepare_wikimedia

__all__ = [
    "EmbeddingConfig",
    "EmbeddingShard",
    "OpenAlexConfig",
    "VerifiedEmbeddingArtifact",
    "build_bge_m3_embeddings",
    "check_all_gfm_corpora",
    "check_openalex_newcomers",
    "fetch_historical_newcomers",
    "fetch_openalex",
    "fetch_thgl_software",
    "fetch_wikimedia",
    "iter_embedding_shards",
    "load_domain",
    "load_domain_view",
    "load_embedding_shard",
    "load_openalex_newcomers",
    "load_openalex_newcomers_view",
    "load_thgl_software_splits",
    "lookup_embedding_rows",
    "newcomer_overlay_status",
    "open_embedding_artifact",
    "open_embedding_artifact_view",
    "parse_topic_selector",
    "prepare_openalex",
    "prepare_thgl_software",
    "prepare_wikimedia",
    "verify_embedding_artifact",
    "verify_openalex_newcomers",
]
