"""Formal, fail-closed corpus preparation for offline GFM experiments."""

from .ogbl_collab import (
    check_ogbl_collab_corpus,
    corpus_manifest_path,
    fetch_ogbl_collab,
    load_ogbl_collab_arrays,
    prepare_ogbl_collab_corpus,
)

__all__ = [
    "check_ogbl_collab_corpus",
    "corpus_manifest_path",
    "fetch_ogbl_collab",
    "load_ogbl_collab_arrays",
    "prepare_ogbl_collab_corpus",
]
