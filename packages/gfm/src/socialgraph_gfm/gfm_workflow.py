"""Compatibility alias for the staged GFM workflow implementation."""

from __future__ import annotations

import sys

from . import workflows as _implementation

adapt_gfm = _implementation.adapt_gfm
check_gfm_task_assets = _implementation.check_gfm_task_assets
embed_gfm_text = _implementation.embed_gfm_text
evaluate_gfm = _implementation.evaluate_gfm
evaluate_gfm_checkpoint_test_once = _implementation.evaluate_gfm_checkpoint_test_once
export_gfm = _implementation.export_gfm
fetch_gfm_openalex = _implementation.fetch_gfm_openalex
fetch_gfm_thgl_software = _implementation.fetch_gfm_thgl_software
fetch_gfm_wikimedia_talk = _implementation.fetch_gfm_wikimedia_talk
prepare_gfm_corpus = _implementation.prepare_gfm_corpus
pretrain_gfm = _implementation.pretrain_gfm
resume_gfm = _implementation.resume_gfm
validate_gfm = _implementation.validate_gfm
verify_gfm_checkpoint_fresh = _implementation.verify_gfm_checkpoint_fresh
verify_gfm_product_checkpoint_fresh = _implementation.verify_gfm_product_checkpoint_fresh
verify_gfm_suite_checkpoint_fresh = _implementation.verify_gfm_suite_checkpoint_fresh
_adapt_worker = _implementation._adapt_worker
_lodo_worker = _implementation._lodo_worker
_pretrain_worker = _implementation._pretrain_worker

__all__ = list(_implementation.__all__)

sys.modules[__name__] = _implementation
