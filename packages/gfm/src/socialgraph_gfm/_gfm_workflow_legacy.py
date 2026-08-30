"""Deprecated compatibility alias for the staged GFM workflows.

The former monolithic implementation was mechanically extracted into
``socialgraph_gfm.workflows``.  This import path remains for integrations that
used it during the repository transition.
"""

from __future__ import annotations

import sys

from . import workflows as _implementation

sys.modules[__name__] = _implementation
