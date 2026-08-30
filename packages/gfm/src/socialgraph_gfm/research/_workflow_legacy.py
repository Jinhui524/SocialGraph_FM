"""Legacy module alias for :mod:`.workflow`.

The module object is deliberately shared so imports and monkeypatches through
either historical path observe the same compatibility surface.
"""

from __future__ import annotations

import sys

from . import workflow as _implementation

__all__ = list(_implementation.__all__)

sys.modules[__name__] = _implementation
