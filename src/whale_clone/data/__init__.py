"""Data loaders: prices (Stooq/Yahoo) and 13F holdings (Dataroma/EDGAR).

Each loader has a network-backed real source and an offline ``demo`` source.
The ``demo`` source generates deterministic synthetic data so the full pipeline
and the validation gates can run with no network — useful in CI and in locked-
down build environments. Demo output is explicitly NOT a market claim.
"""

from __future__ import annotations
