"""On-disk cache for holdings and prices.

Parquet is the system of record (brief, section 6) — never loose CSVs. The
cache is keyed by a stable name so a second run is offline and reproducible.
CSV is offered only as a human-readable export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class Store:
    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def has(self, name: str) -> bool:
        return self._path(name).exists()

    def save(self, name: str, df: pd.DataFrame) -> Path:
        path = self._path(name)
        df.to_parquet(path)
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"no cached frame named {name!r} at {path}")
        return pd.read_parquet(path)

    # --- JSON cache (e.g. CUSIP -> ticker map) ------------------------------
    def _json_path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def load_json(self, name: str, default: Any = None) -> Any:
        path = self._json_path(name)
        if not path.exists():
            return default
        return json.loads(path.read_text())

    def save_json(self, name: str, obj: Any) -> Path:
        path = self._json_path(name)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True))
        return path

    def export_csv(self, df: pd.DataFrame, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=True)
        return out
