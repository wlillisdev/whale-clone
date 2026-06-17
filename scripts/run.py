#!/usr/bin/env python3
"""Thin entry: data -> backtest -> verdict. See ``python -m whale_clone --help``."""

from __future__ import annotations

from whale_clone.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
