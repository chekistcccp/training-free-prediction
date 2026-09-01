from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def dump_json(path: str | os.PathLike[str], payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def parse_grid_cells(text: str, n: int) -> list[str]:
    import re
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
    pattern = rf"\b([{letters}])([1-{n}])\b"
    out: list[str] = []
    for a, b in re.findall(pattern, text.upper()):
        c = f"{a}{b}"
        if c not in out:
            out.append(c)
    return out


def rc_to_cell(r: int, c: int) -> str:
    return f"{chr(ord('A') + r)}{c + 1}"


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def grid_box(width: int, height: int, grid: int, row: int, col: int, parent: Box | None = None) -> Box:
    if parent is None:
        parent = Box(0, 0, width, height)
    xs = np.linspace(parent.x0, parent.x1, grid + 1).round().astype(int)
    ys = np.linspace(parent.y0, parent.y1, grid + 1).round().astype(int)
    return Box(int(xs[col]), int(ys[row]), int(xs[col + 1]), int(ys[row + 1]))
