# src/mimicvlm/utils/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
import csv
import pandas as pd
import torch 

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv(path: PathLike) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)

def write_csv(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)


def read_json(path: PathLike) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {e}") from e
    return out


def write_jsonl(records: Iterable[Dict[str, Any]], path: PathLike) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def append_row_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)
