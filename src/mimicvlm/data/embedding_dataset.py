from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

@dataclass
class ShardIndex:
    shard_path: Path
    start: int
    end: int  # exclusive

class EmbeddingShardDataset(Dataset):
    """
    Loads all embeddings into RAM once.
    Expects shard_*.pt files, but works best with 1 shard.

    If shards contain dicom_ids/study_ids/subject_ids (new format),
    those are exposed as self.dicom_ids, self.study_ids, self.subject_ids.
    Old shards without meta still work fine.
    """
    def __init__(self, root: str | Path):
        root = Path(root)
        shard_paths = sorted(root.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt under {root}")

        zs, ys = [], []
        dicom_ids, study_ids, subject_ids = [], [], []
        has_meta = None

        for p in shard_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            zs.append(d["z"])
            ys.append(d["y"])

            shard_has_meta = "dicom_ids" in d
            if has_meta is None:
                has_meta = shard_has_meta
            elif has_meta != shard_has_meta:
                raise ValueError(f"Mixed old/new shard formats under {root}")

            if shard_has_meta:
                dicom_ids.extend(d["dicom_ids"])
                study_ids.extend(d["study_ids"])
                subject_ids.extend(d["subject_ids"])

        self.z = torch.cat(zs, dim=0)  # [N, D]
        self.y = torch.cat(ys, dim=0)  # [N, 14]

        # Meta — None if old-format shards
        self.dicom_ids:   Optional[List[str]] = dicom_ids   if has_meta else None
        self.study_ids:   Optional[List[int]] = study_ids   if has_meta else None
        self.subject_ids: Optional[List[int]] = subject_ids if has_meta else None

        # O(1) dicom_id -> idx lookup (built only if meta present)
        self.dicom_to_idx: Optional[Dict[str, int]] = (
            {d: i for i, d in enumerate(dicom_ids)} if has_meta else None
        )

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int):
        return self.z[idx], self.y[idx]  # unchanged — no slowdown