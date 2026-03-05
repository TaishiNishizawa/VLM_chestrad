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
    """
    def __init__(self, root: str | Path):
        root = Path(root)
        shard_paths = sorted(root.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt under {root}")

        zs, ys = [], []
        for p in shard_paths:
            d = torch.load(p, map_location="cpu", weights_only=True)
            zs.append(d["z"])
            ys.append(d["y"])

        self.z = torch.cat(zs, dim=0)  # [N, D]
        self.y = torch.cat(ys, dim=0)  # [N, 14]

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int):
        return self.z[idx], self.y[idx]
        
# class EmbeddingShardDataset(Dataset):
#     """
#     Loads (z, y) from sharded torch files produced by precompute script.

#     Each shard file is a dict:
#       - 'z': [M, D] float16/float32
#       - 'y': [M, 14]
#       - optional 'meta'
#     """
#     def __init__(self, root: str | Path, load_meta: bool = False):
#         self.root = Path(root)
#         self.load_meta = load_meta

#         shard_paths = sorted(self.root.glob("shard_*.pt"))

#         if not shard_paths:
#             raise FileNotFoundError(f"No shard_*.pt files found under {self.root}")

#         self._shards: List[ShardIndex] = []
#         total = 0
#         self._shard_sizes: List[int] = []

#         for p in shard_paths:
#             d = torch.load(p, map_location="cpu")
#             m = int(d["z"].shape[0])
#             self._shards.append(ShardIndex(shard_path=p, start=total, end=total + m))
#             self._shard_sizes.append(m)
#             total += m

#         self._n = total

#         # cache currently open shard to avoid repeated torch.load
#         self._cur_shard_i: Optional[int] = None
#         self._cur_data: Optional[Dict[str, torch.Tensor]] = None

#     def __len__(self) -> int:
#         return self._n

#     def _load_shard(self, shard_i: int) -> None:
#         if self._cur_shard_i == shard_i and self._cur_data is not None:
#             return
#         shard = self._shards[shard_i]
#         d = torch.load(shard.shard_path, map_location="cpu")
#         self._cur_shard_i = shard_i
#         self._cur_data = d

#     def __getitem__(self, idx: int):
#         if idx < 0 or idx >= self._n:
#             raise IndexError(idx)

#         # find shard
#         shard_i = 0
#         while not (self._shards[shard_i].start <= idx < self._shards[shard_i].end):
#             shard_i += 1

#         self._load_shard(shard_i)
#         shard = self._shards[shard_i]
#         local_i = idx - shard.start

#         z = self._cur_data["z"][local_i]  # [D]
#         y = self._cur_data["y"][local_i]  # [14]

#         if self.load_meta and "meta" in self._cur_data:
#             meta = self._cur_data["meta"][local_i]
#             return z, y, meta

#         return z, y
    