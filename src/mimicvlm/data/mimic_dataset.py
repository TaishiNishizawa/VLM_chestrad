# src/mimicvlm/data/mimic_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.data.labels import encode_chexpert_row, LabelPolicy

class MimicCXRDataset(Dataset):
    def __init__(
        self,
        mimic_cxr_jpg_root: Union[str, Path],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        label_policy: LabelPolicy = "uncertain_as_negative",
        bad_image_log: Optional[Union[str, Path]] = None,  
        print_skip_every: int = 2000,                      
    ) -> None:
        split_csv = os.path.join(mimic_cxr_jpg_root, "mimic-cxr-2.0.0-split.csv")
        image_root = os.path.join(mimic_cxr_jpg_root, "files")
        label_csv= os.path.join(mimic_cxr_jpg_root, "mimic-cxr-2.0.0-chexpert.csv")
        
        # --- load split df ---
        df = pd.read_csv(split_csv)
        if split is not None:
            split_l = split.lower()
            if split_l not in ["train", "validate", "test"]:
                raise ValueError(f"Invalid split: {split}")
            df = df[df["split"].str.lower() == split_l].reset_index(drop=True)

        self.df = df

        label_df = pd.read_csv(label_csv)
        if label_df.duplicated(["subject_id", "study_id"]).any():
            dups = label_df[label_df.duplicated(["subject_id", "study_id"], keep=False)]
            raise ValueError(f"Duplicate labels for some (subject_id, study_id). Example:\n{dups.head()}")
        self.label_df = label_df.set_index(["subject_id", "study_id"])

        self.transform = transform
        self.label_policy = label_policy
        self.image_root = Path(image_root)

        self.bad_image_log = str(bad_image_log) if bad_image_log is not None else None
        self.print_skip_every = int(print_skip_every)
        self._skip_no_label = 0
        self._skip_bad_image = 0
        self._skip_missing_image = 0

        # 1) Precompute per-row arrays (fast scalar access vs df.iloc each time)
        # Note: dicom_id can be int in csv; force to str once here.
        self.subject_ids = df["subject_id"].to_numpy(dtype=np.int64, copy=True)
        self.study_ids = df["study_id"].to_numpy(dtype=np.int64, copy=True)
        self.dicom_ids = df["dicom_id"].astype(str).to_numpy(copy=True)

        # 2) Precompute label lookup dict:
        #    (subject_id, study_id) -> np.float32[14]
        # This still uses encode_chexpert_row, but only ONCE per (subject, study).
        self.targets_by_study: Dict[Tuple[int, int], np.ndarray] = {}
        for (sid, stid), row in self.label_df.iterrows():
            enc = encode_chexpert_row(row, CHEXPERT_LABELS_14, policy=self.label_policy)
            self.targets_by_study[(int(sid), int(stid))] = enc.targets.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.df)

    def _mimic_jpg_path(self, subject_id: int, study_id: int, dicom_id: str) -> Path:
        # subject_id like 10000032 -> "p10"
        pxx = f"p{str(subject_id)[:2]}"
        return (
            self.image_root
            / pxx
            / f"p{subject_id}"
            / f"s{study_id}"
            / f"{dicom_id}.jpg"
        )
    
    def _append_log(self, msg: str) -> None:
        if self.bad_image_log is None:
            return
        # Simple append; safe enough (may interleave slightly across workers, but lines are usually fine)
        try:
            with open(self.bad_image_log, "a") as f:
                f.write(msg.rstrip() + "\n")
        except Exception:
            pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Union[str, int]]]:
        subject_id = int(self.subject_ids[idx])
        study_id = int(self.study_ids[idx])
        dicom_id = str(self.dicom_ids[idx])

        image_path = self._mimic_jpg_path(subject_id, study_id, dicom_id)

        targets = self.targets_by_study.get((subject_id, study_id), None)
        if targets is None:
            # (keeping your current behavior; ideally avoid printing here)
            self._skip_no_label += 1
            return None
        
        try:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
        except FileNotFoundError:
            self._skip_missing_image += 1
            self._append_log(
                f"MISSING\t{subject_id}\t{study_id}\t{dicom_id}\t{image_path}"
            )
            return None
        except (OSError, UnidentifiedImageError) as e:
            self._skip_bad_image += 1
            self._append_log(f"BAD_IMAGE\t{subject_id}\t{study_id}\t{dicom_id}\t{image_path}\t{repr(e)}")
            return None

        if self.transform is not None:
            img = self.transform(img) 
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        y = torch.from_numpy(targets).float()
        meta = {"subject_id": subject_id, "study_id": study_id, "image_path": str(image_path)}
        return img, y, meta


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def collate_pil(batch):
    """
    Like collate_skip_none but keeps images as list[PIL.Image].
    Use with MedGemma and any model that expects PIL input.
    Returns:
        images: list[PIL.Image]
        targets: (B, 14) tensor
        study_ids: list[str]
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    images    = [x[0] for x in batch]               # list[PIL.Image]
    targets   = torch.stack([x[1] for x in batch])  # (B, 14)
    study_ids = [str(x[2]["study_id"]) for x in batch]
    return images, targets, study_ids