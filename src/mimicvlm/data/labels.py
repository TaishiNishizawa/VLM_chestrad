# src/mimicvlm/data/labels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd

LabelPolicy = Literal["uncertain_as_negative", "ignore_uncertain"]


@dataclass(frozen=True)
class LabelEncoding:
    targets: np.ndarray  # shape (C,), float32 in {0,1} or {0,1,-1} depending on policy
    mask: np.ndarray     # shape (C,), bool; True where label is valid for loss/metrics


def encode_chexpert_row(
    row: pd.Series,
    label_cols: List[str],
    policy: LabelPolicy = "uncertain_as_negative",
) -> LabelEncoding:
    """
    row: pandas Series for one study
    label_cols: list of label column names in desired order
    policy:
      - "uncertain_as_negative": -1 and NaN -> 0, mask all True
      - "ignore_uncertain": -1 -> masked out, NaN -> 0 (still used), mask False for -1 only
        (If you want to ignore NaNs too, we can add another policy.)
    """
    raw = row[label_cols].to_numpy(dtype=np.float32)

    # NaNs are blanks
    nan_mask = np.isnan(raw)
    raw[nan_mask] = 0.0  # default treat missing as negative (common)

    if policy == "uncertain_as_negative":
        raw = np.where(raw == -1.0, 0.0, raw)
        targets = (raw > 0.0).astype(np.float32)
        mask = np.ones_like(targets, dtype=bool)
        return LabelEncoding(targets=targets, mask=mask)

    if policy == "ignore_uncertain":
        mask = raw != -1.0
        # Convert -1 to 0 for storage, but masked out anyway
        raw = np.where(raw == -1.0, 0.0, raw)
        targets = (raw > 0.0).astype(np.float32)
        return LabelEncoding(targets=targets, mask=mask)
