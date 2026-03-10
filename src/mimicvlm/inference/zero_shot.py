from __future__ import annotations
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.inference.json_parser import parse_label_json
from mimicvlm.inference.prompt import build_messages


def run_zero_shot(
    model: MedGemma,
    dataloader: DataLoader,     
    max_new_tokens: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Run zero-shot inference over a dataloader.
    Returns:
        preds:      (N, 14) float32 binary predictions
        targets:    (N, 14) float32 ground truth
        study_ids:  list of N study ids
        n_failed:   number of samples where JSON parsing failed (defaulted to all zeros)
    """
    all_preds, all_targets, all_ids = [], [], []
    n_failed = 0

    for nbatch, batch in enumerate(dataloader):
        if nbatch % 100 == 0:
            print(f"Processing batch {nbatch} / {len(dataloader)}...")
        images, targets, study_ids = batch  # list[PIL], (B,14) tensor, list[str]

        # Build one message list per image in the batch
        messages_batch = [build_messages() for _ in images]
        
        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )

        for i, raw in enumerate(raw_outputs):
            # print(f"Raw output for {study_ids[i]}: {raw}")  # debug
            pred = parse_label_json(raw)
            if pred is None:
                n_failed += 1
                pred = np.zeros(14, dtype=np.float32)
            all_preds.append(pred)
            all_targets.append(targets[i].numpy())
            all_ids.append(study_ids[i])

    return (
        np.stack(all_preds),    # (N, 14)
        np.stack(all_targets),  # (N, 14)
        all_ids,
        n_failed,
    )