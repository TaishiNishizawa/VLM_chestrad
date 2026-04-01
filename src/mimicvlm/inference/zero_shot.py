from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.inference.json_parser import parse_label_json
from mimicvlm.inference.prompt import (
    build_messages,
    logits_to_prompt_text,
    build_biomedclip_messages,
)
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.utils.io import to_device

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

def run_zero_shot_with_biomedclip(
    head : MLPHead,
    model: MedGemma,
    device: torch.device,
    image_dataloader: DataLoader,   
    embedding_dataloader: DataLoader,  
    tuned_thresholds: np.ndarray, 
    threshold_labels: list[str],
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

    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="Running zero-shot inference with BioMedCLIP embeddings",
    ):
        images, _, study_ids = img_batch     # discard image dataset targets
        emb, targets = emb_batch            # (B, D), (B, 14) — ground truth from shards

        B = len(images)

        # Keep images as cpu
        emb = to_device(emb, device).float()
        
        # Generate logit predictions from MLP head
        with torch.no_grad():
            logits = head(emb)
            probs = torch.sigmoid(logits)
            # print(f"\n[DEBUG] MLP head logits (first item): {logits[0].detach().cpu().numpy()}")  
        
        messages_batch = []
        for i in range(B):
            # print("Tuned Thresholds: ", tuned_thresholds)
            # print(f"[DEBUG] MLP head probs\n: {probs[i].detach().cpu().numpy()}")
            prompt_text = logits_to_prompt_text(probs[i], threshold_labels, tuned_thresholds)
            messages_batch.append(build_biomedclip_messages(prompt_text))
        
        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )


        for i, raw in enumerate(raw_outputs):
            print(f"\n[DEBUG] Raw MedGemma output sample {i}:\n{raw}\n")
            pred = parse_label_json(raw)
            print(f"[DEBUG] Parsed prediction for {study_ids[i]}: {pred}")  
            print(f"Ground truth for {study_ids[i]}: {targets[i].numpy()}")
            print("----------------------------------------------------\n")
            if pred is None:
                n_failed += 1
                pred = np.zeros(14, dtype=np.float32)
            all_preds.append(pred)
            all_targets.append(targets[i].cpu().numpy())
            all_ids.append(study_ids[i])

    return (
        np.stack(all_preds),    # (N, 14)
        np.stack(all_targets),  # (N, 14)
        all_ids,
        n_failed,
    )