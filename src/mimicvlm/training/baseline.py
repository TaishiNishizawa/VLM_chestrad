# src/mimicvlm/training/baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
from mimicvlm.training.metrics import MultiLabelMetrics, find_optimal_thresholds
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

@dataclass
class BaselineEvalResult:
    """Lightweight container for baseline eval outputs."""
    loss: float
    metrics: MultiLabelMetrics

def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def _is_frozen(module: nn.Module) -> bool:
    return all((not p.requires_grad) for p in module.parameters())


def assert_frozen_encoder(encoder: nn.Module) -> None:
    """
    Call once after you construct the encoder (and freeze it).
    Raises a clear error if anything is still trainable.
    """
    if not _is_frozen(encoder):
        # Find first offending parameter for a useful message
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                raise RuntimeError(
                    f"Encoder is not frozen: parameter '{name}' has requires_grad=True"
                )
        raise RuntimeError("Encoder is not frozen (some parameters require grad).")


def freeze_encoder(encoder: nn.Module) -> None:
    """Convenience helper. Sets eval() and requires_grad=False everywhere."""
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False


def run_one_epoch_baseline(
    *,
    encoder: nn.Module,
    head: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """
    Baseline training epoch:
      - encoder is frozen
      - head is trained with BCEWithLogits-style loss

    Returns: mean train loss over samples
    """
    # Safety: make sure we're not accidentally training encoder
    encoder.eval()
    head.train()

    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    total_loss = 0.0
    total_n = 0
    
    pbar = tqdm(
        dataloader,
        desc="Train",
        total=len(dataloader),
        leave=False,
    )
    
    for images, targets, _meta in pbar:
        if images is None or targets is None: 
            continue
        images = _to_device(images, device)
        targets = _to_device(targets, device)

        # Encode without graph to save memory and prevent encoder grads.
        with torch.no_grad():
            z = encoder(images)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.amp.autocast(device_type=device.type):
                logits = head(z)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = head(z)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
            optimizer.step()

        bsz = int(images.size(0))
        total_loss += float(loss.item()) * bsz
        total_n += bsz

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_baseline(
    *,
    encoder: nn.Module,
    head: nn.Module,
    criterion: Optional[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_metrics_fn,
    threshold: float | np.ndarray = 0.5,
    amp: bool = False,
) -> BaselineEvalResult:
    """
    Baseline evaluation:
      - encoder frozen
      - head eval
      - collects logits + targets for metric computation

    compute_metrics_fn(logits_np, targets_np) should return either:
      - dict[str, float], or
      - an object with attributes like macro_auroc, macro_f1, etc.
    """
    encoder.eval()
    head.eval()

    total_loss = 0.0
    total_n = 0

    all_logits = []
    all_targets = []

    for images, targets, _meta in dataloader:
        if images is None or targets is None: 
            continue

        images = _to_device(images, device)
        targets = _to_device(targets, device)

        if amp and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                z = encoder(images)
                logits = head(z)
                loss = criterion(logits, targets) if criterion is not None else None
        else:
            z = encoder(images)
            logits = head(z)
            loss = criterion(logits, targets) if criterion is not None else None

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if loss is not None:
            bsz = int(images.size(0))
            total_loss += float(loss.item()) * bsz
            total_n += bsz

    logits_np = torch.cat(all_logits, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy()
    mean_loss = (total_loss / max(total_n, 1)) if criterion is not None else float("nan")

    return BaselineEvalResult(loss=mean_loss, metrics=compute_metrics_fn(logits_np, targets_np, threshold))

def run_one_epoch_head_only(
    *,
    head: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
    grad_clip_norm: Optional[float] = None,
) -> float:
    head.train()
    scaler = torch.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    total_loss = 0.0
    total_n = 0

    # pbar = tqdm(dataloader, desc="Train(head-only)", total=len(dataloader), leave=False)
    for batch in dataloader:
        # batch could be (z,y) or (z,y,meta)
        z, targets = batch[0], batch[1]
        z = _to_device(z, device).float()
        targets = _to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.amp.autocast(device_type=device.type):
                logits = head(z)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = head(z)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip_norm)
            optimizer.step()

        bsz = int(z.size(0))
        total_loss += float(loss.item()) * bsz
        total_n += bsz

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_head_only(
    *,
    head: nn.Module,
    criterion: Optional[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_metrics_fn,
    amp: bool = False,
    threshold: float | np.ndarray = 0.5,
) -> BaselineEvalResult:
    head.eval()
    total_loss = 0.0
    total_n = 0
    all_logits, all_targets = [], []

    for batch in dataloader:
        z, targets = batch[0], batch[1]
        z = _to_device(z, device).float()
        targets = _to_device(targets, device)

        if amp and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                logits = head(z)
        else:
            logits = head(z)

        loss = criterion(logits, targets) if criterion is not None else None
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if loss is not None:
            bsz = int(z.size(0))
            total_loss += float(loss.item()) * bsz
            total_n += bsz

    logits_np = torch.cat(all_logits, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy()
    mean_loss = (total_loss / max(total_n, 1)) if criterion is not None else float("nan")

    return BaselineEvalResult(
        loss=mean_loss,
        metrics=compute_metrics_fn(logits_np, targets_np, threshold),  # return as-is
    )

def tune_threshold_on_val(*,
    head: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            z, targets = batch[0].to(device).float(), batch[1]
            all_logits.append(head(z).cpu())
            all_targets.append(targets.cpu())
    val_logits_np = torch.cat(all_logits).numpy()
    val_targets_np = torch.cat(all_targets).numpy()
    return find_optimal_thresholds(val_logits_np, val_targets_np)