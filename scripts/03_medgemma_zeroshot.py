import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_pil
from mimicvlm.inference.zero_shot import run_zero_shot
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.utils.io import ensure_dir, write_json
from mimicvlm.data.constants import CHEXPERT_LABELS_14


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "validate", "test"])
    ap.add_argument("--save_dir", type=str,
                    default="artifacts/medgemma_zeroshot")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optionally limit to first N samples for debugging")
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = ensure_dir(args.save_dir)
    write_json(vars(args), os.path.join(save_dir, "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MedGemma...")
    model = MedGemma(device=device)

    # Dataset — note: no transform, we pass raw PIL images to the processor
    dataset = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=lambda img: img,  
        label_policy="uncertain_as_negative",
    )

    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(args.limit))


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=False,
    )

    preds, targets, study_ids, n_failed = run_zero_shot(
        model=model,
        dataloader=dataloader,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\nParsing failures: {n_failed}/{len(study_ids)}")

    # For zero-shot hard labels {0,1}, AUROC is undefined — report F1 only
    # Pass preds directly as "logits" — since they're already 0/1,
    # sigmoid(large positive) ~ 1 and sigmoid(0) = 0.5, so we use a low threshold
    metrics = compute_multilabel_metrics(
        logits=preds,          
        targets=targets,
        threshold=0.5,
        compute_auroc=False,   # AUROC requires soft scores, not hard labels
    )

    print("\n--- Zero-shot MedGemma results ---")
    log_per_label_metrics(metrics, CHEXPERT_LABELS_14)

    # Save raw predictions for later analysis
    np.save(os.path.join(save_dir, "preds.npy"), preds)
    np.save(os.path.join(save_dir, "targets.npy"), targets)
    write_json(
        {"n_failed": n_failed, "n_total": len(study_ids),
         "macro_f1": metrics.macro_f1, "macro_precision": metrics.macro_precision,
         "macro_recall": metrics.macro_recall},
        os.path.join(save_dir, "summary.json"),
    )
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()