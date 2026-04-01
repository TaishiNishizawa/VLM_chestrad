import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_pil
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.inference.zero_shot import run_zero_shot, run_zero_shot_with_biomedclip
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.utils.io import ensure_dir, write_json
from mimicvlm.data.constants import CHEXPERT_LABELS_14


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--biomedclip_embedding_dir", type=str, required=True,
                    help="artifacts/embeddings/biomedclip/train")
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
    medGemma = MedGemma(device=device)

    # Dataset — note: no transform, we pass raw PIL images to the processor
    image_dataset = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=lambda img: img,  
        label_policy="uncertain_as_negative",
    )

    # Load embedding dataset to get biomedCLIP-based logits
    embedding_dataset = EmbeddingShardDataset(
        os.path.join(args.biomedclip_embedding_dir, args.split)  
    )  

    # Filter image_dataset to only the dicom_ids present in the embedding shards
    assert embedding_dataset.dicom_ids is not None, "Shards missing meta — rerun precompute"
    embedded_dicoms = set(embedding_dataset.dicom_ids)
    valid_indices = [
        i for i, d in enumerate(image_dataset.dicom_ids)
        if d in embedded_dicoms
    ]
    image_dataset_filtered = torch.utils.data.Subset(image_dataset, valid_indices)

    print(f"  Image dataset: {len(image_dataset)} → {len(image_dataset_filtered)} after filtering bad images")
    print(f"  Embedding dataset: {len(embedding_dataset)}")
    assert len(image_dataset_filtered) == len(embedding_dataset), "Still misaligned after filtering!"


    if args.limit:
        image_dataset = torch.utils.data.Subset(image_dataset, range(args.limit))
        embedding_dataset = torch.utils.data.Subset(embedding_dataset, range(args.limit))

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=False,
    )

    embedding_dataloader = DataLoader(
        embedding_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Construct MLP Head
    best_path = "artifacts/checkpoints/biomedclip_mlp/25656799/best.pt"  
    ckpt = torch.load(best_path, weights_only=False)
    tuned_thresholds = ckpt["tuned_thresholds"]
    threshold_labels = ckpt["threshold_labels"]

    head = MLPHead(in_dim=512, out_dim=14, hidden_dim=512, num_layers=3, dropout=0.1)
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device)
    head.eval()

    preds, targets, study_ids, n_failed = run_zero_shot_with_biomedclip(
        head=head,
        model=medGemma,
        device=device,
        image_dataloader=image_dataloader,
        embedding_dataloader=embedding_dataloader,
        tuned_thresholds=tuned_thresholds,
        threshold_labels=threshold_labels,
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
        compute_auroc=False,  
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