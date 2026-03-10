import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.retrieval.faiss_index import EmbeddingIndex
from mimicvlm.retrieval.report_store import ReportStore
from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_pil
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.inference.rag import run_rag
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.utils.io import ensure_dir, write_json
from mimicvlm.data.constants import CHEXPERT_LABELS_14


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root",  type=str, required=True)
    ap.add_argument("--biomedclip_embedding_dir", type=str, required=True,
                    help="artifacts/embeddings/biomedclip/train")
    ap.add_argument("--split",     type=str, default="test",
                    choices=["validate", "test"])
    ap.add_argument("--k",         type=int, default=3,
                    help="Number of retrieved reports")
    ap.add_argument("--save_dir",  type=str,
                    default="artifacts/medgemma_rag")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = ensure_dir(f"{args.save_dir}/k{args.k}")
    write_json(vars(args), os.path.join(save_dir, "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Training MimicCXRDataset and Embedding...")
    train_dataset = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split="train",
        transform=lambda img: img,  
        label_policy="uncertain_as_negative",
    )
    train_embedding_ds = EmbeddingShardDataset(f"{args.biomedclip_embedding_dir}/train")

    print("Building FAISS index from train embeddings...")
    index = EmbeddingIndex.from_shard_dir(train_dataset, train_embedding_ds)
    print(f"  Index size: {index.index.ntotal} vectors, dim={index.dim}")

    report_store = ReportStore(args.mimic_cxr_jpg_root)

    print(f"Loading {args.split} MimicCXRDataset and Embedding...")
    image_dataset = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=lambda img: img,  
        label_policy="uncertain_as_negative",
    )
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

    print("Loading MedGemma...")
    medGemma = MedGemma(device=device)
    print("Running RAG inference...")
    run_rag(
        model=medGemma, 
        index=index, 
        report_store=report_store, 
        image_dataloader=image_dataloader, 
        embedding_dataloader=embedding_dataloader, 
        k=args.k, max_new_tokens=args.max_new_tokens)
    # preds, targets, study_ids, n_failed, n_missing = run_rag(
    #     model=model,
    #     encoder=encoder,
    #     index=index,
    #     report_store=report_store,
    #     dataloader=dataloader,
    #     k=args.k,
    #     max_new_tokens=args.max_new_tokens,
    # )
    return 
    metrics = compute_multilabel_metrics(
        logits=preds,
        targets=targets,
        threshold=0.5,
        compute_auroc=False,
    )

    print(f"\n--- MedGemma + Text-RAG (k={args.k}) results ---")
    log_per_label_metrics(metrics, CHEXPERT_LABELS_14)

    np.save(os.path.join(save_dir, "preds.npy"), preds)
    np.save(os.path.join(save_dir, "targets.npy"), targets)
    write_json(
        {
            "k": args.k,
            "n_failed": n_failed,
            "n_missing_reports": n_missing,
            "n_total": len(study_ids),
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "macro_accuracy": metrics.macro_accuracy,
        },
        os.path.join(save_dir, "summary.json"),
    )
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()