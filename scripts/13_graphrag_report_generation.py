import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
import json
from pathlib import Path
from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.retrieval.faiss_index import EmbeddingIndex
from mimicvlm.retrieval.report_store import ReportStore
from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_pil_2
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.inference.rag import run_rag, check_rag
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.utils.io import ensure_dir, write_json
from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.report_generation.report_gen import run_text_and_graph_rag_report_generation
from mimicvlm.report_generation.radgraph_eval import evaluate_reports
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.utils.io import to_device
from mimicvlm.graph.label_graph_retriever import LabelGraphRetriever

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root",  type=str, required=True)
    ap.add_argument("--biomedclip_embedding_dir", type=str, required=True,
                    help="artifacts/embeddings/biomedclip/train")
    ap.add_argument("--split",     type=str, default="test",
                    choices=["validate", "test"])
    ap.add_argument("--k_faiss",         type=int, default=3,
                    help="Number of retrieved reports from FAISS")
    ap.add_argument("--k_graph",         type=int, default=3,
                    help="Number of retrieved reports from graph")
    ap.add_argument("--save_dir",  type=str,
                    default="artifacts/medgemma_rag")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--check_rag", action="store_true",
                    help="Run a quick check of RAG retrieval + prompting without saving results")
    ap.add_argument("--graph_path", type=str,
                help="Path to prebuilt label co-occurrence graph JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = ensure_dir(f"{args.save_dir}/k{args.k_faiss}_g{args.k_graph}")
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

    embedded_dicoms = set(train_embedding_ds.dicom_ids)
    valid_indices = [
        i for i, d in enumerate(train_dataset.dicom_ids)
        if d in embedded_dicoms
    ]
    # Filter image_dataset to only the dicom_ids present in the embedding shards
    train_dataset_filtered = torch.utils.data.Subset(train_dataset, valid_indices)
    print(f"  Train Image dataset: {len(train_dataset)} → {len(train_dataset_filtered)} after filtering bad images")
    print(f"  Embedding dataset: {len(train_embedding_ds)}")
    assert len(train_dataset_filtered) == len(train_embedding_ds), "Still misaligned after filtering!"
   
    print("Building FAISS index from train embeddings...")
    index = EmbeddingIndex.from_shard_dir(train_dataset_filtered, train_embedding_ds)
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
    
    assert embedding_dataset.dicom_ids is not None, "Shards missing meta — rerun precompute"
    assert len(image_dataset) == len(embedding_dataset), "Image and EmbeddingShardDataset Mismatched!"

    if args.limit:
        image_dataset = torch.utils.data.Subset(image_dataset, range(args.limit))
        embedding_dataset = torch.utils.data.Subset(embedding_dataset, range(args.limit))

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil_2,
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

    # Construct MLP Head
    best_path = "artifacts/checkpoints/biomedclip_mlp/25656799/best.pt"  
    ckpt = torch.load(best_path, weights_only=False)
    tuned_thresholds = ckpt["tuned_thresholds"]
    threshold_labels = ckpt["threshold_labels"]

    head = MLPHead(in_dim=512, out_dim=14, hidden_dim=512, num_layers=3, dropout=0.1)
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device)
    head.eval()
    
    with open(args.graph_path) as f:
        graph = json.load(f)

    label_graph_retriever = LabelGraphRetriever(
        graph=graph,
        embedding_dataset=train_embedding_ds,
        report_store=report_store,
    )

    generated_path = os.path.join(save_dir, "generated_reports.json")

    run_text_and_graph_rag_report_generation(
        graph_retriever=label_graph_retriever,
        index=index, 
        report_store=report_store, 
        k_faiss=args.k_faiss,
        k_graph=args.k_graph,
        head=head,
        model=medGemma,
        device=device,
        image_dataloader=image_dataloader,
        embedding_dataloader=embedding_dataloader,
        tuned_thresholds=tuned_thresholds,
        threshold_labels=threshold_labels,
        output_path=generated_path,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nEvaluating generated reports with RadGraph F1...")
    results = evaluate_reports(generated_path=generated_path, report_store=report_store, reward_level="partial")
    print(f"\nROUGE-L F1:  {results['mean_rouge_l']:.4f}")
    print(f"Evaluated:   {results['n_evaluated']} studies")


if __name__ == "__main__":
    main()