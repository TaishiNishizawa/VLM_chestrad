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
from mimicvlm.inference.rag import run_rag, check_rag
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.utils.io import ensure_dir, write_json
from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.graph.build_cooccurence_graph import build_cooccurrence_graph

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root",  type=str, required=True)
    ap.add_argument("--biomedclip_embedding_dir", type=str, required=True,
                    help="artifacts/embeddings/biomedclip/train")
    ap.add_argument("--save_dir",  type=str,
                    default="artifacts/medgemma_rag")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--min_cooccurrence", type=int, default=10,
                    help="Minimum co-occurrence count to include an edge in the graph")
    return ap.parse_args()


def main():
    args = parse_args()
    graph_save_dir = ensure_dir(f"{args.save_dir}")
    write_json(vars(args), os.path.join(graph_save_dir, "args.json"))

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
    
    graph_save_path = os.path.join(graph_save_dir, "cooccurrence_graph.json")
    graph : dict = build_cooccurrence_graph(train_embedding_ds, graph_save_path, args.min_cooccurrence)

    

if __name__ == "__main__":
    main()