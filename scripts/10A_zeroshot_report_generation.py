import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

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
from mimicvlm.report_generation.report_gen import run_zeroshot_report_gen
from mimicvlm.report_generation.radgraph_eval import evaluate_reports


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
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--check_rag", action="store_true",
                    help="Run a quick check of RAG retrieval + prompting without saving results")
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = ensure_dir(f"{args.save_dir}/k{args.k}")
    write_json(vars(args), os.path.join(save_dir, "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MedGemma...")
    medGemma = MedGemma(device=device)
    

    # Dataset — note: no transform, we pass raw PIL images to the processor
    dataset = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=lambda img: img,  
        label_policy="uncertain_as_negative",
    )

    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(args.limit))

    report_store = ReportStore(args.mimic_cxr_jpg_root)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil_2,
        pin_memory=False,
    )
  
    print("Running zero-shot report generation...")
    generated_path = os.path.join(save_dir, "generated_reports.json")
    run_zeroshot_report_gen(
        model=medGemma,
        dataloader=dataloader,
        output_path=generated_path,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nEvaluating generated reports with RadGraph F1...")
    results = evaluate_reports(generated_path=generated_path, report_store=report_store, reward_level="partial")
    print(f"\nROUGE-L F1:  {results['mean_rouge_l']:.4f}")
    print(f"Evaluated:   {results['n_evaluated']} studies")



if __name__ == "__main__":
    main()