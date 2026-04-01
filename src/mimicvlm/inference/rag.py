from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.retrieval.faiss_index import EmbeddingIndex
from mimicvlm.retrieval.report_store import ReportStore
from mimicvlm.inference.prompt import build_rag_messages
from mimicvlm.inference.json_parser import parse_label_json


def label_recall_at_k(query_labels: np.ndarray, retrieved_labels: np.ndarray) -> float:
    """
    query_labels:     shape (14,) binary vector for the query image
    retrieved_labels: shape (k, 14) binary matrix for the k retrieved images
    Returns fraction of query's positive labels covered by any retrieved report.
    No Finding (index 8) is treated as a real label, not a special case.
    """
    NO_FINDING_IDX = 8  # position in CHEXPERT_LABELS_14

    positive_mask = query_labels.astype(bool)

    if positive_mask.sum() == 0:
        # Unlabeled / all-zero row — genuinely ambiguous, skip
        return np.nan

    # Standard case: includes No Finding if it's the only positive label
    covered = retrieved_labels[:, positive_mask].any(axis=0)
    return float(covered.mean())

def check_rag(
    model: MedGemma,
    index: EmbeddingIndex,
    report_store: ReportStore,
    image_dataloader: DataLoader,      # yields (list[PIL], _, study_ids)
    embedding_dataloader: DataLoader,  # yields (embeddings, targets)
    k: int = 3,
):
    """
    Run a check of the RAG pipeline on giving dataset to see if top-k retrieved cases are actually similar labels. 
    """
    recalls = []
    query_labels_list = []
    neighbor_labels_list = []
    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="RAG check",
    ):
        images, _, study_ids = img_batch     # discard image dataset targets
        emb, targets = emb_batch            # (B, D), (B, 14) — ground truth from shards

        B = len(images)

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()

        # Retrieve + fetch reports
        for i in range(B):
            neighbor_ids, neighbor_labels = index.query(emb[i], k=k, return_labels=True)
            query_labels = targets[i].numpy() if isinstance(targets, torch.Tensor) else targets[i]
            query_labels_list.append(query_labels)
            neighbor_labels_list.append(neighbor_labels)

            recall = label_recall_at_k(query_labels=query_labels, retrieved_labels=neighbor_labels)
            recalls.append(recall)

    recalls_arr = np.array(recalls, dtype=np.float32)
    query_labels_arr = np.stack(query_labels_list)  # (N, 14) — collect during loop (see below)
    neighbor_labels_arr = np.stack(neighbor_labels_list)  # (N, k, 14)

    NO_FINDING_IDX = 8
    is_no_finding = (query_labels_arr[:, NO_FINDING_IDX] == 1)
    is_pathology  = ~is_no_finding
    is_ambiguous  = (query_labels_arr.sum(axis=1) == 0)

    def summarize(mask, name):
        vals = recalls_arr[mask & ~np.isnan(recalls_arr)]
        if len(vals) == 0:
            print(f"{name}: no samples")
            return
        print(f"{name}: n={len(vals)}, recall={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n=== Label Recall@{k} Breakdown ===")
    summarize(np.ones(len(recalls_arr), dtype=bool), "Overall        ")
    summarize(is_no_finding,                          "No Finding     ")
    summarize(is_pathology,                           "Pathology only ")
    summarize(is_ambiguous,                           "Ambiguous (0s) ")
    print(f"Ambiguous rows (all-zero labels): {is_ambiguous.sum()}")

    print(f"\n=== Per-label Recall@{k} (pathology cases only) ===")
    for j, label in enumerate(CHEXPERT_LABELS_14):
        if label == "No Finding":
            continue
        pos_mask = query_labels_arr[:, j].astype(bool)
        if pos_mask.sum() == 0:
            continue
        covered = neighbor_labels_arr[pos_mask, :, j].any(axis=1)
        print(f"  {label:<30} n={pos_mask.sum():4d}  recall={covered.mean():.4f}")

def run_rag(
    model: MedGemma,
    index: EmbeddingIndex,
    report_store: ReportStore,
    image_dataloader: DataLoader,      # yields (list[PIL], _, study_ids)
    embedding_dataloader: DataLoader,  # yields (embeddings, targets)
    k: int = 3,
    max_new_tokens: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    all_preds, all_targets, all_ids = [], [], []
    n_failed = 0
    n_missing_reports = 0

    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="RAG inference",
    ):
        images, _, study_ids = img_batch     # discard image dataset targets
        emb, targets = emb_batch            # (B, D), (B, 14) — ground truth from shards

        B = len(images)
        targets_np = targets.numpy()

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        # Retrieve + fetch reports
        messages_batch = []
        for i in range(B):
            # index.query returns list of k neighbor (subject_id, study_id, dicom_id)
            neighbor_ids = index.query(emb[i], k=k)
            reports = [r for (subject_id, study_id, dicom_id) in neighbor_ids if (r := report_store.get((subject_id, study_id, dicom_id)))]

            if len(reports) < k:
                n_missing_reports += 1
            
            messages_batch.append(build_rag_messages(reports))

        print(f"\n[DEBUG] Sample messages batch (first item):\n{messages_batch[0][:3]}...")  # Show first 3 messages for first item

        # Batched MedGemma generation
        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )

        for i, raw in enumerate(raw_outputs):
            pred = parse_label_json(raw)
            if pred is None:
                n_failed += 1
                pred = np.zeros(14, dtype=np.float32)

            all_preds.append(pred)
            all_targets.append(targets_np[i])
            all_ids.append(study_ids[i])

    total = len(all_ids)
    if n_failed > 0:
        print(f"\n[WARNING] JSON parse failures: {n_failed}/{total} "
              f"({100 * n_failed / total:.1f}%)")
    if n_missing_reports > 0:
        print(f"[WARNING] Samples with <{k} retrieved reports: "
              f"{n_missing_reports}/{total}")

    # return (
    #     np.stack(all_preds),
    #     np.stack(all_targets),
    #     all_ids,
    #     n_failed,
    #     n_missing_reports,
    # )
    return None