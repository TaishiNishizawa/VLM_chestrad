import numpy as np
import json
from pathlib import Path
from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.retrieval.report_store import ReportStore

class LabelGraphRetriever:
    """
    Given predicted labels for a query image, traverses the co-occurrence
    graph to retrieve training samples — complementary to FAISS retrieval.
    """

    def __init__(
        self,
        graph: dict,
        embedding_dataset: EmbeddingShardDataset,
        report_store: ReportStore,
    ):
        self.graph = graph
        self.embedding_dataset = embedding_dataset
        self.report_store = report_store
        # Precompute dicom_id lookup for index -> (subject_id, study_id, dicom_id)
        self.idx_to_key = list(zip(
            embedding_dataset.subject_ids,
            embedding_dataset.study_ids,
            embedding_dataset.dicom_ids,
        ))

    def retrieve(
        self,
        predicted_labels: np.ndarray,   # (14,) binary, from zero-shot MedGemma
        k: int = 3,
        hop: int = 1,                   # graph traversal depth
        npmi_threshold: float = 0.1,    # minimum edge weight to traverse
    ) -> list[str]:
        """
        Returns k report strings retrieved via graph traversal.
        
        Strategy:
        1. Seed with predicted positive labels
        2. Expand via graph edges up to `hop` hops
        3. Score candidate training samples by how many
           entry-point labels they cover
        4. Return top-k reports
        """
        # --- Seed nodes: predicted positive labels ---
        seed_labels = [
            CHEXPERT_LABELS_14[i]
            for i in range(14)
            if predicted_labels[i] == 1
        ]

        if not seed_labels:
            return []

        # --- Expand: collect candidate labels via graph traversal ---
        candidate_labels = set(seed_labels)
        if hop >= 1:
            for label in seed_labels:
                for neighbor_label, weight in self.graph["edges"].get(label, []):
                    if weight >= npmi_threshold:
                        candidate_labels.add(neighbor_label)

        # --- Score training samples ---
        # Each sample gets +1 for each seed label it has positive
        sample_scores: dict[int, float] = {}
        for label in seed_labels:                   # only seeds drive score, not expanded
            for idx in self.graph["label_to_indices"].get(label, []):
                sample_scores[idx] = sample_scores.get(idx, 0) + 1

        # Boost samples that also cover expanded neighbor labels
        for label in candidate_labels - set(seed_labels):
            for idx in self.graph["label_to_indices"].get(label, []):
                if idx in sample_scores:            # only boost existing candidates
                    sample_scores[idx] += 0.5       # partial credit for expanded labels

        if not sample_scores:
            return []

        # --- Rank and fetch top-k reports ---
        top_indices = sorted(sample_scores, key=lambda x: -sample_scores[x])[:k]
        reports = []
        for idx in top_indices:
            subject_id, study_id, dicom_id = self.idx_to_key[idx]
            report = self.report_store.get((subject_id, study_id, dicom_id))
            if report:
                reports.append(report)

        return reports