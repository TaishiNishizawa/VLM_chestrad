from __future__ import annotations
import numpy as np
import faiss
from pathlib import Path
from mimicvlm.data.mimic_dataset import MimicCXRDataset
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset

class EmbeddingIndex:
    """
    FAISS flat L2 index over BiomedCLIP embeddings.
    Built once from training set shards, queried at inference time.
    """

    def __init__(self, embeddings: np.ndarray, image_keys: list[tuple[int, int, str]]):
        """
        embeddings: (N, D) float32
        image_keys:  list of N (subject_id, study_id, dicom_id) tuples, parallel to embeddings
        """
        assert len(embeddings) == len(image_keys)
        self.image_keys = image_keys
        self.dim = embeddings.shape[1]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-12)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(normed.astype(np.float32))
        self._normed = normed

    @classmethod
    def from_shard_dir(
        cls,
        mimic_dataset: MimicCXRDataset,
        embedding_dataset: EmbeddingShardDataset,
    ) -> EmbeddingIndex:
        # Use the embedding shard's own meta (already excludes the 16 bad images)
        image_keys = list(zip(
            embedding_dataset.subject_ids,
            embedding_dataset.study_ids,
            embedding_dataset.dicom_ids,
        ))
        embeddings = embedding_dataset.z.numpy().astype(np.float32)
        return cls(embeddings, image_keys)

    def query(self, query_emb: np.ndarray, k: int) -> list[tuple[int, int, str]]:
        """
        query_emb: (D,) float32 — single image embedding
        Returns list of k nearest neighbors as
        (subject_id, study_id, dicom_id) tuples.
        """
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        q /= max(np.linalg.norm(q), 1e-12)

        _, indices = self.index.search(q, k + 1)

        return [
            self.image_keys[i]
            for i in indices[0]
            if 0 <= i < len(self.image_keys)
        ][:k]