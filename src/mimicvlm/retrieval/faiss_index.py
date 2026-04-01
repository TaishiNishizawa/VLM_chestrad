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

    def __init__(self, 
        embeddings: np.ndarray, 
        image_keys: list[tuple[int, int, str]],
        labels: np.ndarray | None = None, ):
        """
        embeddings: (N, D) float32
        image_keys:  list of N (subject_id, study_id, dicom_id) tuples, parallel to embeddings
        labels: (N,) int32 — optional labels for each embedding, not used for retrieval but can be stored for later analysis
        """
        assert len(embeddings) == len(image_keys)
        if labels is not None:
                assert len(labels) == len(image_keys)
        
        self.image_keys = image_keys
        self.dim = embeddings.shape[1]
        self.labels = labels

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
        labels = embedding_dataset.y.numpy().astype(np.int32) 

        return cls(embeddings, image_keys, labels=labels)

    def query(
        self,
        query_emb: np.ndarray,
        k: int,
        return_labels: bool = False,
        exclude_self: bool = True,          # skip the query itself if it's in the index
    ) -> list[tuple[int, int, str]] | tuple[list[tuple[int, int, str]], np.ndarray]:
        """
        query_emb:     (D,) float32 — single image embedding
        k:             number of neighbors to return
        return_labels: if True, also return (N, 14) label matrix for neighbors
        exclude_self:  fetch k+1 and drop the top hit (handles query-in-index case)

        Returns:
            keys                       if return_labels=False
            (keys, labels_matrix)      if return_labels=True
                labels_matrix is shape (k, 14), or (k, 14) of NaNs if
                labels were not loaded at construction.
        """
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        q /= max(np.linalg.norm(q), 1e-12)

        fetch_k = k + 1 if exclude_self else k
        _, indices = self.index.search(q, fetch_k)

        valid_indices = [
            i for i in indices[0]
            if 0 <= i < len(self.image_keys)
        ][:k]

        keys = [self.image_keys[i] for i in valid_indices]

        if not return_labels:
            return keys

        if self.labels is None:
            # Graceful fallback: return NaN matrix so callers don't hard-crash
            label_matrix = np.full((len(valid_indices), 14), np.nan, dtype=np.float32)
        else:
            label_matrix = self.labels[valid_indices]  # (k, 14)

        return keys, label_matrix