import numpy as np
import json
from pathlib import Path
from mimicvlm.data.constants import CHEXPERT_LABELS_14
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset


def build_cooccurrence_graph(
    embedding_dataset: EmbeddingShardDataset,
    output_path: str | Path,
    min_cooccurrence: int = 10,  # filter spurious edges
) -> dict:
    """
    Build a label co-occurrence graph from training set labels.
    
    Nodes: 14 CheXpert labels
    Edges: weighted by normalized pointwise mutual information (NPMI)
           between label pairs across training studies.
    Also stores: for each label, the list of training indices
                 where that label is positive (for graph-based retrieval).
    """
    labels = embedding_dataset.y.numpy()  # (N, 14)
    N, L = labels.shape

    # --- Edge weights: NPMI ---
    # NPMI(a,b) = log[P(a,b) / P(a)P(b)] / -log[P(a,b)]
    # ranges from -1 (never co-occur) to 1 (always co-occur)
    p_single = labels.mean(axis=0)          # (14,)  marginal probabilities
    p_joint  = (labels.T @ labels) / N      # (14,14) joint probabilities

    npmi = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            pij = p_joint[i, j]
            if pij < 1e-10:
                continue
            pmi = np.log(pij / (p_single[i] * p_single[j] + 1e-10))
            npmi[i, j] = pmi / (-np.log(pij + 1e-10))

    # --- Inverted index: label -> training sample indices ---
    label_to_indices = {
        CHEXPERT_LABELS_14[i]: np.where(labels[:, i] == 1)[0].tolist()
        for i in range(L)
    }

    # --- Build graph dict ---
    graph = {
        "nodes": CHEXPERT_LABELS_14,
        "edges": {},        # label -> [(neighbor_label, npmi_weight)]
        "label_to_indices": label_to_indices,
        "label_prevalence": {
            CHEXPERT_LABELS_14[i]: float(p_single[i]) for i in range(L)
        }
    }

    for i, label_i in enumerate(CHEXPERT_LABELS_14):
        neighbors = []
        for j, label_j in enumerate(CHEXPERT_LABELS_14):
            if i == j:
                continue
            count = int((labels[:, i] * labels[:, j]).sum())
            if count < min_cooccurrence:
                continue
            neighbors.append((label_j, float(npmi[i, j])))
        # Sort by NPMI descending
        graph["edges"][label_i] = sorted(neighbors, key=lambda x: -x[1])

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Graph saved: {L} nodes, edges written to {output_path}")
    return graph