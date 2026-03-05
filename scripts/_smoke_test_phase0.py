import numpy as np
from src.mimicvlm.utils.seed import set_seed
from src.mimicvlm.training.metrics import compute_multilabel_metrics

set_seed(0)

logits = np.random.randn(100, 14)
targets = (np.random.rand(100, 14) > 0.7).astype(int)

m = compute_multilabel_metrics(logits, targets)
print("macro_f1:", m.macro_f1)
print("macro_auroc:", m.macro_auroc)