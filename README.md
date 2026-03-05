### VLM_graphrag / mimicvlm

**mimicvlm** is a small PyTorch codebase for training and evaluating vision–language / vision models on the MIMIC-CXR dataset, with a focus on:

- **BioMedCLIP encoders** (`BiomedCLIP`)
- **MLP heads** for multi‑label CheXpert classification (14 labels)
- **Precomputing and reusing image embeddings** for faster experimentation

The code is structured as a Python package `mimicvlm` with CLI scripts under `scripts/` and SLURM job wrappers under `slurm/`.

---

### Project layout

- **`src/mimicvlm/`**: Python package
  - **`data/`**: MIMIC‑CXR dataset class, embedding datasets, label utilities, transforms
  - **`models/encoders/`**: `biomedclip.py`, `densenet.py`
  - **`models/heads/`**: `mlp_head.py`
  - **`training/`**: training loops (`baseline.py`), metrics
  - **`utils/`**: I/O helpers, seeding, etc.
- **`scripts/`**:
  - **`01_train_biomedclip_mlp.py`**: train an MLP head on top of BioMedCLIP (either end‑to‑end or using cached embeddings)
  - **`02_precompute_biomedclip_embeddings.py`**: precompute and shard BioMedCLIP embeddings from raw MIMIC‑CXR JPGs
  - **`00_make_splits.py`** (currently empty placeholder)
- **`data/`**:
  - **`mimic_paths.yaml`**: points to the root of the local MIMIC‑CXR JPG tree
  - optional split / report artifacts
- **`artifacts/`**:
  - **`checkpoints/`**: saved model checkpoints (e.g., `biomedclip_mlp`)
  - **`embeddings/`**: cached embeddings (e.g., `embeddings/biomedclip/{train,validate,test}`)
- **`slurm/`**:
  - **`jobs/`**: SLURM job scripts (e.g., `train_biomedclip_mlp.job`, `create_biomed_embeddings.job`)
  - **`logs/`**: SLURM stdout logs

---

### Requirements

Install with **conda** from the repo root, using `requirements.txt` and (optionally) an editable install.

**Option 1 – editable install with conda**

```bash
cd /path/to/VLM_graphrag

# create a new conda env (Python 3.10 is a safe choice for recent PyTorch)
conda create -n mimicvlm python=3.10 -y
conda activate mimicvlm

# install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# install the package in editable mode
pip install -e .
```

**Option 2 – scripts only with conda**

If you only care about running the provided scripts, you can rely on `PYTHONPATH` instead of installing the package:

```bash
cd /path/to/VLM_graphrag

conda create -n mimicvlm python=3.10 -y
conda activate mimicvlm

pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=$PWD/src:$PYTHONPATH
```

**Key dependencies (see `requirements.txt` for versions):**

- **PyTorch + torchvision + torchaudio**
- **HuggingFace stack**: `transformers`, `accelerate`, `datasets`, `huggingface_hub`, `sentencepiece`
- **BioMedCLIP** via `open_clip_torch`
- Standard numerics / data / viz: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- Utilities: `pyyaml`, `tqdm`, etc.

---

### Data: MIMIC‑CXR

This repo assumes you have access to the **MIMIC‑CXR** dataset in JPG form, laid out in the usual MIT PhysioNet structure, for example:

- `mimic_cxr_jpg_root/mimic-cxr-2.0.0-split.csv`
- `mimic_cxr_jpg_root/mimic-cxr-2.0.0-chexpert.csv`
- `mimic_cxr_jpg_root/files/p10/p10000032/s51000000/12345678.jpg`

Here, **`mimic_cxr_jpg_root` should be the path on your system to the local MIMIC‑CXR JPG dataset**.  
You can request and download MIMIC‑CXR from the official PhysioNet page:  
[`https://physionet.org/content/mimic-cxr/2.0.0/`](https://physionet.org/content/mimic-cxr/2.0.0/)

The default path can also be stored in `data/mimic_paths.yaml`, e.g.:

```yaml
mimic_cxr_jpg_root: /path/to/mimic_cxr_jpg_root
```

Update this file (or pass `--mimic_cxr_jpg_root` explicitly to scripts) to match your environment.

---

### Typical workflow

There are two main workflows:

- **(A) Train directly from images using BioMedCLIP as encoder**
- **(B) Precompute embeddings once, then train only the MLP head**

Both use the same dataset class `MimicCXRDataset` and 14‑label CheXpert targets.

---

### A. Precompute BioMedCLIP embeddings

You can precompute embeddings for each split (recommended for repeated experiments).

**CLI usage:**

```bash
conda activate mimicvlm
export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/02_precompute_biomedclip_embeddings.py \
  --mimic_cxr_jpg_root /path/to/mimic_cxr_jpg_root \
  --split train \
  --batch_size 128 \
  --num_workers 8 \
  --device cuda \
  --out_dir artifacts/embeddings/biomedclip \
  --shard_size 50000 \
  --amp \
  --save_meta
```

Repeat with `--split validate` and `--split test`.  
This will write `.pt` shard files containing:

- `z`: image embeddings (float16, shape `[N, D]`)
- `y`: multi‑label targets (float32, shape `[N, 14]`)
- optional `meta` information (if `--save_meta` is used)

You can then train only the MLP head via `EmbeddingShardDataset`.

---

### B. Train BioMedCLIP + MLP head

The main training script is `scripts/01_train_biomedclip_mlp.py`.

**Train directly from images (no cached embeddings):**

```bash
conda activate mimicvlm
export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/01_train_biomedclip_mlp.py \
  --mimic_cxr_jpg_root /path/to/mimic_cxr_jpg_root \
  --batch_size 64 \
  --num_workers 8 \
  --epochs 40 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 512 \
  --num_layers 2 \
  --dropout 0.1 \
  --device cuda \
  --save_dir artifacts/checkpoints/biomedclip_mlp
```

In this mode, `BiomedCLIP` is instantiated, frozen (encoder weights fixed), and used to generate embeddings on the fly.  
Training metrics (loss, macro AUROC, macro F1) are printed each epoch and logged to `metrics.csv`.  
Checkpoints are saved to `best.pt` and `last.pt` under `save_dir`.

**Train from cached embeddings (faster):**

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/01_train_biomedclip_mlp.py \
  --mimic_cxr_jpg_root /path/to/mimic_cxr_jpg_root \
  --use_cached_embeddings \
  --embeddings_dir artifacts/embeddings/biomedclip \
  --batch_size 256 \
  --epochs 40 \
  --lr 1e-3 \
  --device cuda \
  --save_dir artifacts/checkpoints/biomedclip_mlp_from_cached
```

Here, `EmbeddingShardDataset` loads `z` and `y` from disk, and only the MLP head is trained.

---

### SLURM usage (cluster)

SLURM job scripts live in `slurm/jobs/`. They are thin wrappers around the above scripts, configuring:

- working directory (`CWD`) to this repo
- `CUDA_VISIBLE_DEVICES`, resources, and logging paths under `slurm/logs/`

To run a job (example):

```bash
cd /path/to/VLM_graphrag
sbatch slurm/jobs/train_biomedclip_mlp.job
```

You can monitor training progress via the corresponding log file in `slurm/logs/train_head/` or `slurm/logs/biomedclip_mlp/` depending on the job.

---

### Reproducibility notes

- Seeds are controlled via `--seed` and `mimicvlm.utils.seed.set_seed`.
- DataLoader workers also use a seeded worker init function (`seed_worker`).
- Metrics include macro AUROC and macro F1 per epoch; see `training/metrics.py`.

---

### Citation

If you use this code in academic work, please also cite:

- **MIMIC‑CXR** dataset
- **BioMedCLIP** model

Add your own project‑specific citation here as needed.

