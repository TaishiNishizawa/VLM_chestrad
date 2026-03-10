#!/usr/bin/env python3
"""
One-off: extract and save (subject_id, study_id) pairs in shard order.
Run once for train split.
Usage: python scripts/save_study_ids.py --mimic_cxr_jpg_root ... --embedding_dir ...
"""
import argparse
import numpy as np
from mimicvlm.data.mimic_dataset import MimicCXRDataset
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--embedding_dir",       type=str, required=True,
                    help="Saves study_ids.npy to {embedding_dir}/train/")
    return ap.parse_args()

def main():
    args = parse_args()

    ds = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split="train",
        transform=None,
        label_policy="uncertain_as_negative",
    )

    print(f"Extracting IDs for {len(ds)} samples...")

    # Store as list of "subject_id/study_id" strings — simple, unambiguous
    pairs = []
    skipped = 0
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        if sample is None:
            pairs.append((-1, -1))   # sentinel for skipped
            skipped += 1
            continue
        _, _, meta = sample
        pairs.append((meta['subject_id'], meta['study_id']))

    out_path = f"{args.embedding_dir}/train/study_ids.npy"
    np.save(out_path, np.array(pairs, dtype=np.int64))  # shape (N, 2)
    print(f"✓ Saved {len(pairs)} entries ({skipped} skipped/None) to {out_path}")
    print(f"  First 3: {pairs[:3]}")
    print(f"  Last  3: {pairs[-3:]}")

if __name__ == "__main__":
    main()