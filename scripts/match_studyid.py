#!/usr/bin/env python3
"""
One-off: extract and save study_ids in shard order for a given split.
Run once per split (train, validate, test).
Usage: python scripts/save_study_ids.py --mimic_cxr_jpg_root ... --embedding_dir ... --split train
"""
import argparse
import numpy as np
from torch.utils.data import DataLoader
from mimicvlm.data.mimic_dataset import MimicCXRDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--embedding_dir",       type=str, required=True,
                    help="Root embedding dir — study_ids.npy saved to {embedding_dir}/{split}/")
    ap.add_argument("--split",               type=str, default="train")
    return ap.parse_args()

def main():
    args = parse_args()

    ds = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=None,
        label_policy="uncertain_as_negative",
        return_study_id=True,
    )

    print(f"Extracting {len(ds)} study_ids for split='{args.split}'...")

    # No image loading needed — just iterate study_ids directly
    # MimicCXRDataset with transform=None still loads images unfortunately,
    # so use a simple loop but only keep the study_id (index 2)
    study_ids = [ds[i][2] for i in range(len(ds))]

    out_path = f"{args.embedding_dir}/{args.split}/study_ids.npy"
    np.save(out_path, np.array(study_ids))
    print(f"✓ Saved {len(study_ids)} study_ids to {out_path}")
    print(f"  First 3: {study_ids[:3]}")
    print(f"  Last  3: {study_ids[-3:]}")

if __name__ == "__main__":
    main()