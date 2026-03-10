#!/usr/bin/env python3
"""
Sanity check: verify EmbeddingShardDataset is aligned with MimicCXRDataset by dicom_id.
Usage: python scripts/check_alignment.py --mimic_cxr_jpg_root ... --embedding_dir ... --split train
"""
import argparse
import os
import torch
from mimicvlm.data.mimic_dataset import MimicCXRDataset
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.training.baseline import freeze_encoder


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--embedding_dir",       type=str, required=True)
    ap.add_argument("--split",               type=str, default="train")
    ap.add_argument("--save_dir",            type=str, default="artifacts/check_alignment")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BiomedCLIP().to(device)
    freeze_encoder(encoder)

    img_ds = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=encoder.preprocess,
        label_policy="uncertain_as_negative",
        bad_image_log=os.path.join(args.save_dir, f"bad_images_{args.split}.tsv"),
    )
    emb_ds = EmbeddingShardDataset(os.path.join(args.embedding_dir, args.split))

    print(f"Image dataset size:     {len(img_ds)}")
    print(f"Embedding dataset size: {len(emb_ds)}")

    # ── Meta check ────────────────────────────────────────────────────────────
    if emb_ds.dicom_ids is None:
        print("⚠  Shards have no meta (old format) — falling back to size-only check.")
        assert len(img_ds) == len(emb_ds), "SIZE MISMATCH"
        print("✓ Sizes match\n")
        return

    # ── Key-based check ───────────────────────────────────────────────────────
    emb_dicoms = set(emb_ds.dicom_ids)
    img_dicoms = set(img_ds.dicom_ids)  # numpy array -> set of str

    only_in_img = img_dicoms - emb_dicoms
    only_in_emb = emb_dicoms - img_dicoms

    print(f"\nDicom IDs only in image dataset (skipped during caching): {len(only_in_img)}")
    for d in sorted(only_in_img):
        print(f"  {d}")

    print(f"Dicom IDs only in embeddings (should be 0):               {len(only_in_emb)}")
    for d in sorted(only_in_emb):
        print(f"  {d}")

    assert len(only_in_emb) == 0, "Embeddings contain dicom_ids not in the image dataset!"
    print("\n✓ All embedded dicom_ids are valid\n")

    # ── Spot-check label alignment on a few shared samples ───────────────────
    shared = sorted(emb_dicoms & img_dicoms)
    spot_dicoms = shared[:3] + [shared[len(shared) // 2]] + shared[-3:]

    # build dicom -> img_ds index (from the numpy array)
    img_dicom_to_idx = {d: i for i, d in enumerate(img_ds.dicom_ids)}

    print(f"{'Dicom ID':<40}  {'Img label sum':>13}  {'Emb label sum':>13}  {'Match':>6}")
    print("-" * 80)
    all_ok = True
    for dicom in spot_dicoms:
        img_idx = img_dicom_to_idx[dicom]
        emb_idx = emb_ds.dicom_to_idx[dicom]

        img_result = img_ds[img_idx]
        if img_result is None:
            print(f"{dicom:<40}  {'SKIP (bad img)':>13}")
            continue
        _, img_target, _ = img_result
        emb_z, emb_target = emb_ds[emb_idx]

        match = torch.allclose(img_target.float(), emb_target.float())
        if not match:
            all_ok = False
        print(f"{dicom:<40}  {img_target.sum().item():>13.0f}  {emb_target.sum().item():>13.0f}  {'✓' if match else '✗ MISMATCH':>6}")

    print()
    if all_ok:
        print("✓ All spot-checked labels match — safe to proceed.")
    else:
        print("✗ LABEL MISMATCHES DETECTED — do not proceed until resolved.")


if __name__ == "__main__":
    main()