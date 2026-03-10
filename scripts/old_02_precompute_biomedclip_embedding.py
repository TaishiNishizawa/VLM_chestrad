import os, argparse, math, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_skip_none
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.training.baseline import freeze_encoder
from mimicvlm.utils.seed import set_seed, seed_worker
from mimicvlm.utils.io import ensure_dir, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "validate", "test"], required=True)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_dir", type=str, default="artifacts/embeddings/biomedclip")
    ap.add_argument("--shard_size", type=int, default=50000)  # number of samples per shard

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_meta", action="store_true")
    return ap.parse_args()

@torch.no_grad()
def main():
    print("cuda visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir) / args.split
    ensure_dir(out_root)
    write_json(vars(args), out_root / "precompute_args.json")

    encoder = BiomedCLIP().to(device)
    freeze_encoder(encoder)
    encoder.eval()
    torch.backends.cudnn.benchmark = True

    ds = MimicCXRDataset(
        mimic_cxr_jpg_root=args.mimic_cxr_jpg_root,
        split=args.split,
        transform=encoder.preprocess,
        label_policy="uncertain_as_negative",
        bad_image_log=str(out_root / f"bad_images_{args.split}.tsv"),
    )
    print(len(ds), "samples in split", args.split)
    

    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: keep deterministic order
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    total_n = len(ds)
    n_shards = math.ceil(total_n / args.shard_size)

    shard_idx = 0
    shard_count = 0

    z_buf = []
    y_buf = []
    meta_buf = [] if args.save_meta else None

    def flush_shard():
        nonlocal shard_idx, shard_count, z_buf, y_buf, meta_buf
        if shard_count == 0:
            return

        z = torch.cat(z_buf, dim=0)  # [M, D]
        y = torch.cat(y_buf, dim=0)  # [M, 14]

        # store on CPU, compressed by dtype
        z = z.to(torch.float16).cpu()
        y = y.to(torch.float32).cpu()

        payload = {"z": z, "y": y}
        if args.save_meta:
            payload["meta"] = meta_buf

        shard_path = out_root / f"shard_{shard_idx:03d}.pt"
        torch.save(payload, shard_path)

        z_buf = []
        y_buf = []
        if args.save_meta:
            meta_buf = []

        shard_idx += 1
        shard_count = 0

    t0 = time.perf_counter()
    pbar = tqdm(dl, desc=f"precompute[{args.split}]", total=len(dl))

    for images, targets, meta in pbar:
        if images is None or targets is None:
            continue

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                z = encoder(images)
        else:
            z = encoder(images)

        # IMPORTANT: move off GPU immediately
        z_buf.append(z.detach().cpu())
        y_buf.append(targets.detach().cpu())
        if args.save_meta:
            # meta is typically list[dict] per sample from your dataset
            meta_buf.extend(meta)

        shard_count += int(images.size(0))
        if shard_count >= args.shard_size:
            flush_shard()

    flush_shard()

    sec = time.perf_counter() - t0
    print(f"Done. Wrote {shard_idx} shard(s) to {out_root}. Time: {sec/60:.1f} min")

if __name__ == "__main__":
    main()