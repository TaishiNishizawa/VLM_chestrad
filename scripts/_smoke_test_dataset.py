import torch
from torch.utils.data import DataLoader
import os
from mimicvlm.data.mimic_dataset import MimicCXRDataset
from mimicvlm.data.transforms import build_clip_image_transform
from mimicvlm.utils.seed import set_seed, seed_worker
# from data.mimic_paths.yaml import mimic_cxr_jpg_root

def main():
    set_seed(0)
    mimic_cxr_jpg_root = "/gs/gsfs0/shared-lab/duong-lab/MIMIC"
    val_ds = MimicCXRDataset(
        split_csv= os.path.join(mimic_cxr_jpg_root, "mimic-cxr-2.0.0-split.csv"),
        image_root = os.path.join(mimic_cxr_jpg_root, "files"), 
        label_csv= os.path.join(mimic_cxr_jpg_root, "mimic-cxr-2.0.0-chexpert.csv"),
        split="train",
        transform=build_clip_image_transform(224),
        label_policy="uncertain_as_negative",
    )
    print(f"Size of dataset: ", len(val_ds))
    g = torch.Generator()
    g.manual_seed(0)

    dl = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    images, labels, meta = next(iter(dl))
    print("images:", images.shape, images.dtype, images.min().item(), images.max().item())
    print("labels:", labels.shape, labels.dtype)
    print("label positives per class:", labels.sum(dim=0))
    print("example meta:", {k: meta[k][0] for k in meta})



if __name__ == "__main__":
    main()    