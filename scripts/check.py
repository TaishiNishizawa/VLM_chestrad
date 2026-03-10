# scripts/find_bad_images.py
from mimicvlm.data.mimic_dataset import MimicCXRDataset
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from tqdm import tqdm

ds = MimicCXRDataset(
    mimic_cxr_jpg_root="/gs/gsfs0/shared-lab/duong-lab/MIMIC",
    split="train",
    transform=None,  # no transform, just test image loading
    bad_image_log="bad_images_train_full.tsv",
)

bad = []
for idx in tqdm(range(len(ds))):
    result = ds[idx]
    if result is None:
        bad.append(idx)

print(f"Found {len(bad)} bad indices: {bad}")