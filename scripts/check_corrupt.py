from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

import os
from tqdm import tqdm

root = "/gs/gsfs0/shared-lab/duong-lab/MIMIC/files"
corrupt = []

print(f"Walking under: {root}", flush=True)

pbar = tqdm(desc="Scanning JPGs", unit="img", dynamic_ncols=True)
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        if not name.lower().endswith(".jpg"):
            continue
        f = os.path.join(dirpath, name)
        pbar.update(1)
        try:
            with Image.open(f) as img:
                img.convert("RGB")
        except Exception as e:
            corrupt.append((f, str(e)))

pbar.close()

print(f"\nCorrupt: {len(corrupt)}", flush=True)
for f, e in corrupt:
    print(f"  {f}: {e}")