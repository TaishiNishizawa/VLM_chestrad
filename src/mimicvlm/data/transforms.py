# src/mimicvlm/data/transforms.py
from __future__ import annotations

from typing import Callable

import torch
from torchvision import transforms


def build_clip_image_transform(image_size: int = 224) -> Callable:
    # Note: CLIP/BioMedCLIP typically uses these mean/std.
    # If you later use the model's own processor, replace this.
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )