import torch, torch.nn as nn
from typing import Optional, Tuple
from open_clip import create_model_and_transforms, create_model_from_pretrained
import torch.nn.functional as F

class BiomedCLIP(nn.Module):
    """
    Biomed CLIP pretrained VIT model.
    """
    def __init__(self):
        super().__init__()
        
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.embed_dim = 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        
        return feats
