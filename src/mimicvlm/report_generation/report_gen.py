from __future__ import annotations
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from mimicvlm.models.encoders.medgemma import MedGemma
from mimicvlm.inference.json_parser import parse_label_json
from mimicvlm.inference.prompt import (
    build_messages,
    logits_to_prompt_text,
    build_biomedclip_messages,
)
from mimicvlm.report_generation.prompt2 import build_report_gen_messages
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.utils.io import to_device

def run_zeroshot_report_gen(
    model: MedGemma,
    dataloader: DataLoader,
    output_path: str,         
    max_new_tokens: int = 512,
) -> None:
    generated_reports = {}

    for nbatch, batch in enumerate(tqdm(dataloader, desc="Generating reports")):
        # images: list[PIL.Image]
        # targets: (B, 14) tensor
        # image_keys: list[tuple[int, int, str]]  — (subject_id, study_id, dicom_id)
        images, targets, image_keys = batch
        
        messages_batch = [zeroshot_build_report_gen_messages() for _ in images]

        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )

        for i, raw in enumerate(raw_outputs):
            key = "_".join(str(k) for k in image_keys[i]) 
            generated_reports[key] = raw

        # Save incrementally every 100 batches — don't lose progress
        if nbatch % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(generated_reports, f, indent=2)
            print(f"Saved {len(generated_reports)} reports so far...")

    # Final save
    with open(output_path, "w") as f:
        json.dump(generated_reports, f, indent=2)
    print(f"Done. Saved {len(generated_reports)} reports to {output_path}")

def run_labels_report_gen(
    head: MLPHead,
    model: MedGemma,
    device: torch.device,
    image_dataloader: DataLoader,
    embedding_dataloader: DataLoader,
    tuned_thresholds: np.ndarray,
    threshold_labels: list[str],
    output_path: str,
    max_new_tokens: int = 512,
) -> None:
    """
    Run zero-shot report generation with BiomedCLIP label conditioning.
    For each image, we retrieve BiomedCLIP label probabilities, convert them to a prompt, and generate a report with MedGemma.
    Saves a JSON mapping "subjectid_studyid_dicomid" → generated report text.
    """
    generated_reports = {}
    nbatch = 0
    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="Running zero-shot inference with BioMedCLIP embeddings",
    ):
        images, targets, image_keys = img_batch  
        embeddings, targets = emb_batch   
        embeddings = to_device(embeddings, device).float()

        with torch.no_grad():
            logits = head(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()

        batch_size = len(images)
        messages_batch = []
        for i in range(batch_size):            
            prompt_text = logits_to_prompt_text(probs[i], threshold_labels, tuned_thresholds)
            message = build_report_gen_messages(classifier_prompt_text=prompt_text)
            messages_batch.append(message)
        
        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )

        for i, raw in enumerate(raw_outputs):
            key = "_".join(str(k) for k in image_keys[i]) 
            generated_reports[key] = raw

        # Save incrementally every 100 batches — don't lose progress
        if nbatch % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(generated_reports, f, indent=2)
            print(f"Saved {len(generated_reports)} reports so far...")
        nbatch += 1

    # Final save
    with open(output_path, "w") as f:
        json.dump(generated_reports, f, indent=2)
    print(f"Done. Saved {len(generated_reports)} reports to {output_path}")
