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
from mimicvlm.report_generation.prompt2 import (
    zeroshot_build_report_gen_messages, 
    build_report_gen_messages, 
    build_rag_report_gen_messages,
    build_graph_rag_report_gen_messages)
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.utils.io import to_device
from mimicvlm.graph.label_graph_retriever import LabelGraphRetriever

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


def run_textrag_report_generation(
    index: FAISSIndex, 
    report_store: ReportStore, 
    k: int,
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
    Run RAG report generation with BiomedCLIP label conditioning and retrieved reports as context.
    For each image, we retrieve BiomedCLIP label probabilities, convert them to a prompt, retrieve similar reports from FAISS, and generate a report with MedGemma.
    Saves a JSON mapping "subjectid_studyid_dicomid" → generated report text.
    """
    # Implementation would be similar to run_labels_report_gen, but with an additional step for FAISS retrieval and building the RAG prompt.
    n_missing_reports = 0
    generated_reports = {}
    nbatch = 0
    # First generate all prompts and retrieve similar reports for the entire dataset, storing them in a list.
    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="RAG inference",
    ):
        images, targets, image_keys = img_batch  
        embeddings, targets = emb_batch   
        embeddings = to_device(embeddings, device).float()
        
        with torch.no_grad():
            logits = head(embeddings)
            embeddings = embeddings.cpu()
            probs = torch.sigmoid(logits).cpu().numpy()
        
        messages_batch = []
        B = len(images)
        for i in range(B):
            # index.query returns list of k neighbor (subject_id, study_id, dicom_id)
            neighbor_ids = index.query(embeddings[i], k=k)
            reports = [r for (subject_id, study_id, dicom_id) in neighbor_ids if (r := report_store.get((subject_id, study_id, dicom_id)))]

            if len(reports) < k:
                n_missing_reports += 1

            prompt_text = logits_to_prompt_text(probs[i], threshold_labels, tuned_thresholds)
            
            message = build_rag_report_gen_messages(classifier_prompt_text=prompt_text, retrieved_reports=reports)
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


def run_text_and_graph_rag_report_generation(
    graph_retriever: LabelGraphRetriever,
    index: FAISSIndex, 
    report_store: ReportStore, 
    k_faiss: int,
    k_graph: int, 
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
    """
    generated_reports = {}
    nbatch = 0
    # First generate all prompts and retrieve similar reports for the entire dataset, storing them in a list.
    for img_batch, emb_batch in tqdm(
        zip(image_dataloader, embedding_dataloader),
        total=len(image_dataloader),
        desc="RAG inference",
    ):
        images, targets, image_keys = img_batch  
        embeddings, targets = emb_batch   
        embeddings = to_device(embeddings, device).float()
        
        # Step 1: Get predicted labels using the MLP head on BiomedCLIP embeddings
        with torch.no_grad():
            logits = head(embeddings)
            embeddings = embeddings.cpu()
            probs = torch.sigmoid(logits).cpu().numpy()
            predicted_labels = (probs >= tuned_thresholds).astype(int)
        
        # Step 2: For each image in the batch, retrieve similar reports from FAISS and graph-rag using the predicted labels.
        messages_batch = []
        B = len(images)
        for i in range(B):
            # FAISS RETRIEVAL 
            # index.query returns list of k neighbor (subject_id, study_id, dicom_id)
            neighbor_ids = index.query(embeddings[i], k=k_faiss)
            faiss_reports = [r for (subject_id, study_id, dicom_id) in neighbor_ids if (r := report_store.get((subject_id, study_id, dicom_id)))]

            # GRAPH RETRIEVAL 
            graph_reports = graph_retriever.retrieve(predicted_labels[i], k=k_graph)
            prompt_text = logits_to_prompt_text(probs[i], threshold_labels, tuned_thresholds)

            message = build_graph_rag_report_gen_messages(classifier_prompt_text=prompt_text, faiss_reports=faiss_reports, graph_reports=graph_reports)
            messages_batch.append(message)
        
        raw_outputs = model.generate(
            images=images,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
        )

        for i, raw in enumerate(raw_outputs):
            key = "_".join(str(imk) for imk in image_keys[i]) 
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

