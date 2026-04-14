# src/mimicvlm/prompts/prompt2.py
from __future__ import annotations
import numpy as np
import torch
from mimicvlm.data.constants import CHEXPERT_LABELS_14

# ── System prompt ──────────────────────────────────────────────────────────────
ZEROSHOT_REPORT_GEN_SYSTEM = (
    "You are an expert radiologist. "
    "You will be given a chest X-ray image"
    "Your task is to generate a structured radiology report for the given image. "
    "Write in formal radiology language. Be specific and concise. "
    "Do not fabricate findings. If a finding is absent, do not mention it."
)

REPORT_GEN_SYSTEM = (
    "You are an expert radiologist. "
    "You will be given a chest X-ray image, automated classifier predictions, "
    "and optionally reference reports from similar cases. "
    "Your task is to generate a structured radiology report for the given image. "
    "Write in formal radiology language. Be specific and concise. "
    "Do not fabricate findings. If a finding is absent, do not mention it."
)

# ── Classifier predictions → natural language ──────────────────────────────────

def logits_to_report_context(
    probs: torch.Tensor,
    label_names: list[str],
    tuned_thresholds: np.ndarray,
) -> str:
    """Convert sigmoid probs + tuned thresholds to natural language for report gen."""
    assert len(label_names) == len(probs) == len(tuned_thresholds)

    present, uncertain, absent = [], [], []
    for name, prob, thresh in zip(label_names, probs, tuned_thresholds):
        prob_val = prob.item() if isinstance(prob, torch.Tensor) else float(prob)
        if prob_val >= thresh:
            present.append(f"{name} ({prob_val:.2f})")
        elif prob_val >= thresh * 0.6:
            uncertain.append(f"{name} ({prob_val:.2f})")
        else:
            absent.append(name)

    fmt = lambda items: ", ".join(items) if items else "None"

    return (
        "Automated classifier findings for this chest X-ray:\n"
        f"  LIKELY PRESENT:     {fmt(present)}\n"
        f"  UNCERTAIN/POSSIBLE: {fmt(uncertain)}\n"
        f"  LIKELY ABSENT:      {fmt(absent)}\n\n"
        "Use these predictions to inform your report, but base your final "
        "assessment on what you observe in the image."
    )

# ── Report generation instruction ─────────────────────────────────────────────

REPORT_INSTRUCTION = """\
Generate a radiology report for this chest X-ray. 
Your report should follow this structure:

FINDINGS: [Describe observations for each relevant anatomical region: lungs, \
pleura, heart, mediastinum, bones, and any support devices. \
Only describe what is visible. Be specific about location, severity, and laterality.]

IMPRESSION: [Provide a concise summary of the key findings and their clinical \
significance in 1-3 sentences.]

Write in formal radiology language. Do not mention findings that are not \
supported by the image. Do not use bullet points."""

# ── Message builders ───────────────────────────────────────────────────────────
def zeroshot_build_report_gen_messages(
    system: str = ZEROSHOT_REPORT_GEN_SYSTEM,
    instruction: str = REPORT_INSTRUCTION,
) -> list[dict]:
    """
    Baseline zero-shot report generation: image + standard instruction prompt only.
    No Classifier, no RAG, no KG context.
    """
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]

def build_report_gen_messages(
    classifier_prompt_text: str,
    system: str = REPORT_GEN_SYSTEM,
    instruction: str = REPORT_INSTRUCTION,
) -> list[dict]:
    """
    Baseline report generation: image + classifier predictions only.
    No RAG, no KG context.
    """
    preamble = (
        f"{classifier_prompt_text}\n\n"
        "---\n\n"
        "Now generate a report for the following chest X-ray:"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": preamble},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_rag_report_gen_messages(
    classifier_prompt_text: str,
    retrieved_reports: list[str],
    system: str = REPORT_GEN_SYSTEM,
    instruction: str = REPORT_INSTRUCTION,
) -> list[dict]:
    """
    RAG report generation: image + classifier predictions + retrieved reports.
    retrieved_reports: list of k report strings from FAISS retrieval.
    """
    context_parts = []
    for i, report in enumerate(retrieved_reports, 1):
        context_parts.append(f"Reference report {i}:\n{report}")
    context_text = "\n\n---\n\n".join(context_parts)

    rag_preamble = (
        f"{classifier_prompt_text}\n\n"
        "---\n\n"
        "The following are radiology reports from similar chest X-ray cases. "
        "Use them as stylistic and clinical reference — do not copy them directly.\n\n"
        f"{context_text}\n\n"
        "---\n\n"
        "Now generate a report for the following chest X-ray:"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rag_preamble},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_graph_rag_report_gen_messages(
    classifier_prompt_text: str,
    faiss_reports: list[str],
    graph_reports: list[str],
    system: str = REPORT_GEN_SYSTEM,
    instruction: str = REPORT_INSTRUCTION,
) -> list[dict]:
    """
    GraphRAG report generation: image + classifier predictions +
    FAISS-retrieved reports (visual similarity) +
    graph-retrieved reports (concept-driven, deduplicated against FAISS).
    """
    # Deduplicate graph reports against FAISS reports, preserving order
    seen = set(faiss_reports)
    unique_graph_reports = [r for r in graph_reports if r not in seen]

    faiss_parts = [f"Reference report {i}:\n{r}" for i, r in enumerate(faiss_reports, 1)]
    graph_parts = [f"Reference report {len(faiss_reports) + i}:\n{r}" for i, r in enumerate(unique_graph_reports, 1)]

    preamble_parts = [classifier_prompt_text, "---"]

    if faiss_parts:
        preamble_parts += [
            "The following reports are from visually similar chest X-ray cases (retrieved by image embedding similarity):",
            "\n\n---\n\n".join(faiss_parts),
            "---",
        ]

    if graph_parts:
        preamble_parts += [
            "The following reports are from clinically related cases (retrieved by label co-occurrence graph):",
            "\n\n---\n\n".join(graph_parts),
            "---",
        ]

    preamble_parts.append("Now generate a report for the following chest X-ray:")
    preamble = "\n\n".join(preamble_parts)

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": preamble},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
