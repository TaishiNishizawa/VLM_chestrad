from __future__ import annotations
from mimicvlm.data.constants import CHEXPERT_LABELS_14

_LABEL_LIST = "\n".join(f"  - {label}" for label in CHEXPERT_LABELS_14)

ZEROSHOT_SYSTEM = (
    "You are a radiologist assistant. "
    "You will be shown a chest X-ray image and must classify it. "
    "Respond only with valid JSON. Do not add explanation or markdown."
)

ZEROSHOT_USER_TEMPLATE = (
    "You are a radiology AI assistant. Analyze THIS SPECIFIC chest X-ray carefully."
    "For each finding output 1 if present, 0 if absent. Only mark a finding as 1 if there is clear evidence of it in the image. When in doubt, output 0. Base your answer only on what you observe in this image, not on general prevalence.\n\n"
    "Findings:\n"
    f"{_LABEL_LIST}\n\n"
    "Respond with a single JSON object using exactly these keys, "
    "with integer values 0 or 1. Example format:\n"
    '{"Atelectasis": 0, "Cardiomegaly": 1, ...}\n\n'
    "Your response:"
)

_BASE_INSTRUCTION = f"""\
You are a radiology AI assistant. Analyze THIS SPECIFIC chest X-ray carefully.

For each finding below, output 1 only if you can clearly identify it in this image, \
or 0 if it is absent or not clearly visible. Base your answer only on what you \
observe in this image, not on general prevalence.

Findings to classify:
{_LABEL_LIST}

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
Example format:
{{
  "Atelectasis": 0,
  "Cardiomegaly": 1,
  "Consolidation": 0,
  "Edema": 0,
  "Enlarged Cardiomediastinum": 0,
  "Fracture": 0,
  "Lung Lesion": 0,
  "Lung Opacity": 0,
  "No Finding": 0,
  "Pleural Effusion": 1,
  "Pleural Other": 0,
  "Pneumonia": 0,
  "Pneumothorax": 0,
  "Support Devices": 1
}}"""

def build_messages(
    system: str = ZEROSHOT_SYSTEM,
    user: str = _BASE_INSTRUCTION,
) -> list[dict]:
    """Zero-shot: image + instruction only."""
    return [
          {"role": "system", "content": system},
          {"role": "user", "content": [
              {"type": "image"},  
              {"type": "text", "text": user},
          ]},
    ]

def build_rag_messages(retrieved_reports: list[str]) -> list[dict]:
    """
    RAG: prepend retrieved reports as context before the image + instruction.
    retrieved_reports: list of k report strings (FINDINGS/IMPRESSION sections)
    """
    context_parts = []
    for i, report in enumerate(retrieved_reports, 1):
        context_parts.append(f"Reference report {i}:\n{report}")
    context_text = "\n\n---\n\n".join(context_parts)

    rag_preamble = (
        "The following are radiology reports from similar chest X-ray cases. "
        "Use them as reference context to inform your analysis of the new image below.\n\n"
        f"{context_text}\n\n---\n\n"
        "Now analyze the following chest X-ray:"
    )

    return [
        {
          "role": "system", 
          "content": system},
        {
          "role": "user",
          "content": [
              {"type": "text", "text": rag_preamble},
              {"type": "image"},
              {"type": "text", "text": _BASE_INSTRUCTION},
          ],
        }
    ]