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
observe in this image, not on general prevalence. \
Most findings are ABSENT in any given X-ray. Only mark a finding as 1 if there 
is clear, unambiguous visual evidence in this image. If you are uncertain, output 0.

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

RAG_SYSTEM = (
    "You are a radiologist assistant. "
    "You will be given radiology reports from similar cases as reference context, "
    "followed by a new chest X-ray image to classify. "
    "Use the reference reports to inform your judgment, but base your final classification "
    "on what you observe in the new image. "
    "Respond only with valid JSON. Do not add explanation or markdown."
)

def build_rag_messages(
    retrieved_reports: list[str],
    system: str = RAG_SYSTEM,         
    user: str = _BASE_INSTRUCTION,
) -> list[dict]:
    """
    RAG: prepend retrieved reports as context before the image + instruction.
    retrieved_reports: list of k report strings (FINDINGS/IMPRESSION sections)
    """
    context_parts = []
    for i, report in enumerate(retrieved_reports, 1):
        print(f"\n[DEBUG] Retrieved report {i}:\n{report}\n")
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
              {"type": "text", "text": user},
          ],
        }
    ]


def logits_to_prompt_text(
    probs: torch.Tensor,
    label_names: list[str],
    tuned_thresholds: np.ndarray
) -> str:
    lines_present = []
    lines_uncertain = []
    lines_absent = []

    assert len(label_names) == len(probs) == len(tuned_thresholds), \
        "Length of label_names, probs, and tuned_thresholds must match"

    for name, prob, thresh in zip(label_names, probs, tuned_thresholds):
        uncertain_low = thresh * 0.6
        if prob >= thresh:
            lines_present.append(name)
        elif prob >= uncertain_low:
            lines_uncertain.append(name)
        else:
            lines_absent.append(name)

    def fmt(items):
        return ", ".join(items) if items else "None"

    prompt = (
        "A BiomedCLIP-based classifier pre-screened this chest X-ray and predicted the following findings:\n"
        f"LIKELY PRESENT: {fmt(lines_present)}\n"
        f"UNCERTAIN / POSSIBLE: {fmt(lines_uncertain)}\n"
        f"LIKELY ABSENT: {fmt(lines_absent)}\n"
        "Use these predictions as a reference as you make your own analysis of the image."
    )
    return prompt

BIOMEDCLIP_SYSTEM = (
    "You are a radiologist assistant. Analyze THIS SPECIFIC chest X-ray carefully."
    "You will be given predictions from a BiomedCLIP-based classifier as a reference, "
    "followed by a chest X-ray image to classify. "
    "Use the classifier predictions to inform your judgment, but base your final classification "
    "on what you observe in the image. "
    "Respond only with valid JSON. Do not add explanation or markdown."
)

def build_biomedclip_messages(
    classifier_prompt_text: str,
    system: str = BIOMEDCLIP_SYSTEM,
    user: str = _BASE_INSTRUCTION,
) -> list[dict]:
    """
    BiomedCLIP-guided: prepend MLP classifier predictions before the image + instruction.
    classifier_prompt_text: output of logits_to_prompt_text()
    """
    preamble = (
        f"{classifier_prompt_text}\n\n---\n\n"
        "Now analyze the following chest X-ray:"
    )

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": preamble},
                {"type": "image"},
                {"type": "text", "text": user},
            ],
        }
    ]