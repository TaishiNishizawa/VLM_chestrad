from __future__ import annotations
import json
import re
import numpy as np
from mimicvlm.data.constants import CHEXPERT_LABELS_14


def parse_label_json(raw: str) -> np.ndarray | None:
  """
  Parse MedGemma's raw string output into a (14,) binary numpy array.
  Returns None if parsing fails completely.
  
  Handles:
  - Clean JSON
  - JSON wrapped in markdown code fences
  - Minor formatting issues
  """
  # Strip markdown code fences if present
  text = raw.strip()
  text = re.sub(r"^```(?:json)?\s*", "", text)
  text = re.sub(r"\s*```$", "", text)
  text = text.strip()

  # Extract first JSON object found
  match = re.search(r"\{.*\}", text, re.DOTALL)
  if not match:
      return None

  try:
      parsed = json.loads(match.group())
  except json.JSONDecodeError:
      return None

  # Build label vector in canonical order
  labels = np.zeros(len(CHEXPERT_LABELS_14), dtype=np.float32)
  for i, label in enumerate(CHEXPERT_LABELS_14):
      val = parsed.get(label, 0)
      try:
          labels[i] = float(val)
      except (TypeError, ValueError):
          labels[i] = 0.0

  return labels