from __future__ import annotations
import re
from pathlib import Path


class ReportStore:
    """
    Lazy loader for MIMIC-CXR free-text radiology reports.

    MIMIC-CXR report structure:
        {mimic_cxr_root}/files/p{patient_prefix}/p{patient_id}/s{study_id}.txt

    study_id format in MIMIC-CXR is numeric, e.g. "50414267" → s50414267.txt
    """

    def __init__(self, mimic_cxr_root: str | Path):
        self.root = Path(mimic_cxr_root)
        self._cache: dict[tuple[int, int, str], str] = {}

    def get(self, image_key: tuple[int, int, str]) -> str | None:
        if image_key == (-1, -1, -1):
            return None
        if image_key in self._cache:
            return self._cache[image_key]

        subject_id, study_id, dicom_id = image_key
        path = self.root / "files" / f"p{str(subject_id)[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        text = self._extract_sections(text)
        self._cache[image_key] = text
        return text

    def _extract_sections(self, text: str) -> str:
        """
        Pull out FINDINGS and IMPRESSION sections only.
        Falls back to full text if neither section is found.
        """
        sections = []
        for section in ("FINDINGS", "IMPRESSION"):
            pattern = rf"{section}[:\s]*(.*?)(?=\n[A-Z\s]{{4,}}:|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    sections.append(f"{section}:\n{content}")

        return "\n\n".join(sections) if sections else text