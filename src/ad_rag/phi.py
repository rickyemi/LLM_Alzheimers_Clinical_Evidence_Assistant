"""PHI screening: detect and redact protected health information.

Why: nothing patient-identifiable may enter the vector index, prompts, logs or
traces. This is a defence-in-depth GATE, not a guarantee. Regex rules catch the
structured identifiers (SSN, phone, email, MRN, DOB, labelled names). For free
text in production, add Microsoft Presidio (NER based) on top; this module
exposes the same interface so the swap is local.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

PATTERNS: Dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "MRN": re.compile(r"\b(?:MRN|Medical Record(?: Number)?)\s*[:#]?\s*\d{5,}\b", re.I),
    "DOB": re.compile(r"\b(?:DOB|Date of Birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I),
    "PATIENT_NAME": re.compile(r"\b(?:Patient|Pt)(?:\s+Name)?\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"),
}


@dataclass
class PHIFinding:
    kind: str
    start: int
    end: int
    snippet: str


def scan_text(text: str) -> List[PHIFinding]:
    """Return every PHI-like span found in `text`."""
    findings: List[PHIFinding] = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            findings.append(PHIFinding(kind, m.start(), m.end(), m.group(0)))
    return sorted(findings, key=lambda f: f.start)


def redact(text: str) -> str:
    """Replace PHI-like spans with typed placeholders such as [REDACTED:SSN]."""
    out = text
    # Replace from the end so earlier offsets stay valid.
    for f in sorted(scan_text(text), key=lambda f: f.start, reverse=True):
        out = out[: f.start] + f"[REDACTED:{f.kind}]" + out[f.end:]
    return out


def has_phi(text: str) -> bool:
    return bool(scan_text(text))
