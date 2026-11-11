"""Input and output guardrails.

Why: the model is never the only safety control. Deterministic rules run BEFORE
retrieval (scope, safety, prompt-injection, PHI) and AFTER generation (citation
validity). Rules are cheap, testable and auditable, which is what a regulated
workflow needs. Regexes here are intentionally simple and conservative; extend
them from red-team findings (docs/governance.md, red-team log).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .phi import redact, scan_text

DISCLAIMER = ("Decision support for qualified clinicians. Not a diagnosis or treatment "
              "recommendation. Verify against primary sources.")

REFUSALS = {
    "individual_diagnosis": ("I can't diagnose or assess an individual person's condition. "
                             "Please discuss this with a qualified clinician. I can explain "
                             "published guidance on how biomarker tests are used."),
    "dosing": ("I can't provide individualized dosing or treatment instructions. Please consult "
               "the prescribing clinician or the approved product labeling."),
    "prompt_injection": ("I can't follow instructions that try to change my rules. "
                         "Ask a question about Alzheimer's evidence and I will help."),
    "personal_case": ("I can't interpret an individual's personal results. Please review them "
                      "with the treating clinician. I can explain how such tests are interpreted "
                      "in general."),
    "out_of_scope": ("I could not find relevant information in the knowledge base. "
                     "Please contact a human expert."),
}

_RULES = [
    ("prompt_injection", re.compile(
        r"ignore (all |any |your )?(previous|prior|above|earlier) (instructions|rules)|"
        r"reveal (your )?(system )?prompt|pretend (you are|to be)|you are now|disregard (the )?rules", re.I)),
    ("dosing", re.compile(
        r"\b(dose|dosage|dosing|how many (mg|milligrams)|how much .{0,30}(should|to) (take|give|inject))\b", re.I)),
    ("individual_diagnosis", re.compile(
        r"\b(diagnose|diagnosis for)\s+(me|him|her|them|my)\b|"
        r"\b(do|does|did|will|would)\s+(i|he|she|they|my\s+\w+)\s+(have|get|develop)\b.{0,40}\b(alzheimer|dementia)", re.I)),
    ("personal_case", re.compile(
        r"\bmy\s+(mother|father|mom|dad|wife|husband|spouse|parent|grandmother|grandfather|"
        r"brother|sister|result|results|test|lab|scan)\b", re.I)),
]


@dataclass
class InputGuard:
    allowed: bool
    category: Optional[str] = None
    message: Optional[str] = None
    redacted_question: str = ""
    phi_found: List[str] = field(default_factory=list)


def check_input(question: str) -> InputGuard:
    """Pre-retrieval checks. PHI is redacted, unsafe intents are refused."""
    phi = [f.kind for f in scan_text(question)]
    red = redact(question)
    for category, pat in _RULES:
        if pat.search(question):
            return InputGuard(False, category, REFUSALS[category], red, phi)
    return InputGuard(True, None, None, red, phi)


_CITE = re.compile(r"\[source:\s*([^\]]+)\]", re.I)


def extract_citations(answer: str) -> List[str]:
    """Return cited source names, e.g. '[source: appropriate_use_criteria.md]'."""
    return [c.strip() for m in _CITE.finditer(answer) for c in m.group(1).split(",")]


@dataclass
class OutputGuard:
    citations: List[str]
    citation_ok: bool
    unknown_citations: List[str]


def check_output(answer: str, retrieved_sources: List[str]) -> OutputGuard:
    """Every claim must cite, and every cited source must have been retrieved."""
    cites = extract_citations(answer)
    unknown = [c for c in cites if c not in set(retrieved_sources)]
    return OutputGuard(cites, bool(cites) and not unknown, unknown)
