"""LLM backends behind one `generate(system, user)` interface.

* ExtractiveLLM : offline, deterministic. Selects the most question-relevant
                  sentences from the retrieved passages and cites them. It cannot
                  hallucinate by construction, which makes it a useful baseline and
                  lets CI run the full pipeline without an API key.
* AnthropicLLM  : Claude via the Anthropic Messages API for production quality.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import LLMCfg
from .embeddings import analyze


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLM(Protocol):
    model: str

    def generate(self, system: str, user: str, max_tokens: Optional[int] = None) -> LLMResult: ...


_PASSAGE = re.compile(r"\[Passage (\d+) \| source: ([^|\]]+?) \| section: ([^\]]*)\]\n(.*?)(?=\n\n\[Passage |\n\nQUESTION:|\Z)",
                      re.S)


class ExtractiveLLM:
    model = "extractive-baseline"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, system: str, user: str, max_tokens: Optional[int] = None) -> LLMResult:
        q = re.search(r"QUESTION:\s*(.*)", user, re.S)
        question = q.group(1).strip() if q else user
        q_terms = {t for t in analyze(question) if "_" not in t}
        # 1) Collect candidate sentences from every retrieved passage.
        cands = []
        for m in _PASSAGE.finditer(user):
            rank, source, _section, body = int(m.group(1)), m.group(2).strip(), m.group(3), m.group(4)
            for sent in re.split(r"(?<=[.!?])\s+|\n[-*]\s+", body):
                sent = sent.strip(" -*\n")
                if len(sent) >= 25:
                    cands.append((rank, source, sent, {t for t in analyze(sent) if "_" not in t}))
        # 2) IDF over candidates: rare question terms (e.g. 'intermediate') outweigh
        #    ubiquitous ones (e.g. 'biomarker'), which sharpens sentence selection.
        n = max(len(cands), 1)
        df = {t: sum(t in c[3] for c in cands) for t in q_terms}
        w = {t: math.log(1 + n / (1 + df[t])) for t in q_terms}
        scored = []
        for rank, source, sent, terms in cands:
            overlap = q_terms & terms
            if overlap:
                score = sum(w[t] for t in overlap) / math.sqrt(len(terms) + 1) + 0.05 / rank
                scored.append((score, rank, source, sent))
        if not scored:
            text = ("The provided context does not contain the answer. "
                    "Please contact a human expert.")
        else:
            top = sorted(scored, reverse=True)[: self.max_sentences]
            top = sorted(top, key=lambda x: x[1])             # present in retrieval order
            text = "\n".join(f"- {s} [source: {src}]" for _, _, src, s in top)
        return LLMResult(text, len((system + user).split()), len(text.split()), self.model)


class AnthropicLLM:
    """Claude backend. `client` can be injected for tests."""

    def __init__(self, cfg: LLMCfg, client=None):
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        if client is None:
            import anthropic
            client = anthropic.Anthropic()      # reads ANTHROPIC_API_KEY
        self.client = client

    def generate(self, system: str, user: str, max_tokens: Optional[int] = None) -> LLMResult:
        kwargs = dict(model=self.model, max_tokens=max_tokens or self.max_tokens,
                      system=system, messages=[{"role": "user", "content": user}])
        if self.temperature is not None:        # omit when None: some models fix temperature
            kwargs["temperature"] = self.temperature
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResult(text, r.usage.input_tokens, r.usage.output_tokens, self.model)


def get_llm(cfg: LLMCfg) -> LLM:
    if cfg.backend == "anthropic":
        return AnthropicLLM(cfg)
    if cfg.backend == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM backend: {cfg.backend}")
