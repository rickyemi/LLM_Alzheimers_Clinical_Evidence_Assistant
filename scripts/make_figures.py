#!/usr/bin/env python
"""Generate every figure used in the README into docs/images/.

Schematics are drawn with matplotlib patches (no external tools). Data plots
(evaluation, score distribution, embedding map) are computed from the real
pipeline run, so the README always reflects the current code.

    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA

from ad_rag import build_pipeline, load_settings
from ad_rag.evaluation import load_golden

NAVY, TEAL, PURPLE, AMBER, RED, GRAY = "#0C447C", "#0F6E56", "#534AB7", "#BA7517", "#A32D2D", "#5F5E5A"
FILL = {NAVY: "#E6F1FB", TEAL: "#E1F5EE", PURPLE: "#EEEDFE", AMBER: "#FAEEDA", RED: "#FCEBEB", GRAY: "#F1EFE8"}
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 160, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, title, sub="", color=NAVY, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=FILL[color], ec=color, lw=1.1))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=fs - 2.3, color="#333333")


def arrow(ax, p1, p2, color=NAVY, style="-|>", ls="-", rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = canvas(16, 8.6, 170, 92)
    ax.text(2, 88, "System architecture: governed RAG with an agent layer", fontsize=14, fontweight="bold", color=NAVY)
    # lane 1
    ax.text(2, 79, "1. OFFLINE INDEXING (batch, versioned)", fontsize=10, fontweight="bold", color=TEAL)
    l1 = [("Curated corpus", "guidelines, policy, dossiers", GRAY), ("PHI gate", "regex + Presidio (opt.)", RED),
          ("Chunker", "heading + paragraph aware", PURPLE), ("Embedder", "hashing | sentence-transformers", PURPLE),
          ("Vector index", "FAISS + manifest + version", TEAL)]
    xs = [2, 37, 72, 107, 142]
    for (t, s, c), x in zip(l1, xs):
        box(ax, x, 62, 26, 12, t, s, c, 9)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 26.6, 68), (b - 0.6, 68))
    # lane 2
    ax.text(2, 53, "2. ONLINE REQUEST PATH (per request)", fontsize=10, fontweight="bold", color=TEAL)
    l2 = [("Client + API", "FastAPI, X-API-Key", NAVY), ("Input guard", "PHI redact, unsafe refuse", RED),
          ("Retriever", "top-k cosine search", PURPLE), ("Threshold gate", "refuse if weak evidence", RED),
          ("LLM", "Claude | extractive", PURPLE), ("Output guard", "citation validation", RED),
          ("Cited answer", "+ corpus version", TEAL)]
    xs2 = [2, 26, 50, 74, 98, 122, 146]
    for (t, s, c), x in zip(l2, xs2):
        box(ax, x, 36, 20, 12, t, s, c, 8.6)
    for a, b in zip(xs2[:-1], xs2[1:]):
        arrow(ax, (a + 20.6, 42), (b - 0.6, 42))
    arrow(ax, (155, 61.4), (60, 48.7), color=TEAL, ls="--", rad=-0.05)
    ax.text(100, 57.4, "index loaded read-only at startup", fontsize=8, color=TEAL, ha="center",
            bbox=dict(fc="white", ec="none", pad=1))
    # agent band
    box(ax, 50, 17.5, 92, 10, "Agent layer (bounded tool-use loop)", "plan, act, observe; 7 typed tools; human approval for writes", AMBER, 9.2)
    arrow(ax, (60, 28.2), (60, 35.4), color=AMBER, style="<|-|>")
    arrow(ax, (108, 28.2), (108, 35.4), color=AMBER, style="<|-|>")
    ax.text(2, 22, "Agentic AI plugs in here:\nthe retriever is one of its tools", fontsize=8.6, color=AMBER, va="center")
    # lane 3
    ax.text(2, 11.2, "3. GOVERNANCE AND OBSERVABILITY (cross-cutting)", fontsize=10, fontweight="bold", color=TEAL)
    l3 = [("Hash-chained audit log", GRAY), ("Prometheus metrics", GRAY), ("Drift monitor", GRAY),
          ("LangSmith traces (opt.)", GRAY), ("Golden-set release gate", GRAY), ("AI review board", GRAY)]
    for i, (t, c) in enumerate(l3):
        box(ax, 2 + i * 28, 1, 25, 7, t, "", c, 8.2)
    fig.savefig(OUT / "architecture.png"); plt.close(fig)


def fig_agent_loop():
    fig, ax = canvas(15, 8.4, 160, 90)
    ax.text(2, 86, "Agentic loop: the LLM plans, tools act, guardrails bound the autonomy", fontsize=14, fontweight="bold", color=NAVY)
    box(ax, 2, 60, 24, 13, "Task", "multi-step clinical question", NAVY, 9.5)
    box(ax, 33, 60, 24, 13, "Input guard", "PHI redact, refuse unsafe", RED, 9.5)
    arrow(ax, (26.6, 66.5), (32.4, 66.5)); arrow(ax, (57.6, 66.5), (66.4, 66.5))
    # loop
    ax.add_patch(FancyBboxPatch((67, 38), 52, 42, boxstyle="round,pad=0.3,rounding_size=2", fc="white", ec=AMBER, lw=1.4, ls="--"))
    ax.text(93, 77, "Bounded loop (max 6 steps)", ha="center", fontsize=9.5, fontweight="bold", color=AMBER)
    box(ax, 70, 56, 20, 11, "1. Plan", "LLM picks next tool", PURPLE, 9.2)
    box(ax, 96, 56, 20, 11, "2. Act", "validated tool call", AMBER, 9.2)
    box(ax, 83, 40.5, 20, 11, "3. Observe", "result back to LLM", TEAL, 9.2)
    arrow(ax, (90.6, 61.5), (95.4, 61.5), color=AMBER)
    arrow(ax, (106, 55.4), (100, 52.6), color=AMBER)
    arrow(ax, (82, 52.6), (77, 55.4), color=AMBER)
    # outputs
    box(ax, 128, 62, 28, 12, "Output guard", "citations must be valid", RED, 9.2)
    box(ax, 128, 44, 28, 11, "Cited answer", "structured + disclaimer", TEAL, 9.2)
    box(ax, 128, 26, 28, 11, "Escalate to human", "review queue ticket", NAVY, 9.2)
    arrow(ax, (119.6, 68), (127.4, 68)); arrow(ax, (142, 61.4), (142, 55.9))
    arrow(ax, (119.6, 50), (127.4, 33), color=NAVY, ls="--")
    ax.text(126, 28, "unsure, unsafe,\nor step cap hit", fontsize=8, color=NAVY, ha="right")
    # tool registry
    ax.text(2, 51.5, "TOOL REGISTRY (allowlist)", fontsize=9.5, fontweight="bold", color=PURPLE)
    tools = [("search_knowledge_base", "read", TEAL), ("interpret_biomarker", "read, deterministic", TEAL),
             ("check_appropriate_use", "read, rule based", TEAL), ("search_clinical_trials", "read, external API", TEAL),
             ("scan_for_phi", "read", TEAL), ("escalate_to_human", "safe write", NAVY),
             ("save_evidence_brief", "WRITE: approval", RED)]
    for i, (n, r, c) in enumerate(tools):
        y = 45 - i * 4.6
        ax.add_patch(FancyBboxPatch((2, y), 58, 4.1, boxstyle="round,pad=0.2,rounding_size=0.8", fc=FILL[c], ec=c, lw=0.9))
        ax.text(4, y + 2.05, n, fontsize=8.3, va="center", fontweight="bold", color=c)
        ax.text(58, y + 2.05, r, fontsize=7.8, va="center", ha="right", color="#333333")
    arrow(ax, (60.6, 38), (69.4, 56), color=PURPLE, ls="--", rad=0.15)
    # guardrails strip
    ax.add_patch(FancyBboxPatch((2, 1), 154, 12, boxstyle="round,pad=0.3,rounding_size=1.5", fc=FILL[RED], ec=RED, lw=1.1))
    ax.text(4, 10.3, "CONTROLS ON EVERY STEP", fontsize=9.3, fontweight="bold", color=RED)
    ax.text(4, 5, "tool allowlist  |  pydantic schema validation  |  per-tool call cap  |  step cap  |  human approval for write tools"
               "  |  hash-chained audit entry per call  |  deterministic math outside the LLM", fontsize=8.6, color="#333333")
    fig.savefig(OUT / "agent_loop.png"); plt.close(fig)


def fig_roadmap():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    phases = [("0. Foundations", 0, 1, GRAY, "Charter, data-use agreements, PHI SOPs, golden set"),
              ("1. MVP build", 1, 2, PURPLE, "RAG pipeline, guardrails, offline evaluation harness"),
              ("2. Supervised pilot", 2, 3, TEAL, "10-20 clinicians, human in the loop, bias audit"),
              ("3. Production", 3, 4, NAVY, "Containerized API, tracing, drift alerts, runbooks"),
              ("4a. Agentic L1-L2", 3.3, 5.0, AMBER, "Read-only tools, then approval-gated writes"),
              ("4b. Scale and extend", 4, 6, RED, "Multimodal, fine-tuning, regulated SaMD assessment")]
    for i, (name, s, e, c, desc) in enumerate(phases):
        y = len(phases) - i
        ax.barh(y, e - s, left=s, height=0.62, color=FILL[c], edgecolor=c, lw=1.3, zorder=2)
        ax.text(e + 0.08, y, desc, va="center", fontsize=8.8, color="#333333", zorder=3,
                bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_yticks([len(phases) - i for i in range(len(phases))])
    ax.set_yticklabels([p_[0] for p_ in phases], fontsize=10, fontweight="bold")
    gates = [(1, "Gate A: governance approved"), (2, "Gate B: faithfulness >= 95%"),
             (3, "Gate C: zero critical safety events"), (4, "Gate D: alerts verified")]
    for x, t in gates:
        ax.axvline(x, color="#999999", ls=":", lw=1, zorder=1)
        ax.text(x, 7.0, "G", ha="center", va="center", fontsize=8, color="white",
                bbox=dict(boxstyle="circle,pad=0.25", fc=GRAY, ec="none"))
    ax.set_xlim(0, 9.4); ax.set_ylim(0.3, 7.5)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Year 2 H1", "Year 2 H2"])
    [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]; ax.tick_params(axis="y", length=0)
    ax.set_title("AI roadmap with go/no-go gates (G = gate review)", loc="left", fontsize=13, fontweight="bold", color=NAVY)
    fig.savefig(OUT / "roadmap.png"); plt.close(fig)


def fig_autonomy():
    fig, ax = canvas(14, 5.4, 150, 58)
    ax.text(2, 54, "Agent autonomy ladder: earn each level with evidence", fontsize=13.5, fontweight="bold", color=NAVY)
    lv = [("L0  Grounded Q&A", "RAG answers one question\nwith citations", TEAL, "SHIPPED IN THIS REPO"),
          ("L1  Read-only agent", "Multi-step: search, calculate,\ncheck rules, look up trials", TEAL, "SHIPPED IN THIS REPO"),
          ("L2  Supervised actions", "Write tools (save brief, open ticket)\nonly with human approval", AMBER, "PILOT"),
          ("L3  Bounded autonomy", "Pre-approved low-risk actions\nrun unattended, audited", RED, "NOT PLANNED UNTIL\nEVIDENCE + REGULATORY REVIEW")]
    for i, (t, d, c, status) in enumerate(lv):
        x = 2 + i * 37
        h = 24 + i * 6
        ax.add_patch(FancyBboxPatch((x, 6), 34, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=FILL[c], ec=c, lw=1.3))
        ax.text(x + 17, 6 + h - 3.5, t, ha="center", fontsize=10, fontweight="bold", color=c)
        ax.text(x + 17, 6 + h - 11, d, ha="center", fontsize=8.6, color="#333333", va="center")
        ax.text(x + 17, 9.4, status, ha="center", fontsize=7.8, fontweight="bold", color=c, va="center")
    arrow(ax, (3, 3), (147, 3), color=GRAY)
    ax.text(75, 0.2, "increasing autonomy and risk, so increasing controls (approval gates, evaluation, regulatory scope)", ha="center", fontsize=8.6, color=GRAY)
    fig.savefig(OUT / "autonomy_ladder.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
def data_figures():
    s = load_settings()
    p = build_pipeline(s, rebuild=True, with_audit=False)
    golden = load_golden(s.root / "data/eval/golden_set.jsonl")
    res = json.loads((s.root / "reports" / "eval_results.json").read_text())

    # evaluation metrics vs gates
    gates = res["gates"]
    names = list(gates)
    vals = [gates[n]["value"] for n in names]
    thr = [gates[n]["threshold"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    ax.barh(y, vals, color=[TEAL if gates[n]["passed"] else RED for n in names], height=0.55)
    for yi, t, v in zip(y, thr, vals):
        ax.plot([t, t], [yi - 0.38, yi + 0.38], color="black", lw=2)
        ax.text(min(v, 1.0) - 0.02, yi, f"{v:.2f}", va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in names]); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_title("Release gates (black tick = threshold)", loc="left", fontsize=11, fontweight="bold", color=NAVY)
    [ax.spines[sp].set_visible(False) for sp in ("top", "right")]
    ax = axes[1]
    topics = list(res["by_topic"])
    kw = [res["by_topic"][t]["keyword_recall"] or 0 for t in topics]
    ax.barh(topics, kw, color=[TEAL if k >= 0.8 else AMBER for k in kw], height=0.55)
    ax.set_xlim(0, 1.05); ax.invert_yaxis()
    ax.set_title("Keyword recall by topic (extractive baseline)", loc="left", fontsize=11, fontweight="bold", color=NAVY)
    [ax.spines[sp].set_visible(False) for sp in ("top", "right")]
    fig.suptitle(f"Golden-set evaluation ({res['summary']['n_cases']} cases, embedder {res['embedder']}, LLM {res['llm']})", x=0.01, ha="left", fontsize=10, color=GRAY)
    fig.tight_layout(); fig.savefig(OUT / "eval_metrics.png"); plt.close(fig)

    # similarity-score distribution (threshold calibration)
    sc = {"in_scope": [], "out_of_scope": []}
    for g in golden:
        if g["category"] in sc:
            r = p.retrieve(g["question"], 1)
            sc[g["category"]].append(r[0].score if r else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bins = np.linspace(0, 0.6, 25)
    ax.hist(sc["in_scope"], bins=bins, alpha=0.85, color=TEAL, label=f"in-scope (n={len(sc['in_scope'])})")
    ax.hist(sc["out_of_scope"], bins=bins, alpha=0.85, color=RED, label=f"out-of-scope (n={len(sc['out_of_scope'])})")
    thr = s.retrieval.min_similarity
    ax.axvline(thr, color="black", ls="--"); ax.text(thr + 0.005, ax.get_ylim()[1] * 0.9, f"refusal threshold = {thr}", fontsize=9)
    ax.set_xlabel("best retrieval similarity (cosine)"); ax.set_ylabel("questions")
    ax.set_title("Threshold calibration: separation is real but thin", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    ax.legend(frameon=False); [ax.spines[sp].set_visible(False) for sp in ("top", "right")]
    fig.savefig(OUT / "score_distribution.png"); plt.close(fig)

    # embedding map
    X = p.store.vectors
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    docs = sorted({c.doc_id for c in p.store.chunks})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for i, d in enumerate(docs):
        idx = [j for j, c in enumerate(p.store.chunks) if c.doc_id == d]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=55, color=cmap(i % 10), label=d.replace("_", " "), alpha=0.9, edgecolor="white", lw=0.6)
    ax.set_title("Chunk embedding map (PCA of the index): chunks cluster by topic", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[sp].set_visible(False) for sp in ax.spines]
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
    fig.savefig(OUT / "embedding_map.png"); plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_agent_loop(); fig_roadmap(); fig_autonomy()
    if (Path(__file__).resolve().parents[1] / "reports" / "eval_results.json").exists():
        data_figures()
    print("figures written to", OUT)
