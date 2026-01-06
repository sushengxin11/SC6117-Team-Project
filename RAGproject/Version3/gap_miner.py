# gap_miner.py
#
# Functionality:
# 1. Retrieve a batch of papers from arXiv
# 2. Perform structured analysis on each paper using paper_analyzer
# 3. Aggregate the limitations and possible_future_work from all papers
# 4. Call LLM again to generate a set of "technically detailed" research gaps
# This is the direct input for the subsequent "idea generator" and "idea evaluator".

import json
import re
import fitz
from typing import List, Dict, Any, Tuple

from paper_retriever import fetch_arxiv_papers
from llm_client import call_llm_raw
from pdf_utils import download_pdf, _find_sections_in_text
from paper_analyzer import analyze_paper, analyze_paper_with_fulltext, clean_and_parse_json

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# -------------------------------------------------------------------------
# Debias controls
# -------------------------------------------------------------------------
# These are "default-template" gap patterns that frequently appear even when not well-supported.
# We will (1) discourage them in prompts and (2) trigger a second-pass refinement if they dominate.
_GENERIC_GAP_PATTERNS = [
    r"lack(s)?\s+(a\s+)?(clear\s+)?mathematical\s+(definition|formalism|formulation)",
    r"missing\s+mathematical\s+(definition|formalism|formulation)",
    r"lack(s)?\s+theoretical\s+(guarantee|analysis|understanding|foundation)s?",
    r"insufficient\s+ablation(s)?",
    r"lack(s)?\s+(of\s+)?ablation(s)?",
    r"missing\s+ablation(s)?",
    r"poor\s+robustness",
    r"robustness\s+(is\s+)?(weak|poor|limited)",
    r"out[\-\s]?of[\-\s]?distribution\s+(generalization|performance)\s+(is\s+)?(weak|poor|limited)",
]

_GENERIC_RE = re.compile("|".join(_GENERIC_GAP_PATTERNS), re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _count_generic_gaps(gaps: List[Dict[str, Any]]) -> int:
    c = 0
    for g in gaps:
        text = " ".join([
            g.get("description", ""),
            g.get("why_important", ""),
            " ".join(g.get("suggested_directions", []) or []),
        ])
        if _GENERIC_RE.search(text):
            c += 1
    return c


def _has_minimum_technical_specificity(gap: Dict[str, Any]) -> bool:
    """
    Very lightweight heuristic: require at least two "technical" signals in description.
    This prevents pure meta-gaps like "need more theory/ablations" from passing easily.
    """
    desc = _normalize(gap.get("description", ""))
    # A small list of technical cue words that tend to appear in real mechanism-level gaps.
    cues = [
        "attention", "transformer", "diffusion", "score", "rerank", "retrieval", "embedding",
        "graph", "gcn", "gnn", "lora", "prefix", "prompt", "kv cache", "speculative",
        "latency", "throughput", "memory", "context", "chunk", "index", "qdrant",
        "loss", "objective", "gradient", "sampling", "window", "state space", "ssm",
    ]
    hit = 0
    for w in cues:
        if w in desc:
            hit += 1
            if hit >= 2:
                return True
    # If no cue match, still allow if description contains enough concrete nouns (crudely by length)
    return len(desc) >= 180  # long enough descriptions tend to include details


# -------------------------------------------------------------------------
# Prompt: revised to remove default-template bias
# -------------------------------------------------------------------------
SYSTEM_PROMPT_GAP_MINER = """
You are an expert machine learning meta-researcher.
You identify research gaps ONLY from the provided evidence.

Your job:
1) Aggregate limitations/future-work into 3–7 research gaps.
2) Make each gap technically specific (mechanisms, architectures, training regimes, evaluation setups).
3) Output STRICT JSON only.

NON-NEGOTIABLE EVIDENCE RULES:
- Do NOT introduce generic "default" gaps (e.g., "missing mathematical definition", "need more ablations", "poor robustness")
  unless they are explicitly supported by evidence from AT LEAST TWO different papers in the provided input.
- Evidence must be specific and traceable to the provided limitations/future-work items; do not invent.
- Each gap MUST cite 2–4 related_papers, and each evidence_from_paper must paraphrase ONE concrete limitation/future-work item.

ANTI-TEMPLATE RULE (IMPORTANT):
- Avoid meta-gaps that could apply to any ML topic.
- Prefer mechanism-level gaps (e.g., scaling bottlenecks, architectural constraints, failure modes, optimization instability, mismatch between training objective and deployment setting, retrieval/indexing limitations, etc.).
- If you include a Theory/Understanding or Evaluation/Methodology gap, it must mention the exact object being formalized/evaluated
  (e.g., "sliding-window attention cache eviction policy", "prefix-tuning parameterization under frozen backbones", "GraphRAG subgraph selection heuristics"),
  not generic "needs theory" or "needs ablations".

OUTPUT STYLE:
- JSON only.
- Gaps should be actionable and grounded in the provided papers.
"""


def build_gap_mining_prompt(papers_with_analysis: List[Dict[str, Any]]) -> str:
    """
    Construct prompt by packaging limitations/future-work of multiple papers.
    """
    input_data_json = json.dumps(papers_with_analysis, indent=2, ensure_ascii=False)

    prompt = f"""
You are given analysis of multiple papers in JSON format.
Each item has: paper_index, title, limitations, possible_future_work.

Here is the JSON array of papers (DO NOT repeat this array back in your output):
{input_data_json}

Output a STRICT JSON object with the following structure:

{{
  "research_gaps": [
    {{
      "gap_id": "G1",
      "category": "Data/Benchmarks | Model/Architecture | Training/Optimization | Evaluation/Methodology | Robustness/Generalization | Theory/Understanding",
      "description": "2-4 sentences, technically specific. Must mention concrete mechanisms / regimes / constraints.",
      "why_important": "2-4 sentences; explain the impact and where/when this limitation manifests.",
      "related_papers": [
        {{
          "paper_index": 0,
          "title": "Exact paper title here",
          "evidence_from_paper": "Paraphrase ONE concrete limitation/future-work item from the provided input for this paper. No generic claims."
        }},
        {{
          "paper_index": 1,
          "title": "Exact paper title here",
          "evidence_from_paper": "Paraphrase ONE concrete limitation/future-work item from the provided input for this paper."
        }}
      ],
      "suggested_directions": [
        "Concrete, technically detailed research direction addressing the gap.",
        "Another direction, also concrete."
      ]
    }}
  ]
}}

Rules:
- You MUST output valid JSON only. No comments, no extra explanation.
- Produce 3 to 7 gaps.
- Each gap MUST include 2–4 related_papers with paper_index that exists in input.
- Each evidence_from_paper MUST be grounded in that paper's limitations or possible_future_work fields.

DIVERSITY (to avoid duplicates, not to force templates):
- Each gap MUST be assigned exactly ONE category from:
  (1) Data/Benchmarks
  (2) Model/Architecture
  (3) Training/Optimization
  (4) Evaluation/Methodology
  (5) Robustness/Generalization
  (6) Theory/Understanding
- Cover at least 3 DISTINCT categories across all gaps.

ANTI-DEFAULT (CRITICAL):
- Do NOT default to "missing mathematical definition/formalism", "need ablations", or "poor robustness".
  You may include these ONLY IF at least TWO papers explicitly support them in the provided evidence, and you must state
  what exact object/setting is affected (topic-specific).
- Do NOT default to "broader datasets/benchmarks" unless at least TWO papers explicitly cite dataset/benchmark coverage as a limitation.
  If you mention dataset expansion, specify what changes (domain shift, modality, scale regime, label noise, class imbalance, open-set, etc.).

"""
    return prompt


def _build_refine_prompt(
    papers_with_analysis: List[Dict[str, Any]],
    first_pass_result: Dict[str, Any],
) -> str:
    """
    Second-pass refinement prompt to remove template-like gaps and replace them with evidence-backed, topic-specific ones.
    """
    papers_json = json.dumps(papers_with_analysis, indent=2, ensure_ascii=False)
    first_json = json.dumps(first_pass_result, indent=2, ensure_ascii=False)

    return f"""
You are refining a first-pass set of research gaps that may contain generic/template gaps.

INPUT PAPERS (do not repeat in output):
{papers_json}

FIRST-PASS GAPS (to be improved):
{first_json}

Your task:
- Rewrite the gaps to REMOVE generic template gaps unless they are explicitly supported by evidence from at least TWO papers.
- Prefer mechanism-level gaps tightly tied to the papers' stated limitations/future-work.
- Keep 3–7 gaps, cover at least 3 categories.
- Each gap must include 2–4 related_papers, and each evidence_from_paper must paraphrase ONE concrete limitation/future-work item.

Strong prohibitions unless explicitly supported by >=2 papers:
- "missing mathematical definition/formalism"
- "need ablations"
- generic "robustness is poor" statements

Output STRICT JSON only with the same schema:
{{
  "research_gaps": [...]
}}
"""


def extract_relevant_sections_from_pdf_fullscan(
    pdf_url: str,
    save_dir: str = "pdf_cache",
    pages_per_chunk: int = 8,
    include_sections: List[str] = None,
    max_chars: int = 20000,
):
    if include_sections is None:
        include_sections = ["introduction", "methods", "experiments", "limitations", "conclusion"]

    pdf_path = download_pdf(pdf_url, save_dir)
    if not pdf_path:
        return ""

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    collected = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)

        chunk_text = ""
        for p in range(start, end):
            chunk_text += doc[p].get_text("text") + "\n"

        sections = _find_sections_in_text(chunk_text)

        for sec in sections:
            if sec["name"] in include_sections:
                snippet = chunk_text[sec["start"]: sec["end"]]
                collected.append(f"\n===== SECTION {sec['name'].upper()} (pages {start}-{end}) =====\n" + snippet)

    if not collected:
        full_text = ""
        for p in range(min(40, total_pages)):
            full_text += doc[p].get_text("text") + "\n"
        return full_text[:max_chars]

    final = "\n".join(collected)
    return final[:max_chars]


def mine_research_gaps(
    topic: str,
    max_papers: int = 6,
    pre_fetched_papers: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level interface:
    1) Fetch papers
    2) Analyze each paper
    3) Gap mine via LLM
    4) If output shows strong template bias, run an automatic refinement pass
    """
    print(f"[1] Preparing up to {max_papers} papers for topic: {topic!r}")
    papers = pre_fetched_papers if pre_fetched_papers is not None else fetch_arxiv_papers(topic, max_results=max_papers)
    if not papers:
        raise RuntimeError("No papers fetched from arXiv.")

    papers_with_analysis: List[Dict[str, Any]] = []
    for idx, p in enumerate(papers):
        print(f"\n[2] Analyzing paper {idx} / {len(papers)-1}")
        print(f"    Title: {p['title']}")

        title = p["title"]
        abstract = p["summary"]
        pdf_url = p.get("pdf_url")

        fulltext_snippet = ""
        if pdf_url:
            print(f"    PDF URL = {pdf_url}")
            from pdf_processor_fullscan import process_pdf_fullscan
            pdf_structured = process_pdf_fullscan(pdf_url)
            if pdf_structured:
                fulltext_snippet = json.dumps(pdf_structured, indent=2)

        if fulltext_snippet:
            print("    Using STRUCTURED FULL PDF SUMMARY for analysis.")
            analysis = analyze_paper_with_fulltext(title, abstract, fulltext_snippet)
        else:
            print("    [Warning] No usable PDF info, falling back to abstract-only analysis.")
            analysis = analyze_paper(title, abstract)

        limitations = analysis.get("limitations", [])
        future_work = analysis.get("possible_future_work", [])

        papers_with_analysis.append(
            {
                "paper_index": idx,
                "title": title,
                "limitations": limitations,
                "possible_future_work": future_work,
            }
        )

    # First pass
    print("\n[3] Mining research gaps with LLM (pass 1) ...")
    user_prompt = build_gap_mining_prompt(papers_with_analysis)
    raw_output = call_llm_raw(
        system_prompt=SYSTEM_PROMPT_GAP_MINER,
        user_prompt=user_prompt,
        max_output_tokens=4000,
    )

    try:
        result = clean_and_parse_json(raw_output)
    except Exception as e:
        print("Failed to parse JSON from gap mining model output.")
        print("Raw output snippet:\n", raw_output[:500])
        raise e

    gaps = result.get("research_gaps", []) if isinstance(result, dict) else []
    generic_count = _count_generic_gaps(gaps)

    # Additional signal: not enough technical specificity across multiple gaps
    low_spec = 0
    for g in gaps:
        if not _has_minimum_technical_specificity(g):
            low_spec += 1

    # Trigger refinement if "template gaps" dominate or specificity is poor.
    # Example triggers:
    # - >=2 generic gaps among 3–7
    # - or >=2 low-specificity gaps
    should_refine = (generic_count >= 2) or (low_spec >= 2)

    if should_refine:
        print(f"\n[4] Detected template-like bias (generic_count={generic_count}, low_spec={low_spec}). Refining (pass 2) ...")
        refine_prompt = _build_refine_prompt(papers_with_analysis, result)
        raw_output_2 = call_llm_raw(
            system_prompt=SYSTEM_PROMPT_GAP_MINER,
            user_prompt=refine_prompt,
            max_output_tokens=4000,
        )
        try:
            refined = clean_and_parse_json(raw_output_2)
            # Use refined if it parses and has gaps
            if isinstance(refined, dict) and refined.get("research_gaps"):
                return refined
        except Exception:
            # If refinement fails, fall back to first pass
            print("[Warning] Refinement output parse failed; falling back to pass 1 result.")

    return result


def pretty_print_gaps(result: Dict[str, Any]):
    gaps = result.get("research_gaps", [])
    print("\n=== Research Gaps ===")
    if not gaps:
        print("No gaps found or JSON structure is unexpected.")
        return

    for gap in gaps:
        gap_id = gap.get("gap_id", "Unknown")
        description = gap.get("description", "")
        why_important = gap.get("why_important", "")
        related_papers = gap.get("related_papers", [])
        suggested_directions = gap.get("suggested_directions", [])

        print(f"\n[{gap_id}]")
        print("Description:")
        print(f"  {description}")
        print("\nWhy important:")
        print(f"  {why_important}")

        print("\nRelated papers:")
        for rp in related_papers:
            idx = rp.get("paper_index", -1)
            title = rp.get("title", "")
            evidence = rp.get("evidence_from_paper", "")
            print(f"  - (paper_index={idx}) {title}")
            print(f"    Evidence: {evidence}")

        print("\nSuggested directions:")
        for i, d in enumerate(suggested_directions, start=1):
            print(f"  {i}. {d}")

        print("-" * 80)

    print("\n======================\n")


def main():
    print("=== Research Gap Miner Demo ===")
    topic = input("Enter a topic (e.g., 'graph neural network' or 'time series forecasting'): ").strip()
    if not topic:
        topic = "graph neural network"

    max_papers = 6
    result = mine_research_gaps(topic, max_papers=max_papers)
    pretty_print_gaps(result)


if __name__ == "__main__":
    main()




