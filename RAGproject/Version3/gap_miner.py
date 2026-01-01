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
from typing import List, Dict, Any

from paper_retriever import fetch_arxiv_papers
from llm_client import call_llm_raw
from pdf_utils import extract_relevant_sections_from_pdf, download_pdf, _find_sections_in_text
from paper_analyzer import analyze_paper, analyze_paper_with_fulltext, clean_and_parse_json
from json_repair import repair_json

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT_GAP_MINER = """
You are an expert machine learning meta-researcher.
You identify high-level research gaps from paper analyses.

Your job is to:
1) Aggregate limitations into 3–7 high-level research gaps.
2) Ensure gaps are technically specific (mention data modalities, architectures, specific mechanisms, evaluation setups, etc.).
3) Output STRICT JSON only.

IMPORTANT PRINCIPLES:
- Do NOT assume any single type of gap (e.g., mathematical formalism, data scale, robustness) must appear.
- Instead, infer gaps directly from the limitations and future work of the provided papers.
- Typical gap dimensions can include (but are not limited to):
  * Data: missing modalities, low-quality datasets, lack of real-world benchmarks, domain shifts.
  * Models: architectural limitations, scalability issues, inability to handle certain structures or tasks.
  * Training & Optimization: unstable training, high computational cost, poor sample efficiency.
  * Evaluation: missing baselines, weak metrics, limited ablation studies, narrow benchmarks.
  * Robustness & Generalization: poor out-of-distribution performance, vulnerability to noise or adversaries.
  * Theory & Understanding: lack of theoretical guarantees, unclear inductive biases, limited interpretability.
- You may include theoretical/mathematical gaps when they are clearly reflected in the limitations, but do NOT force them.

OUTPUT STYLE:
- JSON only.
- Gaps should be actionable and grounded in the provided papers.

ANTI-DEFAULT RULE (IMPORTANT):
- Do NOT default to "evaluate on broader/larger/more diverse datasets/benchmarks" unless at least TWO papers explicitly cite dataset/benchmark coverage as a limitation.
- If you mention dataset expansion, you must specify exactly what changes (e.g., domain shift type, missing modality, scale regime, label noise, class imbalance, open-set setting), not generic "more datasets".

"""



def build_gap_mining_prompt(papers_with_analysis: List[Dict[str, Any]]) -> str:
    """
    The prompt for LLM is constructed by packaging the limitations and future work of multiple papers.

    Input format example：
    [
      {
        "paper_index": 0,
        "title": "...",
        "limitations": [...],
        "possible_future_work": [...],
      },
      ...
    ]

    JSON output:
    {
      "research_gaps": [
        {
          "gap_id": "G1",
          "category": "Data/Benchmarks | Model/Architecture | Training/Optimization | Evaluation/Methodology | Robustness/Generalization | Theory/Understanding",
          "description": "A concise, technically specific description of this research gap (2-4 sentences).",
          "why_important": "Why this gap is important or impactful (2-4 sentences).",
          "related_papers": [
            {
              "paper_index": 0,
              "title": "Exact paper title here",
              "evidence_from_paper": "Specific limitation/future-work evidence from this paper."
            }
          ],
          "suggested_directions": [
            "A concrete, technically detailed research direction.",
            "Another possible direction."
          ]
        }
      ]
    }



    """
    input_data_json = json.dumps(
        papers_with_analysis,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are given analysis of multiple papers in JSON format.
Each item has: paper_index, title, limitations, possible_future_work.

Here is the JSON array of papers (DO NOT repeat this array back in your output):
{input_data_json}

Please analyze these limitations and future-work items and output a STRICT JSON object with the following structure:

{{
  "research_gaps": [
    {{
      "gap_id": "G1",
      "description": "A concise, technically specific description of this research gap (2-4 sentences). Avoid vague statements.",
      "why_important": "Why this gap is important or impactful (2-4 sentences), with reference to data modalities, model families, or evaluation regimes when possible.",
      "related_papers": [
        {{
          "paper_index": 0,
          "title": "Exact paper title here",
          "evidence_from_paper": "One or two sentences paraphrasing the limitation/future-work from this paper that support this gap. Be specific, e.g., mention missing datasets, missing mathematical formalism, missing baselines, or missing ablations."
        }}
      ],
      "suggested_directions": [
        "A concrete, technically detailed research direction or question addressing this gap.",
        "Another possible direction, again technically specific."
      ]
    }},
    ...
  ]
}}

Rules:
- You MUST output valid JSON only, no comments, no extra explanation.
- Do NOT repeat the input JSON in your output.
- Try to produce between 3 and 7 research gaps.
- Group similar limitations/future-work items across papers into coherent gaps
  (e.g., missing datasets, lack of robust evaluation, architectural constraints, scalability issues, theoretical understanding, etc.).
- Do NOT force any pre-specified theme: infer the gap types purely from the provided limitations.

DIVERSITY CONSTRAINT (CRITICAL):
- Each research gap MUST be assigned exactly ONE category from:
  (1) Data/Benchmarks
  (2) Model/Architecture
  (3) Training/Optimization
  (4) Evaluation/Methodology
  (5) Robustness/Generalization
  (6) Theory/Understanding

- Your final set of 3–7 gaps MUST cover at least 3 DISTINCT categories.

- At most ONE gap may be primarily about "Data/Benchmarks".
- At most ONE gap may be primarily about "Evaluation/Methodology".

- If a gap involves evaluation, it must be tied to a specific methodological weakness
  (e.g., missing ablations, missing baselines, flawed metrics),
  NOT generic statements like "broader or more diverse datasets".

"""
    return prompt


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

    # fallback 1 — if no sections found, return truncated full text
    if not collected:
        full_text = ""
        for p in range(min(40, total_pages)):
            full_text += doc[p].get_text("text") + "\n"
        return full_text[:max_chars]

    final = "\n".join(collected)
    return final[:max_chars]


def mine_research_gaps(topic: str, max_papers: int = 6) -> Dict[str, Any]:
    """
    High-level interface:
    1) Fetch several papers from arXiv based on the topic.
    2) Call `analyze_paper` / `analyze_paper_with_fulltext` on each paper to obtain structured analysis.
    3) Build a gap-mining prompt and call LLM to obtain the `research_gaps` structure.

    return：
        result_dict: After parsing JSON dict:
        {
          "research_gaps": [ ... ]
        }
    """
    print(f"[1] Fetching up to {max_papers} papers for topic: {topic!r}")
    papers = fetch_arxiv_papers(topic, max_results=max_papers)
    if not papers:
        raise RuntimeError("No papers fetched from arXiv.")

    # Step 1: Call analyze_paper for each paper
    papers_with_analysis: List[Dict[str, Any]] = []
    for idx, p in enumerate(papers):
        print(f"\n[2] Analyzing paper {idx} / {len(papers)-1}")
        print(f"    Title: {p['title']}")

        title = p["title"]
        abstract = p["summary"]
        pdf_url = p.get("pdf_url")

        # Extracting only key chapters from the PDF as supplementary context
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

    # Step 2: Call LLM to perform gap mining
    print("\n[3] Mining research gaps with LLM ...")
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

    return result


def pretty_print_gaps(result: Dict[str, Any]):
    """
    print research_gaps list。
    """
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



