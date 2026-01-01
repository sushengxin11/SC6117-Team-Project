# paper_draft_writer.py
#
# Functionality：
#   Starting from a single topic, automatically complete the following：
#     1. Gap finding -> Idea generation -> Idea evaluation -> Top Idea selection
#     2. Retrieve Support Papers for Top Ideas
#     3. Download and read the PDF text of Support Papers (Methods/Experiments)
#     4. Generate a draft paper
#
#   Output: Terminal printing + saving as a Markdown file

import json
import os
import re
import sys

from typing import Dict, Any, List
from llm_client import call_llm_raw
from paper_retriever import fetch_arxiv_papers
from gap_miner import mine_research_gaps
from idea_generator import generate_ideas_from_gaps
from idea_evaluator import evaluate_ideas
from pathlib import Path

# -------------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------------
SYSTEM_PROMPT_PAPER_WRITER = """
You are an expert ML researcher and academic writer.
You write high-quality ML research paper drafts in a standard conference style (NeurIPS/ICML/ICLR).

CRITICAL RULES:
1. You MUST use the provided 'technical_content' EXACTLY. Do NOT hallucinate methods or equations.
2. The METHOD section must:
   - Use equations extracted from the PDF (the 'math' array).
   - Combine the idea's proposed method with real PDF equations to form a mathematically grounded formulation.
   - Include input definition, model pipeline, and loss function.
3. The RELATED WORK section must:
   - Make **explicit technical comparisons** using the PDF's extracted methods and experiments.
   - Mention differences in mathematical formulation, mechanism, and training strategies.
4. The EXPERIMENTS section must:
   - Use the datasets, baselines, and metrics provided in the idea.
   - If the PDF contains experiment details, refer to them explicitly (but do not fabricate results).
5. Output STRICT JSON only.

You must produce:
{
  "title": "...",
  "abstract": "...",
  "introduction": "...",
  "related_work": "...",
  "method": "...",
  "experiments": "...",
  "conclusion": "...",
  "references": [...]
}
"""



def select_top_idea(ideas_result: Dict[str, Any], eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """The best idea will be selected based on the ratings."""
    ideas: List[Dict[str, Any]] = ideas_result.get("ideas", [])
    ranking: List[Dict[str, Any]] = eval_result.get("ranking", [])

    if not ideas or not ranking:
        raise RuntimeError("No ideas or ranking found.")

    idea_map = {idea["idea_id"]: idea for idea in ideas if "idea_id" in idea}
    top_idea_id = ranking[0]["idea_id"]
    top_idea = idea_map.get(top_idea_id)

    if top_idea is None:
        raise RuntimeError(f"Top idea_id={top_idea_id} not found.")

    return top_idea


def fetch_supporting_papers_for_idea(topic: str, idea: Dict[str, Any], max_papers: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve supporting papers and perform a full scan (with cache) on each PDF.
    Write the structured results to the content_snippet for use in the subsequent paper writing stage.
    """

    query = topic

    print(f"\n[Retriever] Searching arXiv for: {query} ...")
    papers = fetch_arxiv_papers(query, max_results=max_papers)

    print(f"[Retriever] Analyzing PDF content for {len(papers)} papers...")

    from pdf_processor_fullscan import process_pdf_fullscan
    import json

    for p in papers:
        pdf_url = p.get("pdf_url")
        # 默认先放上 Abstract
        p["content_snippet"] = f"Abstract: {p.get('summary', '')}"

        if pdf_url:
            print(f"  -> Processing PDF (FULLSCAN): {p['title'][:50]}...")

            structured = process_pdf_fullscan(pdf_url)

            content_parts = []
            # 1. 永远保留 Abstract
            content_parts.append(f"Abstract: {p.get('summary', '')}")

            # 2. 结构化 fullscan 结果
            if structured:
                structured_json = json.dumps(structured, indent=2)
                content_parts.append(
                    "--- STRUCTURED TECHNICAL SUMMARY (math/methods/experiments/limitations) ---\n"
                    + structured_json
                )
                print("     [Success-FULLSCAN] Structured PDF summary extracted (from cache or fresh).")
            else:
                print("     [Warning-FULLSCAN] No structured data extracted from PDF.")

            p["content_snippet"] = "\n\n".join(content_parts)

        else:
            print("     [Skip] No PDF URL available.")

    return papers



def build_paper_draft_prompt(topic: str, idea: Dict[str, Any], support_papers: List[Dict[str, Any]]) -> str:
    paper_summaries = []

    for idx, p in enumerate(support_papers[:6]):
        authors = p.get("authors", [])
        year = p.get("published", "")[:4] if p.get("published","") else ""
        technical = p.get("content_snippet", "")

        paper_summaries.append({
            "index": idx,
            "title": p.get("title", ""),
            "authors": authors,
            "year": year,
            "link": p.get("link", ""),
            "technical_content": technical
        })

    support_json = json.dumps(paper_summaries, indent=2, ensure_ascii=False)
    idea_json = json.dumps(idea, indent=2, ensure_ascii=False)

    prompt = f"""
You are given:
1) A high-level topic: "{topic}"

2) A selected research idea (JSON):
{idea_json}

3) A set of supporting papers (JSON):
{support_json}

IMPORTANT INSTRUCTION ON 'technical_content':
- The technical_content field includes REAL equations, methods, and experiments extracted from the actual PDF.
- You MUST use:
    (a) technical_content.math → to construct formal method equations.
    (b) technical_content.methods → for mechanism descriptions.
    (c) technical_content.experiments → to contrast experimental setups.
    (d) technical_content.limitations → to justify your contribution.

Using this information, write a structured research paper DRAFT in strict JSON format:

{{
  "title": "Paper Title",
  "abstract": "...",
  "introduction": "...",
  "related_work": "Summarize supporting papers. specific technical comparisons are required.",
  "method": "Detailed method section. MUST include LaTeX math formulations (Loss, Model Architecture) consistent with the idea and inspired by supporting papers' notations.",
  "experiments": "Experimental setup, baselines, and metrics.",
  "conclusion": "...",
  "references": ["Ref 1", "Ref 2"]
}}

Rules:
- STRICT JSON output.
- References must be exact (use provided Title/Authors/Link).
"""
    return prompt


def generate_paper_draft(topic: str, idea: Dict[str, Any], support_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate and parse the draft"""
    user_prompt = build_paper_draft_prompt(topic, idea, support_papers)

    print("\n[LLM] Generating draft (this may take a minute)...")
    raw_output = call_llm_raw(
        system_prompt=SYSTEM_PROMPT_PAPER_WRITER,
        user_prompt=user_prompt,
        max_output_tokens=4000,
    )

    try:
        draft = json.loads(raw_output, strict=False)
    except json.JSONDecodeError:
        try:
            # Step 1. Cleaning Markdown
            text = raw_output.strip()
            text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'```$', '', text, flags=re.MULTILINE)
            # 2. repair LaTeX \
            fixed_text = re.sub(r'\\(?![/u"bfnrt\\])', r'\\\\', text)
            draft = json.loads(fixed_text, strict=False)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                print("[Info] Standard parse failed, using json_repair...")
                draft = json.loads(repair_json(raw_output))
            except ImportError:
                print("\n[Critical Error] JSON parse failed. Please pip install json_repair")
                with open("debug_failed_draft.txt", "w", encoding="utf-8") as f:
                    f.write(raw_output)
                raise

    return draft


def pretty_print_draft(draft: Dict[str, Any]):
    print("\n=== Generated Paper Draft ===\n")
    sections = ["title", "abstract", "introduction", "related_work", "method", "experiments", "conclusion"]
    for sec in sections:
        print(f"\n===== {sec.upper()} =====\n")
        print(draft.get(sec, ""))

    print("\n===== REFERENCES =====\n")
    for r in draft.get("references", []):
        print(f"- {r}")
    print("\n==============================\n")


def format_complex_content(content, level=3) -> str:
    """
    Recursively converts complex JSON content (Dict/List) into Markdown strings.
    """
    if isinstance(content, str):
        return content.strip()

    elif isinstance(content, list):
        # Convert lists to Markdown bullet points
        lines = []
        for item in content:
            # Recursively process each item in the list
            formatted_item = format_complex_content(item, level + 1)
            lines.append(f"* {formatted_item}")
        return "\n".join(lines)

    elif isinstance(content, dict):

        lines = []
        for k, v in content.items():
            # Beautification Key
            clean_key = k.replace("_", " ").title()

            formatted_val = format_complex_content(v, level + 1)

            # If the value contains newlines, is a list, or is very long, use a heading format.
            if "\n" in formatted_val or isinstance(v, (dict, list)) or len(formatted_val) > 60:
                header_prefix = "#" * level
                lines.append(f"{header_prefix} {clean_key}\n{formatted_val}")
            else:
                # Otherwise, use the bold key format.
                lines.append(f"**{clean_key}**: {formatted_val}")
        return "\n\n".join(lines)

    else:
        return str(content)


def save_draft_to_markdown(draft: Dict[str, Any], output_dir: str) -> str:
    """Saving as Markdown，using format_complex_content"""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Directory created: {output_dir}")
        except Exception as e:
            print(f"[Warning] Could not create dir {output_dir}, saving to current dir.")
            output_dir = "."

    title = draft.get("title", "untitled_paper")
    filename = re.sub(r"[^A-Za-z0-9]+", "_", title.strip())[:80] + ".md"
    filepath = os.path.join(output_dir, filename)

    lines = [f"# {title}\n"]

    sections = ["abstract", "introduction", "related_work", "method", "experiments", "conclusion"]

    for sec in sections:
        # update second topics
        lines.append(f"## {sec.replace('_', ' ').title()}\n")

        raw_content = draft.get(sec, "")

        formatted_text = format_complex_content(raw_content, level=3)

        lines.append(formatted_text + "\n\n")

    lines.append("## References\n")
    refs = draft.get("references", [])

    if isinstance(refs, list):
        for i, r in enumerate(refs, 1):
            lines.append(f"[{i}] {str(r)}\n")
    elif isinstance(refs, str):
        lines.append(refs + "\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def main():
    print("=== Paper Draft Writer Demo ===")
    topic = input("Please enter a topic: ").strip()
    if not topic:
        topic = "large language models"

    # Step 1. gap -> ideas
    print("\n[1] Mining gaps, generating ideas, and evaluating them ...")
    gaps_result = mine_research_gaps(topic, max_papers=6)  # 增加到 6 篇
    ideas_result = generate_ideas_from_gaps(gaps_result)
    eval_result = evaluate_ideas(ideas_result)

    # Step 2. Select Idea
    print("\n[2] Selecting top idea ...")
    top_idea = select_top_idea(ideas_result, eval_result)
    print(f"Selected top idea: {top_idea.get('working_title')}")

    # Step 3. Fetch papers + PDF Content
    print("\n[3] Fetching supporting papers (and reading PDFs) ...")
    support_papers = fetch_supporting_papers_for_idea(topic, top_idea, max_papers=6)

    # Step 4. Generate Draft
    print("\n[4] Generating paper draft with LLM ...")
    draft = generate_paper_draft(topic, top_idea, support_papers)

    # Step 5. Save
    pretty_print_draft(draft)

    current_dir = Path(__file__).resolve().parent
    target_path = current_dir / "paper_draft"
    target_path.mkdir(parents=True, exist_ok=True)
    filepath = save_draft_to_markdown(draft, output_dir=str(target_path))
    print(f"Paper draft saved to: {filepath}")


if __name__ == "__main__":
    main()
