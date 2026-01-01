# paper_analyzer.py
#
# Functionality: Perform structured analysis on papers (title + abstract + optional text excerpts):
# 1. Research task / problem
# 2. Methodology summary
# 3. Main contributions
# 4. Limitations / shortcomings
# 5. Potential research directions / future work
# 6. Possible experimental designs
# Analyze using LLM, output JSON, and parse it locally into a Python dict.

import json
from typing import Dict
import re
import sys

from llm_client import call_llm_raw
from paper_retriever import fetch_arxiv_papers


SYSTEM_PROMPT = """
You are an expert machine learning researcher.
You analyze ML/NLP/CV/GNN/Optimization papers and extract precise technical content.
You MUST output STRICT JSON only.

CRITICAL RULE: UNIVERSAL MATHEMATICAL FORMULATION REQUIREMENT
=============================================================
For ANY research domain (GNN, Transformer, Diffusion, RL, CV, NLP, Time Series, etc.):
- You MUST express the method using mathematical notation.
- The "method" field MUST include a formal definition of:
  1. Input representation (e.g., graph G=(V,E), image I ∈ ℝ^{H×W×3}, sequence x₁:ₜ, state s_t, etc.).
  2. Core model operations (e.g., message passing: h_v^{(k+1)} = φ( h_v^{(k)}, ⊕_{u∈N(v)} ψ(h_u^{(k)}, e_{uv}) ), transformer attention QKᵀ/√d, convolution, diffusion update q(x_t|x_{t-1}), etc.).
  3. Objective / loss function (e.g., cross-entropy, contrastive loss, KL divergence, MSE).

IMPORTANT:
- Even if the original paper abstract contains NO equations, you MUST construct a standard mathematical formulation based on typical representations in the domain.
- DO NOT say "no math available" unless the paper is clearly non-technical.
- DO NOT invent unrealistic equations inconsistent with the described method. Use standard formulations from common ML practice.

OUTPUT FIELDS:
- "task": one-sentence problem definition.
- "method": 3–6 sentences, MUST contain math as described above.
- "contributions": 2–4 bullet points.
- "limitations": 2–6 limitations (check for missing baselines, ablations, datasets, mathematical clarity).
- "possible_future_work": 2–6 follow-up directions.
- "potential_experiments": 2–6 experiment ideas.

MUST output VALID JSON.
"""



def build_analysis_prompt(title: str, abstract: str) -> str:
    """
    Construct a user prompt for the LLM, requesting it to return JSON.
    """
    template = f"""
You are given the TITLE and ABSTRACT of a research paper.

Please analyze it and output a STRICT JSON object with the following fields:

- "task": a one-sentence description of the problem.
- "method": 3-5 sentences describing the key technical approach.
    - [MANDATORY] You MUST include key mathematical formulations here using LaTeX syntax (single backslash like \\( x_t \\) or just standard LaTeX).
    - Define the inputs, the model mechanism (e.g., attention formula, message passing equation), and the loss function.
    - If no math is found, explicitly state that as a limitation.
- "contributions": a list of 2-4 main contributions.
- "limitations": a list of 2-6 limitations.
    - Specifically check for: Missing baselines, missing ablation studies, and **absence of formal mathematical definitions**.
- "possible_future_work": a list of 2-6 technically specific follow-up directions.
- "potential_experiments": a list of 2-6 concrete experiment ideas.

The output MUST be valid JSON.

TITLE:
{title}

ABSTRACT:
{abstract}
"""
    return template


def clean_and_parse_json(raw_text: str):
    """
    Clean and parse JSON, including fixes for LaTeX formulas and control characters.
    """
    try:
        # A: strict=False
        return json.loads(raw_output := raw_text, strict=False)
    except json.JSONDecodeError:
        try:
            # print("[Info] Standard JSON parse failed, attempting strict repair...")

            # 1. Cleaning Markdown
            text = raw_text.strip()
            text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'```$', '', text, flags=re.MULTILINE)

            # 2. turn \ into \\
            fixed_text = re.sub(r'\\(?![/u"bfnrt\\])', r'\\\\', text)

            return json.loads(fixed_text, strict=False)

        except json.JSONDecodeError:
            # B:using jason_repair
            try:
                from json_repair import repair_json
                return json.loads(repair_json(raw_text))
            except ImportError:
                print("\n[Critical Error] JSON parse failed. Please pip install json_repair.")
                raise
            except Exception:
                print(f"[Error] Failed to parse JSON even with repair. Raw:\n{raw_text}")
                raise


def analyze_paper(title: str, abstract: str) -> Dict:
    user_prompt = build_analysis_prompt(title, abstract)
    raw_output = call_llm_raw(SYSTEM_PROMPT, user_prompt)
    return clean_and_parse_json(raw_output)

def analyze_paper_with_fulltext(title: str, abstract: str, fulltext_snippet: str) -> Dict:
    user_prompt = build_analysis_prompt(title, abstract)
    if fulltext_snippet:
        user_prompt += (
            "\n\n----- BEGIN FULLTEXT SNIPPET -----\n"
            f"{fulltext_snippet}\n"
            "----- END FULLTEXT SNIPPET -----\n"
            "Reminder: Extract mathematical definitions (Loss, Model) from the snippet if available."
        )
    raw_output = call_llm_raw(SYSTEM_PROMPT, user_prompt)
    return clean_and_parse_json(raw_output)


def pretty_print_analysis(analysis: Dict):
    """
    Print results
    """
    print("\n=== Paper Analysis Result ===")

    print("\n[Task]")
    print(analysis.get("task", ""))

    print("\n[Method]")
    print(analysis.get("method", ""))

    print("\n[Contributions]")
    for i, c in enumerate(analysis.get("contributions", []), start=1):
        print(f"  {i}. {c}")

    print("\n[Limitations]")
    for i, c in enumerate(analysis.get("limitations", []), start=1):
        print(f"  {i}. {c}")

    print("\n[Possible Future Work]")
    for i, c in enumerate(analysis.get("possible_future_work", []), start=1):
        print(f"  {i}. {c}")

    print("\n[Potential Experiments]")
    for i, c in enumerate(analysis.get("potential_experiments", []), start=1):
        print(f"  {i}. {c}")

    print("\n=============================\n")


def main():
    """
    Fetch a paper from arXiv and then perform structured analysis.(For testing purposes of paper_analyzer only)
    """
    print("=== Paper Analyzer Demo (arXiv) ===")
    topic = input("Enter a topic (e.g., 'large language models'): ").strip()
    if not topic:
        topic = "large language models"

    print(f"\n[1] Fetching papers for topic: {topic!r} ...")
    papers = fetch_arxiv_papers(topic, max_results=10)
    if not papers:
        print("No papers fetched. Try another topic.")
        return

    paper = papers[0]
    title = paper["title"]
    abstract = paper["summary"]

    print("\n[2] Selected paper:")
    print(f"Title   : {title}")
    print(f"Link    : {paper['link']}")
    print(f"Published: {paper['published']}")
    print("\nAbstract:")
    print(abstract[:500] + "..." if len(abstract) > 500 else abstract)

    print("\n[3] Analyzing paper with LLM ...")
    analysis = analyze_paper(title, abstract)
    pretty_print_analysis(analysis)


if __name__ == "__main__":
    main()


