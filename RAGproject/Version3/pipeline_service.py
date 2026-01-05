# pipeline_service.py
from typing import Any, Dict, List

from gap_miner import mine_research_gaps
from idea_evaluator import evaluate_ideas
from idea_generator import generate_ideas_from_gaps
from paper_draft_writer import (
    fetch_supporting_papers_for_idea,
    generate_paper_draft,
    select_top_idea,
)
from paper_retriever import fetch_arxiv_papers


def run_full_pipeline(topic: str, max_papers: int = 6) -> Dict[str, Any]:
    """
    Convenience synchronous runner (not used by the FastAPI worker).

    Returns a dict with intermediate artifacts so callers can stream/persist them:
      papers, gaps, ideas, evaluation, top_idea, supporting_papers, draft
    """
    papers: List[Dict[str, Any]] = fetch_arxiv_papers(topic, max_results=max_papers)

    gaps = mine_research_gaps(topic, max_papers=max_papers, pre_fetched_papers=papers)
    ideas = generate_ideas_from_gaps(gaps)
    evaluation = evaluate_ideas(ideas)

    top_idea = select_top_idea(ideas, evaluation)
    supporting = fetch_supporting_papers_for_idea(topic, top_idea, max_papers=6)
    draft = generate_paper_draft(topic, top_idea, supporting)

    return {
        "papers": {"papers": papers},
        "gaps": gaps,
        "ideas": ideas,
        "evaluation": evaluation,
        "top_idea": top_idea,
        "supporting_papers": supporting,
        "draft": draft,
    }
