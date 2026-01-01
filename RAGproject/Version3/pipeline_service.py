# pipeline_service.py
from typing import Dict, Any
from gap_miner import mine_research_gaps
from idea_generator import generate_ideas_from_gaps
from idea_evaluator import evaluate_ideas
from paper_draft_writer import select_top_idea, fetch_supporting_papers_for_idea, generate_paper_draft

def run_full_pipeline(topic: str, max_papers: int = 6) -> Dict[str, Any]:
    gaps = mine_research_gaps(topic, max_papers=max_papers)  # :contentReference[oaicite:2]{index=2}
    ideas = generate_ideas_from_gaps(gaps)                    # :contentReference[oaicite:3]{index=3}
    evaluation = evaluate_ideas(ideas)                        # :contentReference[oaicite:4]{index=4}

    top_idea = select_top_idea(ideas, evaluation)             # :contentReference[oaicite:5]{index=5}
    support = fetch_supporting_papers_for_idea(topic, top_idea, max_papers=6)  # :contentReference[oaicite:6]{index=6}
    draft = generate_paper_draft(topic, top_idea, support)    # :contentReference[oaicite:7]{index=7}

    return {
        "gaps": gaps,
        "ideas": ideas,
        "evaluation": evaluation,
        "draft": draft,
    }
