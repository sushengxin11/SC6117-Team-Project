# api_server.py
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gap_miner import mine_research_gaps
from idea_generator import generate_ideas_from_gaps
from idea_evaluator import evaluate_ideas
from paper_draft_writer import (
    fetch_supporting_papers_for_idea,
    generate_paper_draft,
    select_top_idea,
)
from paper_retriever import fetch_arxiv_papers

TASKS_DIR = Path("tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)

Stage = Literal[
    "queued",
    "paper_retrieval",
    "gap_mining",
    "idea_generation",
    "idea_evaluation",
    "support_paper_retrieval",
    "paper_drafting",
    "completed",
    "failed",
]
ResultStage = Literal[
    "papers",
    "gaps",
    "ideas",
    "evaluation",
    "top_idea",
    "supporting_papers",
    "draft",
]


class CreateTaskRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    max_papers: int = Field(6, ge=1, le=12)


class CreateTaskResponse(BaseModel):
    task_id: str
    status: Stage


class TaskStatusResponse(BaseModel):
    task_id: str
    status: Stage
    stage: Stage
    progress: float
    message: str
    topic: Optional[str] = None


class TaskListItem(BaseModel):
    task_id: str
    topic: str
    status: Stage
    created_at: float


app = FastAPI(title="RAG Research Pipeline API", version="0.2")
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")


def _task_dir(task_id: str) -> Path:
    return TASKS_DIR / task_id


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_status(
    task_id: str,
    *,
    status: Stage,
    stage: Stage,
    progress: float,
    message: str,
    topic: Optional[str] = None,
) -> None:
    d = _task_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    status_path = d / "status.json"
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "status": status,
        "stage": stage,
        "progress": float(max(0.0, min(1.0, progress))),
        "message": message,
    }
    if topic is not None:
        payload["topic"] = topic
    _write_json(status_path, payload)


def _result_path(task_id: str, stage: ResultStage) -> Path:
    mapping = {
        "papers": "papers.json",
        "gaps": "gaps.json",
        "ideas": "ideas.json",
        "evaluation": "evaluation.json",
        "top_idea": "top_idea.json",
        "supporting_papers": "supporting_papers.json",
        "draft": "draft.json",
    }
    return _task_dir(task_id) / mapping[stage]


def _pipeline_worker(task_id: str, topic: str, max_papers: int) -> None:
    """
    Pipeline phases and persisted outputs:

    - paper_retrieval -> papers.json
    - gap_mining -> gaps.json
    - idea_generation -> ideas.json
    - idea_evaluation -> evaluation.json + top_idea.json
    - support_paper_retrieval -> supporting_papers.json
    - paper_drafting -> draft.json
    """
    try:
        _update_status(task_id, status="queued", stage="queued", progress=0.0, message="Queued", topic=topic)

        # 0) fetch papers (persist ASAP for frontend)
        _update_status(
            task_id,
            status="paper_retrieval",
            stage="paper_retrieval",
            progress=0.05,
            message="Fetching candidate papers from arXiv...",
        )
        papers = fetch_arxiv_papers(topic, max_results=max_papers)
        if not papers:
            raise RuntimeError("No papers fetched from arXiv.")
        _write_json(_result_path(task_id, "papers"), {"papers": papers})

        # 1) gaps
        _update_status(
            task_id,
            status="gap_mining",
            stage="gap_mining",
            progress=0.20,
            message="Mining research gaps from selected papers...",
        )
        gaps = mine_research_gaps(topic, max_papers=max_papers, pre_fetched_papers=papers)
        _write_json(_result_path(task_id, "gaps"), gaps)

        # 2) ideas
        _update_status(
            task_id,
            status="idea_generation",
            stage="idea_generation",
            progress=0.45,
            message="Generating ideas from gaps...",
        )
        ideas = generate_ideas_from_gaps(gaps)
        _write_json(_result_path(task_id, "ideas"), ideas)

        # 3) evaluation (+ top idea)
        _update_status(
            task_id,
            status="idea_evaluation",
            stage="idea_evaluation",
            progress=0.60,
            message="Evaluating and scoring ideas...",
        )
        evaluation = evaluate_ideas(ideas)
        _write_json(_result_path(task_id, "evaluation"), evaluation)

        top_idea = select_top_idea(ideas, evaluation)
        _write_json(_result_path(task_id, "top_idea"), top_idea)

        # 4) supporting papers
        _update_status(
            task_id,
            status="support_paper_retrieval",
            stage="support_paper_retrieval",
            progress=0.72,
            message="Retrieving supporting papers for top idea...",
        )
        support = fetch_supporting_papers_for_idea(topic, top_idea, max_papers=6)

        # FIX: always persist as an object with a predictable field for frontend rendering
        _write_json(_result_path(task_id, "supporting_papers"), {"papers": support})

        # 5) draft
        _update_status(
            task_id,
            status="paper_drafting",
            stage="paper_drafting",
            progress=0.85,
            message="Drafting paper...",
        )
        draft = generate_paper_draft(topic, top_idea, support)
        _write_json(_result_path(task_id, "draft"), draft)

        _update_status(task_id, status="completed", stage="completed", progress=1.0, message="Completed", topic=topic)

    except Exception as e:
        d = _task_dir(task_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(str(e), encoding="utf-8")
        _update_status(task_id, status="failed", stage="failed", progress=1.0, message=f"Failed: {type(e).__name__}", topic=topic)


@app.post("/api/tasks", response_model=CreateTaskResponse)
def create_task(req: CreateTaskRequest, bg: BackgroundTasks):
    task_id = f"task_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d = _task_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)

    _write_json(
        d / "meta.json",
        {"task_id": task_id, "topic": req.topic, "max_papers": req.max_papers, "created_at": time.time()},
    )
    _update_status(task_id, status="queued", stage="queued", progress=0.0, message="Queued", topic=req.topic)
    bg.add_task(_pipeline_worker, task_id, req.topic, req.max_papers)
    return CreateTaskResponse(task_id=task_id, status="queued")


@app.get("/api/tasks", response_model=List[TaskListItem])
def list_tasks():
    items: List[TaskListItem] = []
    if not TASKS_DIR.exists():
        return items

    for d in sorted(TASKS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        status_path = d / "status.json"
        if not meta_path.exists() or not status_path.exists():
            continue
        meta = _read_json(meta_path)
        status = _read_json(status_path)
        try:
            items.append(
                TaskListItem(
                    task_id=meta["task_id"],
                    topic=meta.get("topic", ""),
                    status=status.get("status", "queued"),
                    created_at=float(meta.get("created_at", 0.0)),
                )
            )
        except Exception:
            continue
    return items


@app.get("/api/tasks/{task_id}/status", response_model=TaskStatusResponse)
def get_status(task_id: str):
    status_path = _task_dir(task_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="task not found")
    s = _read_json(status_path)
    return TaskStatusResponse(**s)


@app.get("/api/tasks/{task_id}/result/{stage}")
def get_result(task_id: str, stage: ResultStage):
    d = _task_dir(task_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="task not found")

    path = _result_path(task_id, stage)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"result not found: {stage}")
    return _read_json(path)


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    d = _task_dir(task_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="task not found")
    shutil.rmtree(d)
    return {"ok": True, "task_id": task_id}
