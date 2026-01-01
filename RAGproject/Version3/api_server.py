# api_server.py
import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
import shutil

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from gap_miner import mine_research_gaps
from idea_generator import generate_ideas_from_gaps
from idea_evaluator import evaluate_ideas
from paper_draft_writer import select_top_idea, fetch_supporting_papers_for_idea, generate_paper_draft

TASKS_DIR = Path("tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)

Stage = Literal["queued", "gap_mining", "idea_generation", "idea_evaluation", "paper_drafting", "completed", "failed"]
ResultStage = Literal["gaps", "ideas", "evaluation", "draft"]

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

class TaskListItem(BaseModel):
    task_id: str
    topic: str
    status: Stage
    created_at: float

app = FastAPI(title="RAG Research Pipeline API", version="0.1")
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

def _task_dir(task_id: str) -> Path:
    return TASKS_DIR / task_id

def _write_json(path: Path, data: Any):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _update_status(task_id: str, *, status: Stage, stage: Stage, progress: float, message: str, topic: Optional[str] = None):
    d = _task_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)
    status_path = d / "status.json"
    payload = {
        "task_id": task_id,
        "status": status,
        "stage": stage,
        "progress": float(max(0.0, min(1.0, progress))),
        "message": message,
    }
    if topic is not None:
        payload["topic"] = topic
    _write_json(status_path, payload)

def _pipeline_worker(task_id: str, topic: str, max_papers: int):
    try:
        _update_status(task_id, status="queued", stage="queued", progress=0.0, message="Queued", topic=topic)

        # 1) gaps
        _update_status(task_id, status="gap_mining", stage="gap_mining", progress=0.05, message="Mining research gaps...")
        gaps = mine_research_gaps(topic, max_papers=max_papers)
        _write_json(_task_dir(task_id) / "gaps.json", gaps)

        # 2) ideas
        _update_status(task_id, status="idea_generation", stage="idea_generation", progress=0.35, message="Generating ideas...")
        ideas = generate_ideas_from_gaps(gaps)
        _write_json(_task_dir(task_id) / "ideas.json", ideas)

        # 3) evaluation
        _update_status(task_id, status="idea_evaluation", stage="idea_evaluation", progress=0.55, message="Evaluating ideas...")
        evaluation = evaluate_ideas(ideas)
        _write_json(_task_dir(task_id) / "evaluation.json", evaluation)

        # 4) draft
        _update_status(task_id, status="paper_drafting", stage="paper_drafting", progress=0.75, message="Drafting paper...")
        top_idea = select_top_idea(ideas, evaluation)
        support = fetch_supporting_papers_for_idea(topic, top_idea, max_papers=6)
        draft = generate_paper_draft(topic, top_idea, support)
        _write_json(_task_dir(task_id) / "draft.json", draft)

        _update_status(task_id, status="completed", stage="completed", progress=1.0, message="Completed")

    except Exception as e:
        d = _task_dir(task_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(str(e), encoding="utf-8")
        _update_status(task_id, status="failed", stage="failed", progress=1.0, message=f"Failed: {type(e).__name__}")

@app.post("/api/tasks", response_model=CreateTaskResponse)
def create_task(req: CreateTaskRequest, bg: BackgroundTasks):
    task_id = f"task_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d = _task_dir(task_id)
    d.mkdir(parents=True, exist_ok=True)

    _write_json(d / "meta.json", {"task_id": task_id, "topic": req.topic, "max_papers": req.max_papers, "created_at": time.time()})

    bg.add_task(_pipeline_worker, task_id, req.topic, req.max_papers)
    _update_status(task_id, status="queued", stage="queued", progress=0.0, message="Queued", topic=req.topic)
    return CreateTaskResponse(task_id=task_id, status="queued")

@app.get("/api/tasks/{task_id}/status", response_model=TaskStatusResponse)
def get_status(task_id: str):
    status_path = _task_dir(task_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="task not found")
    s = _read_json(status_path)
    return TaskStatusResponse(**s)

@app.get("/api/tasks/{task_id}/result/{stage}")
def get_result(task_id: str, stage: ResultStage):
    p = _task_dir(task_id) / f"{stage}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{stage} not ready")
    return _read_json(p)
@app.get("/")
def read_root():
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/ui/index.html")

@app.get("/api/tasks", response_model=List[TaskListItem])
def list_tasks(limit: int = 50):
    items: List[TaskListItem] = []
    for d in sorted(TASKS_DIR.glob("task_*"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        meta = d / "meta.json"
        status = d / "status.json"
        if meta.exists() and status.exists():
            m = _read_json(meta)
            s = _read_json(status)
            items.append(TaskListItem(
                task_id=m["task_id"],
                topic=m["topic"],
                status=s["status"],
                created_at=m["created_at"],
            ))

            @app.delete("/api/tasks/{task_id}")
            def delete_task(task_id: str):
                d = _task_dir(task_id)
                if not d.exists():
                    raise HTTPException(status_code=404, detail="Task not found")
                try:
                    shutil.rmtree(d)
                    return {"status": "success", "message": f"Task {task_id} deleted"}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
    return items
