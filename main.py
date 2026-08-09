"""
main.py — REST API backend
----------------------------
Exposes the existing ResumeScreeningPipeline (built in Module 7) over
HTTP so a custom HTML/JS or React frontend can call it, instead of using
the Streamlit UI. This does not replace streamlit_app/ — both talk to
the same pipeline; pick whichever frontend you want to run.

Run with:  uvicorn api.main:app --reload --port 8000   (from project root)

Endpoints:
  GET  /health              -> {"status": "ok", "semantic_backend": "..."}
  POST /analyze              -> multipart form: job_title, job_description,
                                 resumes (one or more files) -> ranked results as JSON
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prediction.prediction_pipeline import ResumeScreeningPipeline
from utils.file_parser import extract_text

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="AI Resume Screening API", version="1.0")

# Wide-open CORS so a plain static HTML file (opened via file:// or a
# separate dev server on another port) can call this API during
# development. Tighten `allow_origins` to your real frontend's domain
# before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline() -> ResumeScreeningPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ResumeScreeningPipeline(
            role_classifier_path=str(ROOT / "saved_models" / "role_classifier_tuned.joblib")
        )
    return _pipeline


class ResumeResult(BaseModel):
    resume_id: str
    final_score: float
    semantic_similarity: float
    skill_overlap_percent: float
    matching_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    explanation: str
    predicted_category: str
    category_confidence: float


class AnalyzeResponse(BaseModel):
    job_title: str
    winner_id: str
    winner_reason: str
    rankings: List[ResumeResult]
    semantic_backend: str


@app.get("/health")
def health():
    pipeline = get_pipeline()
    return {"status": "ok", "semantic_backend": pipeline.semantic.backend}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    job_title: str = Form(...),
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...),
):
    if not job_description.strip():
        raise HTTPException(400, "job_description is required")
    if not resumes:
        raise HTTPException(400, "at least one resume file is required")

    resume_texts = {}
    for f in resumes:
        raw = await f.read()
        try:
            text = extract_text(f.filename, raw)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if text.strip():
            resume_texts[f.filename] = text

    if not resume_texts:
        raise HTTPException(400, "no readable text extracted from the uploaded resumes")

    pipeline = get_pipeline()
    result = pipeline.run(job_description, resume_texts, job_title=job_title)

    return AnalyzeResponse(
        job_title=result.job_title,
        winner_id=result.winner_id,
        winner_reason=result.winner_reason,
        semantic_backend=pipeline.semantic.backend,
        rankings=[
            ResumeResult(
                resume_id=r.resume_id,
                final_score=r.final_score,
                semantic_similarity=r.semantic_similarity,
                skill_overlap_percent=r.skill_overlap_percent,
                matching_skills=r.matching_skills,
                missing_skills=r.missing_skills,
                extra_skills=r.extra_skills,
                explanation=r.explanation,
                predicted_category=r.predicted_category,
                category_confidence=r.category_confidence,
            )
            for r in result.rankings
        ],
    )
