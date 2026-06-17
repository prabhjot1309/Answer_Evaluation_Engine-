"""
Answer Evaluation Engine — FastAPI Application
Render-ready: respects $PORT, CORS enabled
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import httpx

from app.evaluator import evaluate

app = FastAPI(
    title="Answer Evaluation Engine",
    description="Automated scoring for technical interview answers (DSA, DBMS, OS)",
    version="1.0.0",
)

# ── CORS — allow any origin (needed if a frontend calls this API) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class EvaluationRequest(BaseModel):
    question: str
    answer: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @field_validator("answer")
    @classmethod
    def answer_strip(cls, v):
        return v.strip() if v else ""


class EvaluationResponse(BaseModel):
    score: float
    feedback: str
    confidence: float


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Render uses this path to confirm the service is up."""
    return {"status": "ok", "service": "answer-evaluation-engine"}


@app.get("/")
async def root():
    """Root endpoint — useful for quick browser check."""
    return {
        "service": "Answer Evaluation Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(req: EvaluationRequest):
    """
    Evaluate a candidate's answer to a technical question.

    - **question**: The interview/assessment question.
    - **answer**: The candidate's response.

    Returns a **score** (0–10), **feedback** string, and **confidence** (0–1).
    """
    try:
        result = await evaluate(req.question, req.answer)
        return EvaluationResponse(**result)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API error: {exc.response.status_code} — {exc.response.text}",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timed out. Please retry.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


# ── Global error handler ──
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected error: {str(exc)}"},
    )
