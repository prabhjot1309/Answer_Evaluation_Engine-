"""
Answer Evaluation Engine — FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, field_validator
import httpx
import os

from app.evaluator import evaluate

app = FastAPI(
    title="Answer Evaluation Engine",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── index.html sits at /app/index.html inside the container ──
HTML_PATH = "/app/index.html"

def load_html():
    with open(HTML_PATH, "r") as f:
        return f.read()

# ── Models ──
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

# ── Endpoints ──

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=load_html())

@app.get("/health")
async def health():
    return {"status": "ok", "service": "answer-evaluation-engine"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(req: EvaluationRequest):
    try:
        result = await evaluate(req.question, req.answer)
        return EvaluationResponse(**result)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timed out. Please retry.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {str(exc)}"})
