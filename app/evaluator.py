"""
Answer Evaluation Engine — Hybrid Rule-Based + LLM Evaluator
Uses Google Gemini API (free tier, no credit card needed)
"""

import re
import json
import httpx
from typing import Optional
import os

DOMAIN_KEYWORDS = {
    "dsa": [
        "array", "linked list", "tree", "graph", "stack", "queue", "heap",
        "hash", "sort", "search", "complexity", "big o", "recursion",
        "dynamic programming", "greedy", "binary", "node", "pointer",
        "algorithm", "traversal", "bfs", "dfs", "time complexity", "space complexity",
        "o(n)", "o(log n)", "o(1)", "pivot", "merge sort", "quick sort",
    ],
    "dbms": [
        "database", "sql", "query", "table", "index", "join", "normalization",
        "primary key", "foreign key", "acid", "transaction", "deadlock",
        "schema", "relation", "tuple", "attribute", "view", "trigger",
        "stored procedure", "nosql", "mongodb", "constraint", "er diagram",
        "1nf", "2nf", "3nf", "bcnf", "rollback", "commit", "isolation",
    ],
    "os": [
        "process", "thread", "scheduling", "memory", "virtual memory",
        "paging", "segmentation", "deadlock", "mutex", "semaphore",
        "context switch", "interrupt", "system call", "kernel", "cpu",
        "page fault", "cache", "disk", "file system", "ipc", "pipe",
        "race condition", "critical section", "banker's algorithm",
        "round robin", "fcfs", "sjf", "priority scheduling",
    ],
}

ALL_KEYWORDS = {kw for kws in DOMAIN_KEYWORDS.values() for kw in kws}

# ──────────────────────────────────────────────
# Rule-Based Pre-checks
# ──────────────────────────────────────────────

def is_empty_or_blank(answer: str) -> bool:
    return not answer or not answer.strip()

def is_too_short(answer: str, min_words: int = 3) -> bool:
    return len(answer.strip().split()) < min_words

def is_gibberish(answer: str) -> bool:
    clean = answer.strip()
    if not clean:
        return True
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in clean) / max(len(clean), 1)
    if alpha_ratio < 0.4:
        return True
    if re.fullmatch(r"(.)\1{4,}", clean):
        return True
    return False

def keyword_relevance_score(question: str, answer: str) -> float:
    q_lower = question.lower()
    a_lower = answer.lower()
    q_keywords = [kw for kw in ALL_KEYWORDS if kw in q_lower]
    if not q_keywords:
        return 0.5
    hits = sum(1 for kw in q_keywords if kw in a_lower)
    return min(hits / len(q_keywords), 1.0)

def rule_based_check(question: str, answer: str) -> dict | None:
    if is_empty_or_blank(answer):
        return {
            "score": 0.0,
            "feedback": "No answer was provided. Please write a response to be evaluated.",
            "confidence": 1.0,
            "rule_triggered": "empty_answer",
        }
    if is_gibberish(answer):
        return {
            "score": 0.0,
            "feedback": "The answer appears to be gibberish. Please provide a meaningful response.",
            "confidence": 0.95,
            "rule_triggered": "gibberish",
        }
    if is_too_short(answer, min_words=3):
        return {
            "score": 1.0,
            "feedback": "The answer is too brief. Please elaborate with more detail.",
            "confidence": 0.9,
            "rule_triggered": "too_short",
        }
    return None

# ──────────────────────────────────────────────
# LLM Evaluation — Google Gemini (free)
# ──────────────────────────────────────────────

GEMINI_PROMPT = """You are a balanced technical interview evaluator for Computer Science: DSA, DBMS, and OS.

Scoring rubric (be fair and generous):
- 0-2: Completely wrong or irrelevant
- 3-4: Has some idea but major errors or very incomplete
- 5-6: Understands basics but missing important details
- 7-8: Good answer — correct and reasonably complete (most good answers land here)
- 9-10: Excellent — thorough, accurate, well-explained

Rules:
1. A short but CORRECT answer deserves 7+. Brevity is not a flaw.
2. Only score below 5 if the answer is actually wrong or irrelevant.
3. Feedback: 2-3 sentences — what they got right, what could improve.
4. Never invent facts not in the answer.

Respond ONLY with valid JSON, no markdown:
{
  "score": <float 0.0-10.0>,
  "feedback": "<2-3 sentence feedback>",
  "confidence": <float 0.0-1.0>
}"""


async def llm_evaluate(question: str, answer: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_PROMPT}]},
        "contents": [{
            "parts": [{
                "text": f"Question: {question}\n\nCandidate's Answer: {answer}\n\nEvaluate and respond with JSON."
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 512,
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # Strip markdown fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    result = json.loads(raw_text)

    score = max(0.0, min(10.0, float(result["score"])))
    score = round(score * 2) / 2
    confidence = max(0.0, min(1.0, float(result["confidence"])))
    feedback = str(result["feedback"]).strip()

    return {"score": score, "feedback": feedback, "confidence": confidence}


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

async def evaluate(question: str, answer: str) -> dict:
    rule_result = rule_based_check(question, answer)
    if rule_result:
        return {
            "score": rule_result["score"],
            "feedback": rule_result["feedback"],
            "confidence": rule_result["confidence"],
        }

    kw_relevance = keyword_relevance_score(question, answer)
    llm_result = await llm_evaluate(question, answer)

    final_score = llm_result["score"]
    final_confidence = llm_result["confidence"]

    if kw_relevance < 0.05 and final_score > 6.0:
        final_score = max(final_score * 0.8, 4.0)
        final_confidence = round(final_confidence * 0.85, 2)

    return {
        "score": round(final_score, 1),
        "feedback": llm_result["feedback"],
        "confidence": round(final_confidence, 2),
    }
