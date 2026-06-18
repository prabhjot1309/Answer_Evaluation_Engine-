"""
Answer Evaluation Engine — Hybrid Rule-Based + LLM Evaluator
"""

import re
import json
import httpx
from typing import Optional

# ──────────────────────────────────────────────
# Domain keyword sets for relevance pre-check
# ──────────────────────────────────────────────
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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

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

def detect_domain(question: str, answer: str) -> Optional[str]:
    combined = (question + " " + answer).lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

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
            "feedback": "The answer appears to be gibberish or random characters. Please provide a meaningful response.",
            "confidence": 0.95,
            "rule_triggered": "gibberish",
        }
    if is_too_short(answer, min_words=3):
        return {
            "score": 1.0,
            "feedback": "The answer is too brief to evaluate meaningfully. Please elaborate with more detail.",
            "confidence": 0.9,
            "rule_triggered": "too_short",
        }
    return None

# ──────────────────────────────────────────────
# LLM Evaluation
# ──────────────────────────────────────────────

LLM_SYSTEM_PROMPT = """You are a balanced and encouraging technical interview evaluator for Computer Science topics: Data Structures & Algorithms (DSA), Database Management Systems (DBMS), and Operating Systems (OS).

Your job is to evaluate a candidate's answer fairly and generously — reward partial knowledge, not just perfect answers.

Scoring rubric (be generous, not strict):
- 0–2: Completely wrong, totally irrelevant, or shows zero understanding
- 3–4: Has some relevant idea but major errors or very incomplete
- 5–6: Understands the basics but missing important details or depth
- 7–8: Good answer — correct, reasonably complete, minor gaps are fine
- 9–10: Excellent — thorough, accurate, well-explained with examples or nuance

Important guidelines:
1. A short but CORRECT answer still deserves a 7+. Brevity is not a flaw.
2. If the answer captures the core concept correctly, score it 7 or above.
3. Only score below 5 if the answer is actually wrong or irrelevant — not just incomplete.
4. Never hallucinate. Base feedback ONLY on what the candidate wrote.
5. Feedback must be 2–3 sentences: what they got right, what could be improved.
6. Confidence: how certain you are about the score (0.0–1.0).

Respond ONLY with valid JSON — no markdown, no extra text:
{
  "score": <float 0.0 to 10.0, one decimal place>,
  "feedback": "<2-3 sentence feedback>",
  "confidence": <float 0.0 to 1.0, two decimal places>
}"""


async def llm_evaluate(question: str, answer: str) -> dict:
    user_message = f"""Question: {question}

Candidate's Answer: {answer}

Evaluate the answer and respond with the JSON object."""

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": LLM_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

    raw_text = data["content"][0]["text"].strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    result = json.loads(raw_text)

    score = max(0.0, min(10.0, float(result["score"])))
    score = round(score * 2) / 2   # snap to nearest 0.5
    confidence = max(0.0, min(1.0, float(result["confidence"])))
    feedback = str(result["feedback"]).strip()

    return {"score": score, "feedback": feedback, "confidence": confidence}


# ──────────────────────────────────────────────
# Hybrid Evaluation Orchestrator
# ──────────────────────────────────────────────

async def evaluate(question: str, answer: str) -> dict:
    # Step 1: Rule-based fast path
    rule_result = rule_based_check(question, answer)
    if rule_result:
        return {
            "score": rule_result["score"],
            "feedback": rule_result["feedback"],
            "confidence": rule_result["confidence"],
        }

    # Step 2: Keyword relevance signal
    kw_relevance = keyword_relevance_score(question, answer)

    # Step 3: LLM evaluation
    llm_result = await llm_evaluate(question, answer)

    final_score = llm_result["score"]
    final_confidence = llm_result["confidence"]

    # Step 4: Only penalise if relevance is VERY low (< 0.05) AND score is high
    # This only catches truly off-topic answers, not short/simple ones
    if kw_relevance < 0.05 and final_score > 6.0:
        final_score = max(final_score * 0.8, 4.0)
        final_confidence = round(final_confidence * 0.85, 2)

    return {
        "score": round(final_score, 1),
        "feedback": llm_result["feedback"],
        "confidence": round(final_confidence, 2),
    }
