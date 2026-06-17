"""
Test Suite — Answer Evaluation Engine
Covers: valid answers, edge cases, all 3 domains (DSA, DBMS, OS)
Run with: pytest tests/ -v
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.evaluator import (
    is_empty_or_blank,
    is_too_short,
    is_gibberish,
    detect_domain,
    keyword_relevance_score,
    rule_based_check,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def mock_llm_good():
    """Mock LLM returning a high-score response."""
    return {
        "score": 8.0,
        "feedback": "Good answer covering the main concepts accurately.",
        "confidence": 0.85,
    }

@pytest.fixture
def mock_llm_poor():
    """Mock LLM returning a low-score response."""
    return {
        "score": 2.0,
        "feedback": "The answer is incorrect and misses key concepts.",
        "confidence": 0.90,
    }


# ──────────────────────────────────────────────
# Unit Tests — Rule-Based Helpers
# ──────────────────────────────────────────────

class TestRuleBasedHelpers:

    def test_empty_string_is_empty(self):
        assert is_empty_or_blank("") is True

    def test_whitespace_only_is_empty(self):
        assert is_empty_or_blank("   ") is True

    def test_none_is_empty(self):
        assert is_empty_or_blank(None) is True

    def test_valid_answer_is_not_empty(self):
        assert is_empty_or_blank("Binary search runs in O(log n)") is False

    def test_short_answer_detected(self):
        assert is_too_short("Yes") is True
        assert is_too_short("I don't know") is False

    def test_gibberish_all_numbers(self):
        assert is_gibberish("1234567890!@#$") is True

    def test_gibberish_repeated_char(self):
        assert is_gibberish("aaaaaaaaaa") is True

    def test_gibberish_valid_text(self):
        assert is_gibberish("A stack follows LIFO order.") is False

    def test_detect_dsa_domain(self):
        q = "What is the time complexity of merge sort?"
        a = "Merge sort runs in O(n log n) time using divide and conquer."
        assert detect_domain(q, a) == "dsa"

    def test_detect_dbms_domain(self):
        q = "What is normalization in databases?"
        a = "Normalization reduces redundancy using 1NF, 2NF, and 3NF."
        assert detect_domain(q, a) == "dbms"

    def test_detect_os_domain(self):
        q = "Explain deadlock in operating systems."
        a = "Deadlock occurs when processes wait for each other's resources."
        assert detect_domain(q, a) == "os"

    def test_keyword_relevance_high(self):
        q = "What is a binary search tree?"
        a = "A binary search tree is a tree data structure where each node has at most two children."
        score = keyword_relevance_score(q, a)
        assert score >= 0.5

    def test_keyword_relevance_low_irrelevant_answer(self):
        q = "What is a binary search tree?"
        a = "The weather today is sunny and warm in Delhi."
        score = keyword_relevance_score(q, a)
        assert score < 0.5


# ──────────────────────────────────────────────
# Unit Tests — Rule-Based Short Circuits
# ──────────────────────────────────────────────

class TestRuleBasedCheck:

    def test_empty_answer_returns_score_zero(self):
        result = rule_based_check("What is a stack?", "")
        assert result is not None
        assert result["score"] == 0.0
        assert result["confidence"] == 1.0
        assert result["rule_triggered"] == "empty_answer"

    def test_gibberish_returns_score_zero(self):
        result = rule_based_check("What is a stack?", "!@#$%^&*()")
        assert result is not None
        assert result["score"] == 0.0
        assert result["rule_triggered"] == "gibberish"

    def test_too_short_returns_low_score(self):
        result = rule_based_check("What is a stack?", "No")
        assert result is not None
        assert result["score"] == 1.0
        assert result["rule_triggered"] == "too_short"

    def test_valid_answer_passes_to_llm(self):
        result = rule_based_check(
            "What is a stack?",
            "A stack is a LIFO data structure supporting push and pop operations."
        )
        assert result is None  # None means: pass to LLM


# ──────────────────────────────────────────────
# Integration Tests — API Endpoint
# ──────────────────────────────────────────────

@pytest.mark.asyncio
class TestEvaluateEndpoint:

    async def test_empty_answer_no_llm_call(self):
        """Empty answer must return score=0 without calling LLM."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "What is a binary tree?",
                "answer": ""
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 0.0
        assert data["confidence"] == 1.0
        assert "No answer" in data["feedback"]

    async def test_whitespace_answer_score_zero(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "Explain paging.",
                "answer": "     "
            })
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.0

    async def test_gibberish_answer_score_zero(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "Explain deadlock.",
                "answer": "!@#$%^&*()1234"
            })
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.0

    async def test_too_short_answer_low_score(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "Explain normalization.",
                "answer": "yes"
            })
        assert resp.status_code == 200
        assert resp.json()["score"] <= 2.0

    async def test_missing_question_field(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={"answer": "Some answer"})
        assert resp.status_code == 422

    async def test_missing_answer_field(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={"question": "What is a stack?"})
        assert resp.status_code == 422

    async def test_empty_question_rejected(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "",
                "answer": "A stack uses LIFO."
            })
        assert resp.status_code == 422

    # ── Tests with mocked LLM (no real API call) ──

    @patch("app.evaluator.llm_evaluate", new_callable=AsyncMock)
    async def test_dsa_good_answer(self, mock_llm):
        mock_llm.return_value = {
            "score": 8.5,
            "feedback": "Correct explanation of binary search with time complexity.",
            "confidence": 0.88,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "Explain binary search and its time complexity.",
                "answer": (
                    "Binary search works on sorted arrays by repeatedly halving the search space. "
                    "It compares the target with the middle element. If equal, it returns the index. "
                    "If target is smaller, it searches the left half; otherwise the right half. "
                    "Time complexity is O(log n) and space complexity is O(1) for iterative version."
                )
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] >= 7.0
        assert data["confidence"] >= 0.7
        assert len(data["feedback"]) > 10

    @patch("app.evaluator.llm_evaluate", new_callable=AsyncMock)
    async def test_dbms_good_answer(self, mock_llm):
        mock_llm.return_value = {
            "score": 7.5,
            "feedback": "Good explanation of ACID properties with correct definitions.",
            "confidence": 0.82,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "What are ACID properties in database transactions?",
                "answer": (
                    "ACID stands for Atomicity, Consistency, Isolation, and Durability. "
                    "Atomicity ensures all operations in a transaction succeed or all fail. "
                    "Consistency ensures the database moves from one valid state to another. "
                    "Isolation ensures concurrent transactions don't interfere. "
                    "Durability ensures committed transactions persist even after a crash."
                )
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] >= 6.0

    @patch("app.evaluator.llm_evaluate", new_callable=AsyncMock)
    async def test_os_good_answer(self, mock_llm):
        mock_llm.return_value = {
            "score": 9.0,
            "feedback": "Excellent and thorough explanation of deadlock with all four conditions.",
            "confidence": 0.92,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "What are the four necessary conditions for deadlock?",
                "answer": (
                    "The four necessary conditions for deadlock are: "
                    "1. Mutual Exclusion — at least one resource must be held non-shareable. "
                    "2. Hold and Wait — a process holds resources and waits for more. "
                    "3. No Preemption — resources cannot be forcibly taken from a process. "
                    "4. Circular Wait — a circular chain of processes each waiting for the next."
                )
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] >= 8.0

    @patch("app.evaluator.llm_evaluate", new_callable=AsyncMock)
    async def test_irrelevant_answer_penalised(self, mock_llm):
        """LLM gives 6.0 but keyword relevance is near zero — should be penalised."""
        mock_llm.return_value = {
            "score": 6.0,
            "feedback": "The answer does not relate to the question asked.",
            "confidence": 0.75,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "Explain the concept of virtual memory in OS.",
                "answer": "I really enjoy cooking pasta and trying new Italian recipes every weekend."
            })
        assert resp.status_code == 200
        data = resp.json()
        # Score should be reduced due to irrelevance penalty
        assert data["score"] <= 5.0

    @patch("app.evaluator.llm_evaluate", new_callable=AsyncMock)
    async def test_partial_answer_mid_score(self, mock_llm):
        mock_llm.return_value = {
            "score": 5.0,
            "feedback": "Partially correct but missing key details about implementation.",
            "confidence": 0.72,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "What is a hash table and how does it handle collisions?",
                "answer": "A hash table stores key-value pairs and is very fast."
            })
        assert resp.status_code == 200
        data = resp.json()
        assert 3.0 <= data["score"] <= 7.0

    async def test_health_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_response_schema_valid(self):
        """Ensure response always includes score, feedback, confidence."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/evaluate", json={
                "question": "What is paging?",
                "answer": ""
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "feedback" in data
        assert "confidence" in data
        assert isinstance(data["score"], (int, float))
        assert isinstance(data["feedback"], str)
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["score"] <= 10.0
        assert 0.0 <= data["confidence"] <= 1.0
