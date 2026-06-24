# 🧠 Answer Evaluation Engine

> An AI-powered automated answer evaluation system that scores candidate responses with meaningful feedback — built for interview and assessment platforms.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker)](https://www.docker.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=flat&logo=render)](https://answer-evaluation-engine.onrender.com)

---

## 📌 Overview

The **Answer Evaluation Engine** is a core module for scoring candidate answers in interview and assessment systems. Given a question and a candidate's answer, it returns a **score (0–10)**, **actionable feedback**, and a **confidence value** — using a hybrid approach that combines rule-based checks with LLM-powered semantic evaluation.

Supports multiple technical domains out of the box: **DSA**, **DBMS**, and **Operating Systems**.

---

## 🚀 Live API

**Base URL:** [`https://answer-evaluation-engine.onrender.com`](https://answer-evaluation-engine.onrender.com)

| Endpoint | Method | Description |
|---|---|---|
| `/evaluate` | `POST` | Evaluate a candidate's answer |
| `/health` | `GET` | Health check |
| `/docs` | `GET` | Interactive Swagger UI |

---

## ⚙️ How It Works

The engine uses a **hybrid evaluation pipeline**:

1. **Rule-based checks** — Detect empty answers, gibberish, and off-topic responses instantly (no LLM call needed).
2. **LLM evaluation** — For valid answers, the question + answer are passed to a language model which assesses correctness, completeness, and relevance.
3. **Scoring** — A normalized score from 0–10 is produced alongside a confidence value (0–1) indicating how certain the evaluator is.

```
Input ──► Rule Filter ──► LLM Evaluator ──► Score + Feedback + Confidence
               │
               └──► Empty / Irrelevant ──► Score: 0, immediate return
```

---

## 📥 API Reference

### `POST /evaluate`

**Request Body:**

```json
{
  "question": "What is a binary search tree?",
  "answer": "A BST is a tree where each node has at most two children, and the left child is smaller than the parent while the right child is larger."
}
```

**Response:**

```json
{
  "score": 7.5,
  "feedback": "Good explanation of BST structure and ordering property. Consider also mentioning time complexity (O(log n) average) and use cases like sorted data retrieval.",
  "confidence": 0.85
}
```

**Field descriptions:**

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Answer quality score from 0.0 to 10.0 |
| `feedback` | `string` | Specific, actionable feedback for the candidate |
| `confidence` | `float` | Evaluator confidence from 0.0 to 1.0 |

---

## 🧪 Edge Cases & Behavior

| Scenario | Behavior |
|---|---|
| Empty answer (`""`) | Returns `score: 0`, no LLM call made |
| Irrelevant / off-topic answer | Detected and penalized; low score with feedback |
| Partially correct answer | Scored proportionally with targeted feedback |
| Vague but related answer | Mid-range score with suggestions to elaborate |

---

## 🏗️ Project Structure

```
Answer_Evaluation_Engine/
├── app/
│   ├── main.py            # FastAPI app entry point
│   ├── evaluator.py       # Core hybrid evaluation logic
│   ├── models.py          # Pydantic request/response models
│   └── prompts.py         # LLM prompt templates
├── tests/
│   ├── test_evaluate.py   # Valid + invalid test cases
│   └── test_edge_cases.py # Empty, irrelevant, domain tests
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| FastAPI | 0.115.0 | Web framework & API layer |
| Uvicorn | 0.30.0 | ASGI server |
| Pydantic | 2.9.2 | Request/response validation |
| HTTPX | 0.27.0 | Async HTTP client (LLM calls) |
| Pytest | 8.2.0 | Testing framework |
| pytest-asyncio | 0.23.7 | Async test support |
| Docker | — | Containerization |

---

## 📦 Local Setup

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repo

```bash
git clone https://github.com/prabhjot1309/Answer_Evaluation_Engine-.git
cd Answer_Evaluation_Engine-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the root directory:

```env
# Add your LLM API key if required
OPENAI_API_KEY=your_api_key_here
# or
ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

API is now available at `http://localhost:8000`

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 🐳 Docker

### Build and run

```bash
docker build -t answer-eval-engine .
docker run -p 8000:8000 answer-eval-engine
```

---

## ✅ Running Tests

```bash
pytest
```

Tests cover:

- ✅ Valid answers across DSA, DBMS, and OS domains
- ✅ Empty answer → score of 0
- ✅ Completely irrelevant answers
- ✅ Partially correct answers
- ✅ Response schema validation

---

## 📚 Supported Domains

The engine is domain-agnostic but has been validated against:

| Domain | Example Topics |
|---|---|
| **DSA** | Arrays, Trees, Graphs, Sorting, Dynamic Programming |
| **DBMS** | Normalization, Indexing, Transactions, SQL, ACID |
| **OS** | Scheduling, Deadlocks, Memory Management, Paging |

---

## 🗺️ Roadmap

- [ ] Multi-language support (Python, Java code answers)
- [ ] Batch evaluation endpoint (`/evaluate/batch`)
- [ ] Domain-specific scoring rubrics
- [ ] Confidence calibration improvements
- [ ] Admin dashboard for evaluation history

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a pull request.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Prabhjot Singh**
GitHub: [@prabhjot1309](https://github.com/prabhjot1309)
