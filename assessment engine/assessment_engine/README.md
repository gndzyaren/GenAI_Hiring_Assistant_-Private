# 🧠 GenAI Assessment Engine

An intelligent, adaptive assessment system powered by **local HuggingFace models**. No external API keys required.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│   Home → JD Setup → Adaptive Assessment → Report Dashboard  │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP (httpx)
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Backend                             │
│  /api/assessment/create  →  /next-question  →  /submit      │
└──────┬──────────────┬──────────────────┬────────────────────┘
       │              │                  │
  JD Analyzer    CAT Engine         Dedup Service
  (keyword NLP)  (2PL IRT MLE)     (FAISS + MiniLM)
       │
  Question Generator
  (Mistral-7B-Instruct + CodeGen-350M)
```

---

## 🤖 Models Used (Auto-downloaded from HuggingFace Hub)

| Model | Purpose | Size |
|-------|---------|------|
| `mistralai/Mistral-7B-Instruct-v0.2` | MCQ + Numerical question generation | ~14 GB (fp16) |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic deduplication embeddings | ~90 MB |
| `Salesforce/codegen-350M-mono` | Coding problem generation | ~700 MB |

> **CPU Note:** Mistral-7B is large. On CPU, generation will be slow (1-5 min per question).  
> **GPU recommended:** 16GB VRAM for Mistral-7B in fp16, or 6GB with 4-bit quantization (auto-enabled).  
> **Alternative:** Edit `.env` to use a smaller model like `microsoft/phi-2` for faster CPU inference.

---

## 📁 Project Structure

```
assessment_engine/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── models/
│   │   └── schemas.py             # Pydantic data models
│   ├── routers/
│   │   └── assessment.py          # REST API endpoints
│   └── services/
│       ├── model_loader.py        # HuggingFace model management
│       ├── jd_analyzer.py         # JD → domain/topic extraction
│       ├── question_generator.py  # Mistral + CodeGen generation
│       ├── deduplication.py       # FAISS semantic dedup
│       ├── cat_engine.py          # IRT 2PL adaptive testing
│       └── orchestrator.py        # Pipeline coordinator
├── frontend/
│   └── app.py                     # Streamlit multi-page UI
├── data/
│   └── questions_db/              # Persisted FAISS index (auto-created)
├── run_backend.py                 # Start FastAPI server
├── run_frontend.py                # Start Streamlit app
├── requirements.txt
└── .env                           # Configuration
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- 16 GB RAM minimum (for Mistral-7B on CPU)
- GPU with 16GB VRAM recommended (CUDA 11.8+)

### 2. Install dependencies
```bash
cd assessment_engine
pip install -r requirements.txt
```

### 3. (Optional) Configure models in `.env`
```bash
# For lighter CPU usage, change to:
QUESTION_GEN_MODEL=microsoft/phi-2
# or
QUESTION_GEN_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 4. Run the backend (Terminal 1)
```bash
python run_backend.py
```
Backend starts at: `http://127.0.0.1:8000`  
API docs: `http://127.0.0.1:8000/docs`

### 5. Run the frontend (Terminal 2)
```bash
python run_frontend.py
```
Opens at: `http://localhost:8501`

---

## 🎯 Features

### ✅ Automated Question Generation
- MCQ, Numerical, and Coding formats
- Structured prompt engineering for Mistral-7B
- Fallback template bank when model is unavailable

### ✅ Computer Adaptive Testing (CAT)
- **2-Parameter Logistic IRT Model** for ability estimation
- **Newton-Raphson MLE** for real-time θ (theta) updates
- Questions selected by maximum Fisher information
- Smooth difficulty transitions (easy → medium → hard)

### ✅ Non-Repetition (Semantic Dedup)
- `all-MiniLM-L6-v2` embeddings (384-dim)
- FAISS flat cosine similarity index
- Configurable threshold (default: 85% similarity = duplicate)
- Persisted across sessions (questions_db/)

### ✅ JD-Driven Domain Detection
- Keyword → domain mapping for 15+ engineering domains
- Electrical, Mechanical, Civil, Software/CS
- Auto-detects programming language requirements

### ✅ Assessment Report
- IRT ability score (θ) with level classification
- Section-wise accuracy breakdown
- Adaptive difficulty path visualization
- Strengths, improvement areas, recommendations
- Downloadable JSON report

---

## 📊 Adaptive Testing Logic

```
Candidate starts at θ = 0.0 (medium difficulty)

After each response:
  θ_new = MLE(all responses, IRT 2PL model)
  
  if θ >= 1.0:  → next question = HARD
  if θ >= -0.5: → next question = MEDIUM
  else:          → next question = EASY

Questions selected by: argmax(Fisher_Info(θ, a, b))
```

---

## 🔧 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/assessment/create` | POST | Generate assessment from JD |
| `/api/assessment/{id}/next-question` | GET | Get next adaptive question |
| `/api/assessment/{id}/submit-answer` | POST | Submit answer, get feedback |
| `/api/assessment/{id}/report` | GET | Get full assessment report |
| `/api/assessment/{id}/status` | GET | Get session progress |
| `/health` | GET | Backend health check |

---

## 💡 Tips

- **First run:** Model download takes 10-20 minutes depending on your internet speed
- **GPU users:** Install `bitsandbytes` for 4-bit quantization (`pip install bitsandbytes`)
- **Offline use:** Models are cached in `~/.cache/huggingface/hub/` after first download
- **Customize:** Edit `SIMILARITY_THRESHOLD` in `.env` to tune deduplication strictness
