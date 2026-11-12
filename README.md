# 🩺 Medical RAG Chatbot

**🔗 Live demo:** https://sammi-medical-rag.streamlit.app

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a 759-page medical encyclopedia. Answers are grounded in retrieved text and returned with source page numbers, and the bot refuses questions outside its knowledge base.

The project includes a **FastAPI backend**, a **Streamlit frontend**, a **Docker** setup, and a **custom evaluation pipeline** used to tune retrieval.

---

## 🏗️ Architecture

```
Streamlit UI  ──►  FastAPI  (POST /ask)  ──►  RAG pipeline
                                               ├─ Embeddings: all-MiniLM-L6-v2
                                               ├─ Vector search: FAISS (top-5 chunks)
                                               └─ LLM: Groq (gpt-oss-120b)
```

- **Locally / Docker:** the Streamlit UI calls the FastAPI backend over HTTP.
- **Live demo:** Streamlit Community Cloud runs a single process, so the UI calls the RAG pipeline directly (`USE_API=false`).

---

## 🔌 API

`POST /ask`

```json
// Request
{ "question": "What are the symptoms of diabetes?" }

// Response
{
  "answer": "The symptoms of diabetes include: ...",
  "sources": ["Page 437", "Page 436", "Page 445", "Page 439"]
}
```

If the answer is not in the knowledge base, the API returns `"I don't know."` with an empty `sources` list. `GET /health` returns the service status.

---

## 📊 Evaluation

A test set of **41 questions** (`test_questions.csv`): 36 in-scope questions with reference answers and source pages, plus 5 out-of-scope questions to check that the bot refuses instead of hallucinating.

| Configuration | Retrieval Hit Rate | Answer Accuracy | Out-of-Scope Refusals |
|---|---|---|---|
| chunk 500, top-k 3 (baseline) | 92% | 75% | 100% |
| **chunk 500, top-k 5 (final)** | **94%** | **86%** | **100%** |
| chunk 1000, top-k 3 | 89% | 86% | 100% |

**Key findings**
- Most baseline failures came from chunk boundaries: the right page was retrieved, but not the chunk containing the answer.
- Increasing retrieval depth (top-k 3 → 5) raised accuracy from 75% to 86% without hurting grounding.
- Answer accuracy was scored with an LLM-as-a-judge and spot-checked manually; hit rate is measured at page level, so it is somewhat generous.

---

## 🛠️ Tech Stack

Python · LangChain · FAISS · Hugging Face (sentence-transformers) · Groq · FastAPI · Pydantic · Streamlit · Docker

---

## 🚀 Run Locally

```bash
git clone https://github.com/sammi-123456789/medical-rag-chatbot.git
cd medical-rag-chatbot
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

Start the backend and the frontend in two terminals:

```bash
uvicorn app.main:app --reload     # API docs at http://127.0.0.1:8000/docs
streamlit run medibot.py          # Chat UI at http://localhost:8501
```

## 🐳 Run with Docker

```bash
docker build -t medical-rag .
docker run --env-file .env -p 7860:7860 -p 8000:8000 medical-rag
```

- Chat UI: http://localhost:7860
- API docs: http://localhost:8000/docs

The API key is passed at runtime with `--env-file` and is excluded from the image via `.dockerignore`.

---

## 📁 Project Structure

```
├── app/
│   ├── main.py                  # FastAPI app (/ask, /health)
│   └── rag.py                   # RAG pipeline (FAISS + Groq)
├── medibot.py                   # Streamlit chat UI
├── create_memory_for_llm.py     # PDF → chunks → embeddings → FAISS index
├── evaluate.py                  # Evaluation pipeline
├── test_questions.csv           # 41-question test set
├── eval_results_*.csv           # Per-question evaluation results
├── vector/db_faiss/             # Prebuilt FAISS index
├── Dockerfile
└── requirements.txt
```

---

## 🔮 Future Improvements

- Pass chat history to the model to handle follow-up questions
- Evaluate retrieval at chunk level and expand the test set
- Add automated tests for the API

---

⚠️ *This project is for educational purposes only and is not a substitute for professional medical advice.*