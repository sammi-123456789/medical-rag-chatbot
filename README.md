# 🩺 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using content from a 759-page medical encyclopedia (The Gale Encyclopedia of Medicine, Vol. C–F). Answers are grounded in the retrieved text, returned with their source documents, and the bot refuses to answer questions outside its knowledge base.

The project goes beyond a basic RAG setup by including a **custom evaluation pipeline** that measures retrieval and answer quality, and uses those measurements to tune the system.

---

## ⚙️ How It Works

```
Medical PDF ──► Chunking (500 chars, 50 overlap) ──► Embeddings (all-MiniLM-L6-v2) ──► FAISS index
                                                                                          │
User question ──► Embed question ──► Retrieve top-5 chunks ◄──────────────────────────────┘
                                            │
                                            ▼
                     Grounded prompt + Groq LLM (GPT-OSS-120B) ──► Answer + source documents
```

1. **Ingestion** (`create_memory_for_llm.py`): loads the PDF, splits it into overlapping chunks, embeds them with a Hugging Face sentence-transformer, and stores them in a FAISS vector index.
2. **Retrieval + Generation** (`medibot.py`): a Streamlit chat app retrieves the most relevant chunks and passes them to the LLM with a prompt that restricts answers to the provided context.
3. **Evaluation** (`evaluate.py`): runs a curated test set through the pipeline and scores the results.

---

## 📊 Evaluation

A test set of **41 questions** (`test_questions.csv`) was built from the source PDF: 36 in-scope questions with reference answers and source page numbers, plus 5 out-of-scope questions (e.g. sports, programming) to check that the bot refuses instead of hallucinating.

Three metrics are measured:

- **Retrieval hit rate**: did the correct source page appear in the retrieved chunks?
- **Answer accuracy**: is the answer correct compared to the reference? Scored with an LLM-as-a-judge and spot-checked manually.
- **Out-of-scope refusals**: does the bot say "I don't know" for questions outside the document?

### Results

| Configuration | Retrieval Hit Rate | Answer Accuracy | Out-of-Scope Refusals |
|---|---|---|---|
| chunk 500, top-k 3 (baseline) | 92% | 75% | 100% |
| **chunk 500, top-k 5 (final)** | **94%** | **86%** | **100%** |
| chunk 1000, top-k 3 | 89% | 86% | 100% |

### Key Findings

- **Chunk boundaries were the main failure cause.** In the baseline, several questions retrieved the correct page but not the specific chunk containing the answer, so the model correctly replied "I don't know". In one case a sentence was split across two chunks, leaving the answer incomplete.
- **Giving the model more context fixed this.** Increasing retrieval depth (top-k 3 → 5) or chunk size (500 → 1000) both raised accuracy from 75% to 86%. The top-k 5 setup was chosen because it had the highest hit rate and reuses the existing index.
- **Grounding held up.** Out-of-scope refusal stayed at 100% across all configurations, so the extra context did not cause hallucination.
- **Limitations of the metrics.** The hit rate is measured at page level, not chunk level, so it is somewhat generous. The LLM judge occasionally marked correct but less detailed answers as wrong, which is why results were manually spot-checked.

Detailed per-question results are in `eval_results_k5.csv` and `eval_results_c1000_k3.csv`.

---

## 🛠️ Tech Stack

Python · LangChain · FAISS · Hugging Face (sentence-transformers) · Groq · Streamlit · Pandas

---

## 🚀 Run Locally

```bash
git clone https://github.com/sammi-123456789/medical-rag-chatbot.git
cd medical-rag-chatbot
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_api_key
```

Start the chatbot:

```bash
streamlit run medibot.py
```

Run the evaluation:

```bash
python evaluate.py
```

> The source PDF is not included in this repository. To rebuild the index from your own documents, place PDFs in a `data/` folder and run `python create_memory_for_llm.py`.

---

## 📁 Project Structure

```
├── create_memory_for_llm.py     # PDF → chunks → embeddings → FAISS index
├── connect_memory_with_llm.py   # Command-line version of the QA chain
├── medibot.py                   # Streamlit chat app
├── evaluate.py                  # Evaluation pipeline (hit rate, accuracy, refusals)
├── test_questions.csv           # 41-question test set
├── eval_results_*.csv           # Per-question evaluation results
├── vector/db_faiss/             # Prebuilt FAISS index
└── requirements.txt
```

---

## 🔮 Future Improvements

- Pass chat history to the model so it can handle follow-up questions
- Display sources as file name and page number instead of raw text
- Deploy the app publicly
- Evaluate at chunk level and expand the test set

---

⚠️ *This project is for educational purposes only and is not a substitute for professional medical advice.*
