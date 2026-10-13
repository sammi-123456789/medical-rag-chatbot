"""
RAG Chatbot Evaluation Script
-----------------------------
Ye script test_questions.csv ke har sawaal ko chatbot se poochti hai aur 2 cheezein check karti hai:
1. Retrieval Hit Rate  -> kya sahi page top-k chunks mein aaya?
2. Answer Accuracy     -> kya LLM ka jawab sahi hai? (LLM-as-a-judge)

Run:  python evaluate.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

# ---------------- Settings (yahi change karke experiment karna) ----------------
DB_FAISS_PATH = "vector/db_faiss_1000"
TEST_FILE = "test_questions.csv"
TOP_K = 3
MODEL_NAME = "openai/gpt-oss-120b"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

JUDGE_PROMPT = """You are grading a medical chatbot's answer.

Question: {question}
Reference answer: {reference}
Chatbot answer: {answer}

Rules:
- If the chatbot answer contains the key facts of the reference answer, it is CORRECT.
- If the reference answer is "I don't know", the chatbot is CORRECT only if it says it doesn't know or refuses.
- Extra wording is fine. Wrong or missing key facts is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


# ---------------- Setup ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatGroq(model_name=MODEL_NAME, temperature=0.0, groq_api_key=os.environ["GROQ_API_KEY"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,
                                                input_variables=["context", "question"])},
)


def judge(question, reference, answer):
    """Doosra LLM call jo jawab ko CORRECT / INCORRECT mark karta hai."""
    verdict = llm.invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    return "INCORRECT" not in verdict.content.upper()


# ---------------- Evaluation loop ----------------
tests = pd.read_csv(TEST_FILE)
results = []

for i, row in tests.iterrows():
    question = row["question"]
    reference = row["reference_answer"]
    expected_page = row["expected_page"]          # khali = out-of-scope sawaal

    response = qa_chain.invoke({"query": question})
    answer = response["result"]
    # PyPDFLoader page 0 se count karta hai, isliye +1 (PDF viewer jaisa number)
    retrieved_pages = [doc.metadata.get("page", -1) + 1 for doc in response["source_documents"]]

    in_scope = pd.notna(expected_page)
    hit = int(expected_page) in retrieved_pages if in_scope else None
    correct = judge(question, reference, answer)

    results.append({
        "question": question,
        "in_scope": in_scope,
        "expected_page": expected_page,
        "retrieved_pages": retrieved_pages,
        "retrieval_hit": hit,
        "answer": answer,
        "correct": correct,
    })
    print(f"[{i + 1}/{len(tests)}] hit={hit}  correct={correct}  |  {question[:60]}")
    time.sleep(2)  # Groq free tier rate limit se bachne ke liye

df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False)

# ---------------- Summary ----------------
in_scope_df = df[df["in_scope"]]
out_scope_df = df[~df["in_scope"]]

print("\n========== EVALUATION SUMMARY ==========")
print(f"Settings: top_k={TOP_K}, model={MODEL_NAME}")
print(f"Total questions        : {len(df)}")
print(f"Retrieval hit rate     : {in_scope_df['retrieval_hit'].mean():.0%}  (in-scope only)")
print(f"Answer accuracy        : {in_scope_df['correct'].mean():.0%}  (in-scope only)")
if len(out_scope_df):
    print(f"Out-of-scope refusals  : {out_scope_df['correct'].mean():.0%}")
print("Detailed results saved to eval_results.csv")
