from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask_question

app = FastAPI(title="Medical RAG API")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return ask_question(req.question)