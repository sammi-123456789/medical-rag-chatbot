import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vector/db_faiss"
TOP_K = 5  # final config from evaluation

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Ye sab sirf ek baar load hoga, jab server start hoga
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=os.environ["GROQ_API_KEY"],
)

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": TOP_K}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


def ask_question(question: str) -> dict:
    response = qa_chain.invoke({"query": question})

    pages = []
    for doc in response["source_documents"]:
        page = doc.metadata.get("page")
        if page is not None:
            label = f"Page {page + 1}"  # PDF loader 0 se count karta hai
            if label not in pages:
                pages.append(label)

    return {"answer": response["result"], "sources": pages}