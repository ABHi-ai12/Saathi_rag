from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import json
import numpy as np
from dotenv import load_dotenv
import os

from rag.rag import load_rag

# Load env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print("KEY:", api_key)

app = FastAPI()

# Load knowledge
with open("data/knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

texts = [item["content"] for item in knowledge]

# 🔥 SIMPLE embedding (dummy lightweight)
def simple_embed(text):
    return np.array([len(text)])  # lightweight trick

embeddings = np.array([simple_embed(t) for t in texts])

print("RAG Server Ready ✅")

# Lazy RAG loader
qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        try:
            print("Initializing RAG... ⏳")
            qa_chain = load_rag()
            print("RAG Loaded ✅")
        except Exception as e:
            print(f"RAG Error: {str(e)}")
            # Fallback or re-try later
    return qa_chain

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

class ChatRequest(BaseModel):
    message: str

# 🔍 Search endpoint
@app.post("/search")
def search(req: QueryRequest):
    query_embedding = simple_embed(req.query)

    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    indices = distances.argsort()[:req.top_k]

    results = []
    for i in indices:
        results.append({
            "content": knowledge[i]["content"],
            "topic": knowledge[i]["topic"],
            "score": float(distances[i])
        })

    return {"results": results}

# Health
@app.get("/health")
def health():
    return {"status": "ok"}

# Home
@app.get("/")
def home():
    return {"message": "RAG running 🚀"}

# Chat
@app.post("/chat")
def chat(req: ChatRequest):
    chain = get_qa_chain()
    if not chain:
        return {"reply": "Sorry, AI assistant is starting up or failed to load. Please check logs."}
    
    try:
        response = chain.run(req.message)
        return {"reply": response}
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return {"reply": f"Sorry, encountered an error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)