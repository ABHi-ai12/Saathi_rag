from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load model + data
print("Loading model... ⏳")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/knowledge.json", "r") as f:
    knowledge = json.load(f)

texts = [item["content"] for item in knowledge]

# Build FAISS index
print("Building FAISS index... ⏳")
embeddings = model.encode(texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

print("RAG Server Ready ✅")

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

@app.post("/search")
def search(req: QueryRequest):
    query_embedding = model.encode([req.query], convert_to_numpy=True)
    
    distances, indices = index.search(
        query_embedding.astype(np.float32), 
        req.top_k
    )
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "content": knowledge[idx]["content"],
            "topic": knowledge[idx]["topic"],
            "score": float(distances[0][i])
        })
    
    return {"results": results}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "RAG running 🚀"}

if __name__ == "__main__":
    uvicorn.run(app, host="[IP_ADDRESS]", port=10000)
