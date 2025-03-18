import faiss
import numpy as np
import os
import gc
import uvicorn
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import ollama
import nltk

# Manually download missing resources
nltk.download("punkt")
nltk.download("punkt_tab")  # Fix for Render

# File paths
faiss_index_path = "fire_service_faiss.index"
chunks_file_path = "fire_service_chunks.txt"

# Initialize FastAPI
app = FastAPI()

### **Step 1: Load FAISS Index & Chunks if Already Created** ###
if os.path.exists(faiss_index_path) and os.path.exists(chunks_file_path):
    print("🔥 FAISS Index & Chunks Found! Skipping document processing...")
    
    # Load FAISS index
    faiss_index = faiss.read_index(faiss_index_path)
    
    # Load chunks
    with open(chunks_file_path, "r") as f:
        chunks = f.readlines()
    
else:
    print("⚠️ FAISS Index not found! Run document processing and vectorization first.")

    # Raise error since we shouldn't process documents in production
    raise FileNotFoundError("FAISS Index & Chunks not found! Please generate them first.")

### **Step 2: Load Sentence Transformer Model (Smaller Model for Less RAM)** ###
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v1")

### **Step 3: Query API (Search FAISS & Use Ollama)** ###
@app.get("/query")
def query_api(question: str = Query(..., description="Enter your question")):
    query_embedding = model.encode([question], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, k=3)  # Retrieve top 3 relevant chunks
    
    retrieved_context = " ".join([chunks[i] for i in I[0]])

    # Construct query for Ollama
    prompt = f"""
    Context:
    {retrieved_context}

    Question: {question}
    
    Answer based only on the given context.
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return {"answer": response["message"]["content"]}

### **Step 4: Run FastAPI (Use Render's PORT)** ###
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use environment variable for Render
    uvicorn.run(app, host="0.0.0.0", port=port)
