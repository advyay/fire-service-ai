import faiss
import numpy as np
import pandas as pd
import docx
import uvicorn
import nltk
import gc
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import ollama
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer data
nltk.download("punkt")

# Paths to files
pdf_path = "Final_Report.pdf"
docx_path = "MARCH.docx"
xlsx_path = "Final.xlsx"

# Initialize FastAPI
app = FastAPI()

### **Step 1: Extract Text from Files (Optimized for Low Memory)** ###
def extract_text_stream(filepath):
    """ Reads large files in a streaming manner to reduce memory usage """
    if filepath.endswith(".pdf"):
        text = []
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        return " ".join(text)
    
    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        return " ".join(para.text for para in doc.paragraphs)

    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, engine="openpyxl")  # Ensure openpyxl is used
        return "\n".join(df.astype(str).values.flatten())

    return ""

# Extract text with reduced memory usage
pdf_text = extract_text_stream(pdf_path)
docx_text = extract_text_stream(docx_path)
xlsx_text = extract_text_stream(xlsx_path)

# Free memory by deleting file readers
gc.collect()

# Combine extracted text
full_text = pdf_text + " " + docx_text + " " + xlsx_text

### **Step 2: Chunk the Text using Sentence Tokenization (Optimized for Meaningful Contexts)** ###
def split_into_chunks(text, chunk_size=1000):
    """ Splits text into chunks using sentence tokenization """
    sentences = sent_tokenize(text)  # More natural splits
    chunks, current_chunk = [], []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = split_into_chunks(full_text, chunk_size=1000)

# Free up memory after chunking
del full_text
gc.collect()

### **Step 3: Generate Embeddings (Using Lighter Model to Reduce RAM Usage)** ###
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v1")  # Lighter model

embeddings = model.encode(chunks, convert_to_numpy=True)

### **Step 4: Store Embeddings in FAISS (Efficient Storage to Reduce Memory Load)** ###
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Free memory after FAISS indexing
del embeddings
gc.collect()

# Save FAISS index
faiss.write_index(faiss_index, "fire_service_faiss.index")

# Save chunks for retrieval reference
with open("fire_service_chunks.txt", "w") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

# Free memory
del chunks
gc.collect()

### **Step 5: Query API with Efficient Search** ###
@app.get("/query")
def query_api(question: str = Query(..., description="Enter your question")):
    query_embedding = model.encode([question], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, k=3)  # Retrieve top 3 relevant chunks
    
    retrieved_context = " ".join([open("fire_service_chunks.txt").readlines()[i] for i in I[0]])

    # Construct prompt for Ollama
    prompt = f"""
    Context:
    {retrieved_context}

    Question: {question}
    
    Answer based only on the given context.
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return {"answer": response["message"]["content"]}

### **Step 6: Run FastAPI (Use $PORT for Render)** ###
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 10000))  # Use environment variable for Render
    uvicorn.run(app, host="0.0.0.0", port=port)
