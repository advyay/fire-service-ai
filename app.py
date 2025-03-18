import faiss
import numpy as np
import pandas as pd
import docx
import uvicorn
import nltk
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import ollama
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download("punkt")

# Paths to files
pdf_path = "Final_Report.pdf"
docx_path = "MARCH.docx"
xlsx_path = "Final.xlsx"

# Initialize FastAPI
app = FastAPI()

### Step 1: Extract Text from Files ###
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_docx_text(docx_path):
    doc = docx.Document(docx_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

def extract_xlsx_text(xlsx_path):
    df = pd.read_excel(xlsx_path)
    return df.to_string()

# Extract text from all files
pdf_text = extract_pdf_text(pdf_path)
docx_text = extract_docx_text(docx_path)
xlsx_text = extract_xlsx_text(xlsx_path)

# Combine extracted text
full_text = pdf_text + " " + docx_text + " " + xlsx_text

### Step 2: Chunk the Text using Sentence Tokenization ###
def split_into_chunks(text, chunk_size=1000):
    sentences = text.split(". ")  # Alternative splitting using periods
    chunks = []
    current_chunk = []

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

### Step 3: Generate Embeddings ###
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

### Step 4: Store Embeddings in FAISS ###
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save FAISS index
faiss.write_index(faiss_index, "fire_service_faiss.index")

# Save chunks
with open("fire_service_chunks.txt", "w") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

### Step 5: Query API ###
@app.get("/query")
def query_api(question: str = Query(..., description="Enter your question")):
    query_embedding = model.encode([question], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, k=3)  # Retrieve top 3 relevant chunks
    
    retrieved_context = " ".join([chunks[i] for i in I[0]])

    # Construct prompt for Ollama
    prompt = f"""
    Context:
    {retrieved_context}

    Question: {question}
    
    Answer based only on the given context.
    """

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return {"answer": response["message"]["content"]}

### Step 6: Run FastAPI ###
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
