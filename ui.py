import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="AI Fire Service Chatbot", layout="wide")

st.title("🚒 Bihar Fire Service AI Chatbot")
st.write("Ask questions related to Bihar Fire Service reports.")

# User input
question = st.text_input("Enter your question:")

# Query FastAPI
if st.button("Ask AI") and question:
    response = requests.get(f"http://localhost:8000/query", params={"question": question})
    answer = response.json()["answer"]
    
    st.subheader("Answer:")
    st.write(answer)
