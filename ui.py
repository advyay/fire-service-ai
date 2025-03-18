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
    try:
        response = requests.get("https://fire-service-ai.onrender.com/query", params={"question": question})
        response.raise_for_status()  # Ensure request was successful
        answer = response.json().get("answer", "No answer received.")
    except requests.exceptions.RequestException as e:
        answer = f"Error connecting to API: {e}"
    
    st.subheader("Answer:")
    st.write(answer)
