import streamlit as st
import os
from rag_pipeline import load_documents, build_vector_db, ask_question

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 RAG System – URLs + Documents")

# --- Sidebar ---
st.sidebar.header("🔗 Data Sources")

urls_input = st.sidebar.text_area(
    "Enter URLs (one per line)",
    height=150
)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
    type=["pdf", "txt", "md"]
)

if st.sidebar.button("📥 Build Knowledge Base"):
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

    # Save uploaded files
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        documents = load_documents(urls, UPLOAD_DIR)
        build_vector_db(documents)

    st.sidebar.success("Knowledge base ready!")

# --- Main QA ---
st.header("💬 Ask a Question")

query = st.text_input("Your question")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_question(query)
        st.success("Answer")
        st.write(answer)
