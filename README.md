# 📚 RAG System – URLs + Documents

A Retrieval-Augmented Generation (RAG) application built using **LangChain**, **Chroma**, **HuggingFace embeddings**, **Groq LLM**, and **Streamlit**.

## ✨ Features
- Ingest web URLs (Wikipedia, blogs, etc.)
- Upload documents (PDF, TXT, MD)
- Semantic search using vector embeddings
- Context-aware answers powered by LLM
- Streamlit-based interactive UI

## 🧠 Tech Stack
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Groq (LLaMA 3.1)
- Streamlit

## 🚀 How to Run

pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py

##  📌 Use Cases
Study assistant
Research Q&A
Private knowledge base
Document chatbot

## 🧾 Architecture
Documents + URLs → Chunking → Embeddings → Vector DB → Retrieval → LLM Answer

## Commit README:
git add README.md
git commit -m "Add README"
git push
