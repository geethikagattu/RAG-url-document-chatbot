import os
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"

PROMPT = """
You are an AI assistant answering questions using retrieved context.

If the question is asking for a DEFINITION or GENERAL EXPLANATION,
prefer encyclopedic, factual context over fictional examples.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

def load_documents(urls, docs_path):
    documents = []

    # URLs
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())

    # Uploaded files
    loaders = [
        DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]

    for loader in loaders:
        if os.path.exists(docs_path):
            documents.extend(loader.load())

    return documents


def build_vector_db(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    return db


def ask_question(query):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(PROMPT).format(
        context=context,
        question=query
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )



    return llm.invoke(prompt).content
