import os
import tempfile
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import streamlit as st

# Initialize embedding function
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./csv-chroma-store")
collection = chroma_client.get_or_create_collection(
    name="csv_rag_app",
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"},
)

# Function to load and chunk CSV file
def process_csv(uploaded_file):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    df = pd.read_csv(temp_path)

    # Clean up
    os.unlink(temp_path)

    # Combine all columns into a single string per row
    combined_rows = df.astype(str).agg(" | ".join, axis=1).tolist()

    # Chunk rows if needed (optional)
    chunks = [combined_rows[i:i+10] for i in range(0, len(combined_rows), 10)]
    text_chunks = ["\n".join(chunk) for chunk in chunks]
    return text_chunks

# Ingest into ChromaDB
def add_to_vectorstore(docs):
    for i, chunk in enumerate(docs):
        collection.add(
            documents=[chunk],
            ids=[f"doc-{i}"]
        )

# Query function
def query_knowledgebase(query):
    results = collection.query(query_texts=[query], n_results=3)
    docs = [doc for doc in results["documents"][0]]
    return "\n---\n".join(docs)

# Streamlit UI
st.set_page_config(page_title="CSV Q&A App", layout="wide")
st.title("📊 CSV Document Q&A App")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully")
    with st.spinner("🔄 Processing and indexing..."):
        splits = process_csv(uploaded_file)
        add_to_vectorstore(splits)
        st.success("✅ Document indexed in vector store")

    st.markdown("---")
    user_query = st.text_input("Ask a question based on the uploaded CSV:")
    if user_query:
        with st.spinner("💬 Searching..."):
            answer = query_knowledgebase(user_query)
            st.markdown("### 🔍 Top Matching Chunks:")
            st.write(answer)
