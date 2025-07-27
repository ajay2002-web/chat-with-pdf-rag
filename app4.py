import os
import tempfile
import uuid
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama

# ------------------------------------------
# 🧠 System Prompt for LLM
# ------------------------------------------
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
If the user says something like 'bye', 'thank you', or 'exit', respond politely and do not continue the conversation.
"""

# ------------------------------------------
# 📄 PDF Processing
# ------------------------------------------
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# ------------------------------------------
# 🧠 Vector Store
# ------------------------------------------
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma-v2")
    return chroma_client.get_or_create_collection(
        name=st.session_state.collection_name,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    batch_size = 50
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    st.success("✅ Data added to the vector store!")

# ------------------------------------------
# 🔍 Query & Response
# ------------------------------------------
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break

def call_llm(context, messages):
    # Reconstruct the message history for Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}]

    # Append all past user/assistant messages
    for msg in messages:
        ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the current context to the latest user message
    ollama_messages[-1]["content"] = f"Context: {context}\nQuestion: {ollama_messages[-1]['content']}"

    # Stream response from Ollama
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=ollama_messages,
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break



# ------------------------------------------
# 🚀 Streamlit UI
# ------------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set page config at top


# Initialize collection name
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"rag_session_{uuid.uuid4().hex[:8]}"

# Load Lottie animation
lottie_pdf = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vnikrcia.json")

# ------------------------------------------
# 💅 UI Styling
# ------------------------------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 42px;
        color: #00ADB5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #EEEEEE;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 13px;
        color: #888;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    <div class="big-title">📘 Ask Your PDF</div>
    <div class="sub-header">Powered by LLaMA 3 and local vector search ✨</div>
    """,
    unsafe_allow_html=True
)

# st.caption("Powered by LLaMA 3 and local vector search ✨")


st_lottie(lottie_pdf, height=200, key="pdf_lottie")

# ------------------------------------------
# 📤 Sidebar Upload & Processing
# ------------------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("**Upload PDF Files for QnA**", type=["pdf"], accept_multiple_files=False)
    process = st.button("⚙️ Process Document")

if uploaded_file and process:
    file_name_clean = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
    splits = process_document(uploaded_file)
    add_to_vector_collection(splits, file_name_clean)

# ------------------------------------------
# 💬 User Query
# ------------------------------------------
# st.markdown("### Ask a question about your uploaded document:")
# prompt = st.text_area("Your Question")
# ask = st.button("🔥 Ask")

# if ask and prompt:
#     results = query_collection(prompt)
#     if results.get("documents"):
#         context = results["documents"][0]
#         response = call_llm(context=context, prompt=prompt)
#         st.write_stream(response)
#     else:
#         st.warning("No relevant context found in the uploaded document.")
        


# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at the bottom
user_prompt = st.chat_input("💬 Ask anything about your document...")

if user_prompt:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Query the vector store and LLM
    results = query_collection(user_prompt)
    if results.get("documents"):
        context = results["documents"][0]

        # Stream the response if call_llm uses chat history
        with st.chat_message("assistant"):
            response_stream = call_llm(context=context, messages=st.session_state.messages)
            full_response = st.write_stream(response_stream)

        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        # If no document found
        with st.chat_message("assistant"):
            st.warning("No relevant context found in the uploaded document.")
        st.session_state.messages.append({"role": "assistant", "content": "No relevant context found in the uploaded document."})

