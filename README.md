# 📄 Chat with Your PDF — RAG App using Streamlit + LLaMA 3

Interact with your PDF files in natural language using a lightweight Retrieval-Augmented Generation (RAG) system powered by **local embeddings** and **LLaMA 3**. Upload any PDF and ask real-time questions — the app finds relevant content and generates accurate, context-aware responses.

![Streamlit App Screenshot](screenshot.png) <!-- Replace with actual screenshot path -->

---

## 🚀 Features

- 🔍 **Ask questions about any PDF**  
- ⚡ **Fast and local** — no OpenAI or API keys required  
- 🧠 **LLaMA 3 integration** for high-quality responses  
- 📚 **ChromaDB vector search** for document retrieval  
- 🎨 Simple and responsive **Streamlit UI**

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI & frontend
- [LLaMA 3 (local)](https://ollama.com/library/llama3) – for language generation
- [ChromaDB](https://www.trychroma.com/) – vector database for similarity search
- [nomic-embed-text](https://docs.nomic.ai/Nomic-Embed-Text/) – for embedding PDF content
- Python + PyMuPDF (fitz) – for PDF parsing

---

## 📦 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/ajay2002-web/chat-with-pdf-rag.git
cd chat-with-pdf-rag
