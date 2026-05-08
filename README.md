## Project Title
**ChatWithPDF — RAG-Based PDF Question Answering System**

An AI chatbot that lets users upload any PDF and ask questions about it using RAG — powered by LangChain, OpenAI, and ChromaDB.

---

Technical Description

- Built a **RAG pipeline** from scratch — PDF loading, chunking, OpenAI embeddings, ChromaDB vector storage, and GPT-based answer generation
- Implemented **semantic search** using cosine similarity to retrieve the top-k most relevant document chunks for any user query
- Added **multi-turn chat memory** using LangChain message history so the assistant remembers previous questions in a session
- Developed a clean **Streamlit web UI** with PDF upload, real-time chat interface, and chat history reset
- Optimized chunking strategy using `RecursiveCharacterTextSplitter` with tuned `chunk_size` and `chunk_overlap` for better retrieval accuracy

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| AI / LLM | OpenAI GPT-3.5-turbo, OpenAI Embeddings |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Frontend | Streamlit |
| PDF Processing | PyPDF |

---
