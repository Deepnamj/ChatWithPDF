# ChatWithPDF — RAG-Based PDF Question Answering System

An AI chatbot that lets users upload any PDF and ask questions about it using RAG — powered by LangChain, OpenAI GPT-3.5-turbo, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangChain](https://img.shields.io/badge/LangChain-RAG-teal) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-purple) ![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## About

ChatWithPDF lets you upload any PDF document and have a multi-turn conversation about its contents. A full RAG pipeline handles loading, chunking, embedding, and retrieval — GPT-3.5-turbo then generates grounded, context-aware answers. Chat memory persists across turns so follow-up questions work naturally.

---

## Features

- Upload any PDF and start asking questions instantly
- Full RAG pipeline — chunking, OpenAI embeddings, ChromaDB vector storage, GPT-3.5-turbo answer generation
- Semantic search via cosine similarity to retrieve the top-k most relevant chunks per query
- Multi-turn chat memory using LangChain message history — the assistant remembers previous questions
- Optimized chunking with `RecursiveCharacterTextSplitter` (tuned `chunk_size` + `chunk_overlap`) for better retrieval accuracy
- Clean Streamlit UI with PDF upload, real-time chat, and chat history reset

---

## How It Works

```
Upload PDF → Parse + chunk → Embed (OpenAI) → Store (ChromaDB)
User query → Retrieve top-k chunks → GPT-3.5-turbo → Answer
```

---

## Tech Stack

| Category        | Tools                                   |
|----------------|-----------------------------------------|
| Language        | Python                                  |
| AI / LLM        | OpenAI GPT-3.5-turbo, OpenAI Embeddings |
| RAG Framework   | LangChain                               |
| Vector Database | ChromaDB                                |
| Frontend        | Streamlit                               |
| PDF Processing  | PyPDF                                   |

---

## Installation

```bash
git clone https://github.com/Deepnamj/ChatWithPDF.git
cd ChatWithPDF
pip install -r req.txt
```

---

## Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a PDF, and start chatting.

---

## Project Structure

```
ChatWithPDF/
├── chroma_db/               # ChromaDB persistent storage
│   └── chroma.sqlite3
├── app.py                   # Streamlit UI + chat interface
├── rag_backend.py           # PDF loading, chunking, embedding, retrieval
├── req.txt                  # Dependencies
├── ChatWithPDF_Sample.pdf   # Sample PDF for testing
└── README.md
```

---

## License

MIT License — free to use, modify, and distribute.

---

Built by [Deepna Maria Jimson](https://github.com/Deepnamj) · March 2025
