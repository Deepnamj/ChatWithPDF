# rag_backend.py
# This file handles all the RAG (Retrieval-Augmented Generation) logic:
# - Loading and indexing the PDF
# - Retrieving relevant chunks
# - Generating answers using OpenAI

import os
from langchain_community.document_loaders import PyPDFLoader          # reads PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter    # splits text into chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI             # embeddings + chat model
from langchain_chroma import Chroma                                    # vector database
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # message types

# ----- API Key -----
# Set this in your terminal before running:
#   export OPENAI_API_KEY="sk-..."

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')#used for both embeddings and chat

# Path where ChromaDB will store the vector embeddings on disk
CHROMA_PATH = "chroma_db"

# ----- Prompt Template -----
# This tells the model to answer ONLY from the retrieved document context
PROMPT_TEMPLATE = """Use the following document context to answer the question.

Document Context:
{context}

Answer based only on the context above. If you can't find the answer, say so clearly."""

# ----- System Message -----
# This sets the assistant's overall behavior at the start of every conversation
SYSTEM_MESSAGE = SystemMessage(content="""You are a helpful assistant that answers questions 
based only on the provided document context. 
If the answer is not in the context, say "I don't have enough information in the document to answer that."
Do not make up information.
Do not say "according to the context" or "mentioned in the context" or similar.""")


def get_model():
    """
    Returns a ChatOpenAI instance using GPT-4o.
    You can change model_name to:
      - "gpt-4o"        (most capable, higher cost)
      - "gpt-4o-mini"   (faster, cheaper, good for most tasks)
      - "gpt-3.5-turbo" (cheapest)
    """
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",  # change model here if needed
        temperature=0,             # 0 = focused and deterministic answers
    )


def index_pdf(pdf_path: str) -> Chroma:
    """
    Loads a PDF, splits it into chunks, converts them to
    vector embeddings, and stores them in ChromaDB.

    Steps:
      1. Load PDF pages using PyPDFLoader
      2. Split pages into 500-character chunks with 50-char overlap
      3. Embed each chunk using OpenAI embeddings
      4. Store embeddings in ChromaDB on disk

    Returns the ChromaDB instance for later querying.
    """

    # Step 1: Load all pages from the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Step 2: Split pages into smaller chunks
    # chunk_size=500  → each chunk is max 500 characters
    # chunk_overlap=50 → chunks overlap by 50 chars to preserve context at boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    # Step 3: Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Step 4: Store chunks as vectors in ChromaDB
    db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    return db_chroma


def get_answer(db_chroma: Chroma, query: str, chat_history: list) -> str:
    """
    Given a user query, retrieves the top 5 most relevant chunks
    from ChromaDB and sends them along with the question to OpenAI
    to generate a grounded answer.

    Steps:
      1. Search ChromaDB for top 5 chunks closest to the query
      2. Combine chunks into a single context string
      3. Format the prompt with context + question
      4. Send system message + chat history + prompt to OpenAI
      5. Return the response

    Args:
      db_chroma    : the indexed ChromaDB instance
      query        : the user's question
      chat_history : list of previous HumanMessage/AIMessage objects

    Returns the answer as a string.
    """

    # Step 1: Find the 5 most relevant chunks using cosine similarity
    docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

    # If nothing relevant is found, return early
    if not docs_chroma:
        return "I couldn't find relevant information in the document."

    # Step 2: Join the retrieved chunks into one context block
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    # Step 3: Format the prompt with the context and user question
    full_query = f"{PROMPT_TEMPLATE.format(context=context_text)}\n\nQuestion: {query}"

    # Step 4: Build the full message list:
    #   [system message] + [past conversation] + [current question with context]
    messages = [SYSTEM_MESSAGE] + chat_history + [HumanMessage(content=full_query)]

    # Step 5: Call OpenAI and return the response
    model = get_model()
    response = model.invoke(messages)

    return response.content


def init_chat_history() -> list:
    """Returns an empty list to start a fresh chat session."""
    return []