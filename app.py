# app.py
# This is the Streamlit frontend for the PDF Chat app.
# It handles the UI: file upload, chat display, and user input.
# All RAG logic is imported from rag_backend.py

import streamlit as st
import tempfile
import os
from langchain_core.messages import HumanMessage, AIMessage
from rag_backend import index_pdf, get_answer, init_chat_history

# ----- Page Config -----
# Sets the browser tab title and icon
st.set_page_config(page_title="PDF Chat", page_icon="📄")

# ----- App Title -----
st.title("📄 PDF Chat")
st.caption("Upload a PDF and ask questions about it")
st.divider()

# ----- Session State -----
# Streamlit reruns the script on every interaction,
# so we use session_state to persist data across reruns.

if "db_chroma" not in st.session_state:
    st.session_state.db_chroma = None       # stores the indexed ChromaDB

if "chat_history" not in st.session_state:
    st.session_state.chat_history = init_chat_history()  # stores HumanMessage/AIMessage objects for memory

if "messages" not in st.session_state:
    st.session_state.messages = []          # stores messages just for display purposes

# ----- PDF Upload Section -----
# File uploader widget — accepts only PDF files
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Show the Index button only after a file is uploaded
    if st.button("Index PDF"):
        with st.spinner("Indexing PDF... this may take a moment"):

            # Save the uploaded file to a temporary location on disk
            # (PyPDFLoader needs a file path, not a file object)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Index the PDF — load, split, embed, store in ChromaDB
            st.session_state.db_chroma = index_pdf(tmp_path)

            # Reset chat history whenever a new PDF is indexed
            st.session_state.chat_history = init_chat_history()
            st.session_state.messages = []

            # Delete the temp file after indexing is done
            os.unlink(tmp_path)

        st.success("✅ PDF indexed! You can now ask questions.")

st.divider()

# ----- Chat Section -----
# Only show chat after a PDF has been indexed
if st.session_state.db_chroma:

    # Display all previous messages in the chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):   # "user" or "assistant"
            st.write(msg["content"])

    # Chat input box — appears at the bottom of the screen
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        # Show the user's message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get the answer from Grok via rag_backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.db_chroma,   # the indexed vector DB
                    user_input,                    # current question
                    st.session_state.chat_history  # past conversation for memory
                )
            st.write(answer)

        # Update chat history with the latest exchange
        # (used internally by the model for multi-turn memory)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Save the assistant reply for display
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Button to clear the conversation and start fresh
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = init_chat_history()
        st.session_state.messages = []
        st.rerun()  # force Streamlit to re-render the page

else:
    # Shown when no PDF has been indexed yet
    st.info("👆 Upload a PDF and click **Index PDF** to get started.")