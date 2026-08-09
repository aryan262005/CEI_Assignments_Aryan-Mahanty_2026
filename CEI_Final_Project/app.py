import os
import tempfile
import streamlit as st

from src.indexing import build_index_from_files
from src.query_engine import ask_question

st.set_page_config(page_title="AI-Powered Study Assistant", page_icon="📚", layout="wide")

st.title(" AI-Powered Study Assistant")
st.caption("Upload your study material and ask questions — answers come straight from your documents (RAG).")

# ---- Session state ----
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar: upload ----
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT study material",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading, chunking, and embedding your documents..."):
            temp_paths = []
            temp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                temp_paths.append(path)

            index, sample_text = build_index_from_files(temp_paths)
            st.session_state.index = index
            st.session_state.messages = []

        st.success(f"Indexed {len(uploaded_files)} document(s). Ask away!")
        with st.expander("🔍 Preview extracted text (debug)"):
            st.text(sample_text)

    st.divider()
    st.caption("Powered by LlamaIndex + Cohere + Pinecone")

# ---- Main: chat ----
if st.session_state.index is None:
    st.info("👈 Upload and process a document first to start asking questions.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about your uploaded material...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                answer = ask_question(st.session_state.index, question)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
