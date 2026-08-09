"""
Handles: loading documents -> chunking -> embeddings -> storing in Pinecone
"""
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

from src.config import (
    COHERE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_DIM,
)


def configure_llama_index():
    """Set global LLM + embedding model for LlamaIndex"""
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_document",
    )
    Settings.llm = Cohere(
        api_key=COHERE_API_KEY,
        model="command-r-08-2024",
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def get_pinecone_index():
    """Create Pinecone index if it doesn't exist, return handle"""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def load_documents(file_paths: list[str]) -> list[Document]:
    """
    Loads PDFs (via PyMuPDF, which handles font-encoding issues much better
    than the default pypdf-based reader) and plain text files.
    """
    documents = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            text_parts = []
            with fitz.open(path) as pdf:
                for page in pdf:
                    text_parts.append(page.get_text())
            full_text = "\n".join(text_parts).strip()
            if full_text:
                documents.append(Document(text=full_text, metadata={"file_name": path}))
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read().strip()
            if full_text:
                documents.append(Document(text=full_text, metadata={"file_name": path}))
    return documents


def build_index_from_files(file_paths: list[str]) -> VectorStoreIndex:
    """
    Takes a list of uploaded file paths, loads + chunks + embeds them,
    and stores into Pinecone. Returns a queryable index.
    """
    configure_llama_index()

    documents = load_documents(file_paths)
    if not documents:
        raise ValueError("No readable text found in the uploaded file(s).")

    pinecone_index = get_pinecone_index()

    # Clear any old/stale vectors before adding fresh ones
    try:
        pinecone_index.delete(delete_all=True)
    except Exception:
        pass  # index might already be empty

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    sample_text = documents[0].text[:300]
    return index, sample_text


def load_existing_index() -> VectorStoreIndex:
    """
    Reconnects to an already-populated Pinecone index
    (use this when documents are already uploaded from a past session)
    """
    configure_llama_index()

    pinecone_index = get_pinecone_index()
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(vector_store)
