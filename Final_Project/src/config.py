"""
Central config - loads API keys from .env (local) or Streamlit secrets (cloud)
"""
import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str) -> str:
    """
    Works both locally (.env file) and on Streamlit Cloud (st.secrets)
    """
    # Try Streamlit secrets first (for deployed app)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fallback to environment variable (for local dev)
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing required secret: {key}")
    return value


COHERE_API_KEY = get_secret("COHERE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "study-assistant")

# Chunking config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding dimension for Cohere embed-english-v3.0
EMBED_DIM = 1024
