"""
Wraps the LlamaIndex query engine for asking questions against the index
"""
from llama_index.core import VectorStoreIndex


def get_query_engine(index: VectorStoreIndex, top_k: int = 3):
    """
    top_k = how many relevant chunks to retrieve per question
    """
    return index.as_query_engine(
        similarity_top_k=top_k,
        streaming=False,
    )


def ask_question(index: VectorStoreIndex, question: str, top_k: int = 3) -> str:
    query_engine = get_query_engine(index, top_k=top_k)
    response = query_engine.query(question)
    return str(response)
