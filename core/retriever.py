from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from core.ingestor import load_faiss_index


def build_hybrid_retriever(chunks: list, index_path: str = "data/faiss_index"):
    """
    Builds a hybrid retriever combining:
    - BM25 (keyword-based, sparse) — weight 0.4
    - FAISS (semantic, dense)      — weight 0.6
    Fusion via Reciprocal Rank Fusion (RRF).
    """
    # Dense retriever — semantic similarity via embeddings
    vectorstore = load_faiss_index(index_path)
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # Sparse retriever — exact keyword matching
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    # Ensemble — RRF fusion
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )
    return hybrid_retriever
