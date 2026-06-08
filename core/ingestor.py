import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Current Gemini text embedding model. text-embedding-004 (used in older
# tutorials) has been retired. gemini-embedding-001 is the stable text model;
# we request 768 dimensions to keep the FAISS index small and fast.
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embeddings client used for both indexing and querying."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        output_dimensionality=EMBEDDING_DIM,
    )


def load_and_chunk(pdf_paths: list[str]) -> list:
    """
    Load PDFs and split into chunks.
    Returns list of Document objects with source + page metadata.
    """
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded: {path} ({len(docs)} pages)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def build_faiss_index(chunks: list, index_path: str = "data/faiss_index") -> None:
    """
    Embed chunks using Gemini and persist FAISS index to disk.
    """
    embeddings = _get_embeddings()
    print("Building FAISS index... (this may take a minute)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")


def load_faiss_index(index_path: str = "data/faiss_index"):
    """
    Load a persisted FAISS index from disk.
    """
    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
