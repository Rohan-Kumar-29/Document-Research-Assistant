import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.ingestor import load_and_chunk, build_faiss_index
from core.retriever import build_hybrid_retriever
from core.generator import build_chain, get_retrieved_docs
from core.evaluator import score_response, interpret_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []


def _filename(source: str) -> str:
    """Return just the filename from a full path (cross-platform)."""
    return source.split("/")[-1].split("\\")[-1]


def _display_page(doc) -> str:
    """PyPDFLoader pages are 0-indexed; show 1-indexed to match the real PDF."""
    page = doc.metadata.get("page")
    return str(page + 1) if page is not None else "?"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Research Assistant")
    st.caption("Hybrid RAG · RAGAS Evaluation · Gemini + LangChain")
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports multiple PDFs. Each page is indexed separately.",
    )

    if uploaded_files:
        if st.button("Build Index", type="primary", use_container_width=True):
            with st.spinner("Loading, chunking, and embedding..."):
                tmp_paths = []
                for f in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", dir="sample_docs"
                    )
                    tmp.write(f.read())
                    tmp.close()
                    tmp_paths.append(tmp.name)

                try:
                    chunks = load_and_chunk(tmp_paths)
                    build_faiss_index(chunks)
                finally:
                    # Clean up temp PDFs so sample_docs/ doesn't fill up.
                    for p in tmp_paths:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

                # Build the retriever + chain once and cache them, instead of
                # rebuilding on every query.
                retriever = build_hybrid_retriever(chunks)
                st.session_state.chunks = chunks
                st.session_state.retriever = retriever
                st.session_state.chain = build_chain(retriever)
                st.session_state.indexed = True
                st.session_state.history = []

            st.success(f"Indexed {len(uploaded_files)} file(s) → {len(chunks)} chunks")

    if st.session_state.indexed:
        st.divider()
        st.caption(f"Index ready · {len(st.session_state.chunks)} chunks")

    st.divider()
    st.caption("How it works")
    st.markdown("""
    1. Upload PDFs → chunked + embedded
    2. Query → hybrid BM25 + FAISS retrieval
    3. Gemini generates cited answer
    4. RAGAS scores the response live
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ask your documents")

if not st.session_state.indexed:
    st.info("Upload PDFs in the sidebar and click **Build Index** to get started.")
    st.stop()

query = st.chat_input("Ask a question about your documents...")

# Show history
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", item["scores"]["faithfulness"])
        col2.metric("Answer Relevance", item["scores"]["answer_relevancy"])
        col3.metric("Context Precision", item["scores"]["context_precision"])

# Handle new query
if query:
    retriever = st.session_state.retriever
    chain = st.session_state.chain

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            docs = get_retrieved_docs(retriever, query)

        with st.spinner("Generating answer..."):
            answer = chain.invoke(query)

        st.write(answer)

        with st.expander("Retrieved source chunks", expanded=False):
            for i, doc in enumerate(docs):
                src = _filename(doc.metadata.get("source", "unknown"))
                page = _display_page(doc)
                st.markdown(f"**Chunk {i+1}** — `{src}` · Page {page}")
                st.text(
                    doc.page_content[:400]
                    + ("..." if len(doc.page_content) > 400 else "")
                )
                st.divider()

        with st.spinner("Evaluating response quality..."):
            contexts = [d.page_content for d in docs]
            scores = score_response(query, answer, contexts)
            label, explanation = interpret_score(scores["faithfulness"])

        st.subheader("Response Quality (RAGAS)")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Faithfulness", scores["faithfulness"],
            help="Is the answer supported by the context? Close to 1.0 = no hallucination.",
        )
        col2.metric(
            "Answer Relevance", scores["answer_relevancy"],
            help="Does the answer address the question?",
        )
        col3.metric(
            "Context Precision", scores["context_precision"],
            help="Are the retrieved chunks relevant to the question?",
        )

        if scores["faithfulness"] >= 0.8:
            st.success(f"**{label}** — {explanation}")
        elif scores["faithfulness"] >= 0.5:
            st.warning(f"**{label}** — {explanation}")
        else:
            st.error(f"**{label}** — {explanation}")

        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "scores": scores,
        })
