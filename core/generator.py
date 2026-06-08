import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# gemini-1.5-flash has been retired and returns 404. gemini-2.5-flash is the
# current free-tier flash model and is verified available on this API key.
LLM_MODEL = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are a precise research assistant. Your job is to answer questions \
using ONLY the provided document context.

Rules you must follow:
1. Every factual claim must have an inline citation in the format [Source: filename, Page X]
2. If the context does not contain enough information, say exactly: \
"The uploaded documents do not contain sufficient information to answer this question."
3. Never use knowledge from outside the provided context
4. Be concise and direct — no filler phrases

Context:
{context}

Question: {question}

Answer (with inline citations):"""

CITATION_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)


def _citation_page(doc) -> str:
    """
    PyPDFLoader stores pages 0-indexed. Display 1-indexed page numbers so a
    citation that says "Page 1" matches the actual first page of the PDF.
    """
    page = doc.metadata.get("page")
    if page is None:
        return "?"
    return str(page + 1)


def format_docs_with_metadata(docs: list) -> str:
    """
    Format retrieved docs with source metadata for the prompt.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        # Get just the filename, not the full path
        filename = source.split("/")[-1].split("\\")[-1]
        page = _citation_page(doc)
        parts.append(f"[Source: {filename}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(retriever):
    """
    Build the full RAG chain:
    query → retrieve → format → prompt → LLM → parse
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,       # Low temp = factual, less hallucination
        max_output_tokens=1024,
    )

    chain = (
        {
            "context": retriever | format_docs_with_metadata,
            "question": RunnablePassthrough(),
        }
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def get_retrieved_docs(retriever, query: str) -> list:
    """
    Get raw retrieved docs separately — needed for RAGAS eval.
    """
    return retriever.invoke(query)
