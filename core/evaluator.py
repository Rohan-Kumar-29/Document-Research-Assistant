from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset

from core.config import get_api_key
from core.ingestor import EMBEDDING_MODEL, EMBEDDING_DIM


# Same model as the generator. gemini-2.0-flash keeps the generous free-tier
# limit (1,500 requests/day); the 2.5 models are capped at only 20/day.
JUDGE_MODEL = "gemini-2.0-flash"


def score_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict:
    """
    Score a single RAG response using RAGAS metrics.

    Metrics:
    - Faithfulness: Is the answer supported by the context? (hallucination check)
    - Answer Relevancy: Does the answer actually address the question?
    - Context Precision: Are the retrieved chunks relevant?

    Returns a dict of metric_name -> float (0.0 to 1.0)
    """
    llm = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        google_api_key=get_api_key(),
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    # answer_relevancy needs an embeddings model to score the answer; the LLM
    # alone is not enough. Reuse the same Gemini embedding model as ingestion.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_api_key(),
        output_dimensionality=EMBEDDING_DIM,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth if ground_truth else answer],
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    # In RAGAS 0.2 the scores are read from a DataFrame. Pull the single row and
    # coerce to float; missing/NaN scores fall back to 0.0 so the UI never breaks.
    df = result.to_pandas()

    def _get(metric: str) -> float:
        try:
            value = float(df[metric].iloc[0])
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0
        if value != value:  # NaN check
            return 0.0
        return round(value, 3)

    return {
        "faithfulness": _get("faithfulness"),
        "answer_relevancy": _get("answer_relevancy"),
        "context_precision": _get("context_precision"),
    }


def interpret_score(faithfulness_score: float) -> tuple[str, str]:
    """
    Returns (confidence_label, explanation) based on faithfulness score.
    """
    if faithfulness_score >= 0.8:
        return "High confidence", "Answer is well-grounded in the source documents."
    elif faithfulness_score >= 0.5:
        return "Moderate confidence", "Answer is partially supported. Verify key claims."
    else:
        return "Low confidence", "Answer may not be fully supported by retrieved context."
