import json
import time

from dotenv import load_dotenv

load_dotenv()

from core.ingestor import load_and_chunk, build_faiss_index
from core.retriever import build_hybrid_retriever
from core.generator import build_chain, get_retrieved_docs
from core.evaluator import score_response

# The Gemini free tier is rate limited (15 req/min) and RAGAS makes several
# judge calls per question on top of generation. Sleep between questions to
# stay under the limit; bump this up if you still see 429 errors.
SLEEP_BETWEEN_QUESTIONS = 15

# Point these at the PDFs your eval_dataset.json questions are written against.
PDF_PATHS = ["sample_docs/doc1.pdf", "sample_docs/doc2.pdf"]


def main() -> None:
    chunks = load_and_chunk(PDF_PATHS)
    build_faiss_index(chunks)
    retriever = build_hybrid_retriever(chunks)
    chain = build_chain(retriever)

    with open("tests/eval_dataset.json", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    failed = []
    for idx, item in enumerate(eval_set):
        q = item["question"]
        gt = item.get("ground_truth", "")
        try:
            answer = chain.invoke(q)
            docs = get_retrieved_docs(retriever, q)
            contexts = [d.page_content for d in docs]
            scores = score_response(q, answer, contexts, ground_truth=gt)
            results.append({"question": q, "answer": answer, "scores": scores})
            print(f"Q: {q[:60]}...")
            print(f"Scores: {scores}\n")
        except Exception as e:
            # Don't let one rate-limited question abort the whole batch.
            failed.append(q)
            print(f"Q: {q[:60]}...")
            print(f"FAILED ({type(e).__name__}): {e}\n")

        # Be polite to the free tier between questions (skip after the last).
        if idx < len(eval_set) - 1:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    if not results:
        print("No questions scored successfully — check your API quota and PDFs.")
        return

    avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["scores"]["answer_relevancy"] for r in results) / len(results)
    avg_precision = sum(r["scores"]["context_precision"] for r in results) / len(results)

    print("=== AGGREGATE RESULTS ===")
    print(f"Avg Faithfulness:     {avg_faithfulness:.3f}")
    print(f"Avg Answer Relevancy: {avg_relevancy:.3f}")
    print(f"Avg Context Precision:{avg_precision:.3f}")
    print(f"Scored questions:     {len(results)}")
    if failed:
        print(f"Failed questions:     {len(failed)} (re-run these once quota resets)")


if __name__ == "__main__":
    main()
