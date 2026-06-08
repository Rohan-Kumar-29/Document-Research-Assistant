# Multi-Document Research Assistant

A hybrid RAG system that answers questions from multiple PDFs with inline citations
and automatically evaluates every response using RAGAS metrics.

**Live demo:** _add your Streamlit Cloud link here_

---

## What it does

Upload multiple PDFs → ask questions → get cited answers → see live quality scores.

The key differentiator: every response is scored for faithfulness, answer relevance,
and context precision using RAGAS — so you don't just get an answer, you get a
confidence signal telling you how much to trust it.

---

## Architecture

**Pipeline:**
1. PDF ingestion → `RecursiveCharacterTextSplitter` (chunk size 800, overlap 120)
2. Embedding → Gemini `gemini-embedding-001` (768 dims) → stored in FAISS (local)
3. Retrieval → Hybrid: BM25 (0.4) + FAISS (0.6) via `EnsembleRetriever` with RRF fusion
4. Generation → Gemini 2.5 Flash with a citation-enforcing system prompt
5. Evaluation → RAGAS: faithfulness, answer relevancy, context precision

---

## Design Decisions

**Why chunk size 800 with 15% overlap?**
Smaller chunks (300–400) improve retrieval precision but lose cross-sentence context.
Larger chunks (1200+) dilute the embedding signal. 800 characters with 15% overlap
preserves semantic coherence while keeping each chunk focused enough for accurate
embedding.

**Why BM25 weight 0.4 and FAISS weight 0.6?**
Dense retrieval (FAISS) handles semantic queries better — "what does the author argue" —
while BM25 handles exact-term queries better — model names, numbers, acronyms. The 40/60
split favours FAISS because most user queries are semantic rather than keyword-based.

**Why Gemini 2.0 Flash?**
Cost efficiency and free-tier headroom: Flash is fast, cheap, and strong for
citation-grounded QA (the model only has to read provided context and cite it). The 1.5
models are retired, and the 2.5 models are capped at just 20 requests/day on the free
tier — too low for this app, which makes ~6 calls per question when evaluation is on.
Gemini 2.0 Flash keeps the generous 1,500 requests/day free tier, so the demo stays
usable without billing.

**Why faithfulness as the primary eval metric?**
Faithfulness directly measures hallucination risk — whether the answer is supported by
the retrieved context. For a document QA system, this is the most critical failure mode.
Answer relevancy checks if we're answering the right question; context precision checks if
we're retrieving the right chunks.

**Known limitations:**
- Does not support scanned PDFs (no OCR) — text-based PDFs only
- FAISS index is rebuilt per session (no cross-session persistence on Streamlit Cloud)
- RAGAS evaluation adds several seconds per response (LLM-as-judge)
- The Gemini free tier is rate limited; heavy batch evaluation can hit quota limits

**Future improvements:**
- Cross-encoder reranking after hybrid retrieval
- Multi-hop reasoning for complex questions
- Session persistence for the FAISS index
- Table and image extraction from PDFs

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Gemini 2.0 Flash |
| Embeddings | Gemini `gemini-embedding-001` |
| Orchestration | LangChain 0.3 |
| Vector store | FAISS (local) |
| Keyword search | BM25 (rank-bm25) |
| Evaluation | RAGAS 0.2 |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Evaluation Results

Run `tests/run_eval.py` against your own test documents and paste the measured
averages here. (Numbers below are placeholders — replace them with real output.)

| Metric | Score |
|---|---|
| Avg Faithfulness | X.XX |
| Avg Answer Relevancy | X.XX |
| Avg Context Precision | X.XX |

---

## Run locally

```bash
git clone https://github.com/Rohan-Kumar-29/Document-Research-Assistant
cd Document-Research-Assistant
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
GOOGLE_API_KEY=your_key_here
```

Then run:

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   `app.py` on the `main` branch.
3. **In Advanced settings, set the Python version to 3.11.** This is required —
   the RAGAS 0.2 / LangChain 0.3 stack does not build on Python 3.12+, and Cloud
   defaults to a newer Python. The version can't be changed after deploy without
   redeploying, so set it now. (A `runtime.txt` is currently unreliable on Cloud;
   use the Advanced settings dialog.)
4. In Advanced settings → Secrets, add your key in TOML form:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
5. Deploy.

---

## Project structure

```
.
├── app.py                 Streamlit UI (entry point)
├── core/
│   ├── ingestor.py        PDF loading, chunking, embedding, FAISS index
│   ├── retriever.py       Hybrid BM25 + FAISS retriever
│   ├── generator.py       LLM chain with citation prompt
│   └── evaluator.py       RAGAS scoring
├── tests/
│   ├── eval_dataset.json  Q&A pairs for evaluation
│   └── run_eval.py        Batch evaluation runner
├── requirements.txt
└── .env.example
```
