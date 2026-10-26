# RAG Project From Scratch

A Retrieval-Augmented Generation (RAG) system built from the ground up to understand and evaluate the individual components of a modern RAG pipeline before relying on frameworks such as LangChain or LlamaIndex.

The project started as an experimental notebook and evolved into a modular production-style application with:

- PDF ingestion and cleaning
- sentence-aware chunking
- dense embeddings
- semantic retrieval
- BM25 and hybrid retrieval experiments
- Reciprocal Rank Fusion (RRF)
- cross-encoder reranking
- query rewriting
- decomposition
- HyDE
- metadata filtering
- parent-child retrieval
- context compression
- retrieval evaluation
- generation evaluation
- Qwen2.5-7B-Instruct generation
- remote GPU inference through Google Colab
- FastAPI model serving
- API-key authentication
- ngrok tunneling
- local RAG API
- persistence
- latency measurement
- error handling
- automated API tests

The final production baseline was intentionally kept simpler than the full experimental system because evaluation showed that several additional retrieval techniques did not improve results enough to justify their added complexity.

---

# 1. Project Goal

The goal of this project was not simply to build another PDF chatbot.

The main objective was to understand the complete RAG pipeline at a low level:

```text
Documents
    ↓
Cleaning
    ↓
Chunking
    ↓
Embeddings
    ↓
Retrieval
    ↓
Reranking
    ↓
Context Construction
    ↓
Prompt
    ↓
LLM
    ↓
Grounded Answer
```

Each major RAG technique was implemented and evaluated separately before choosing the final production architecture.

This makes the project both:

- a working RAG application
- an experimental environment for understanding retrieval and generation behavior

---

# 2. Final Architecture

The final validated pipeline is:

```text
User Question
      ↓
SentenceTransformer Embedding
      ↓
Dense Retrieval — Top 20
      ↓
Cross-Encoder Reranking
      ↓
Top 5 Chunks
      ↓
Grounded Prompt
      ↓
HTTP Request
      ↓
ngrok
      ↓
FastAPI Model Server on Google Colab GPU
      ↓
Qwen2.5-7B-Instruct
      ↓
Grounded Answer
```

The local application handles:

```text
PDF processing
Chunking
Embeddings
Index persistence
Retrieval
Reranking
API
```

The remote Colab environment handles:

```text
Qwen2.5-7B inference
```

This design was chosen because the development machine does not have a local CUDA-capable GPU.

---

# 3. Technology Stack

## Core

- Python
- NumPy
- PyPDF
- NLTK
- Sentence Transformers
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- Requests
- python-dotenv
- pytest

## Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

### Cross-Encoder Reranker

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Final Generator

```text
Qwen/Qwen2.5-7B-Instruct
```

The 7B model is run in a quantized configuration on Google Colab.

An earlier generator experiment used:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

but evaluation showed that model capability was a significant bottleneck even when retrieval was correct.

---

# 4. Corpus

The current corpus contains three research papers:

```text
1706.03762v7.pdf
2405.07987v5.pdf
NeurIPS_ML4PS_2025_87.pdf
```

They include work related to:

- the Transformer architecture
- the Platonic Representation Hypothesis
- representational alignment in astronomy models

After preprocessing and chunking:

```text
Total chunks: 240
Embedding matrix shape: (240, 384)
```

---

# 5. Project Structure

```text
Rag-project/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── generation.py
│   ├── persistence.py
│   ├── indexing.py
│   └── pipeline.py
│
├── scripts/
│   ├── __init__.py
│   ├── build_index.py
│   ├── test_pipeline.py
│   ├── test_retrieval.py
│   ├── inspect_notebook.py
│   └── extract_production_functions.py
│
├── notebooks/
│   └── rag_experiments.ipynb
│
├── tests/
│   └── test_api.py
│
├── data/
├── artifacts/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

The notebook contains the broader experimental work.

The `src/` directory contains only the components selected for the production pipeline.

---

# 6. Document Ingestion

PDFs are loaded page by page using `PyPDF`.

Each page is represented with metadata:

```python
{
    "document": "...",
    "page": 1,
    "text": "..."
}
```

This metadata is preserved through chunking so retrieved chunks can later be traced back to their source document and page.

---

# 7. Cleaning

The cleaning stage:

- ignores pages with no extracted text
- removes empty pages
- strips unnecessary whitespace
- normalizes repeated spaces/tabs
- normalizes excessive newlines

The goal is deliberately conservative.

The pipeline does not heavily modify the source text because aggressive preprocessing can destroy useful information.

---

# 8. Chunking

The production pipeline uses sentence-aware chunking.

Default approximate chunk size:

```text
1000 characters
```

A one-sentence overlap is maintained between consecutive chunks.

Conceptually:

```text
Chunk 1:
Sentence A
Sentence B
Sentence C

Chunk 2:
Sentence C
Sentence D
Sentence E
```

The overlap helps preserve information that crosses chunk boundaries.

Each chunk contains:

```python
{
    "document": "...",
    "page": ...,
    "chunk_id": ...,
    "chunk_text": "..."
}
```

---

# 9. Embeddings

Chunk text is embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are currently stored locally as a NumPy array.

Current embedding shape:

```text
240 × 384
```

Similarity is calculated using cosine similarity.

---

# 10. Index Persistence

The indexing pipeline does not need to rebuild embeddings every time the application starts.

The processed corpus and embeddings are persisted as:

```text
artifacts/corpus.json
artifacts/embeddings.npy
```

`save_index()` stores the index.

`load_index()` restores it when the RAG pipeline starts.

The artifacts directory is excluded from Git because generated index files should be reproducible rather than committed.

---

# 11. Experimental Retrieval Techniques

Several retrieval strategies were implemented before selecting the production baseline.

## Dense Retrieval

Semantic similarity using sentence-transformer embeddings.

```text
Query
↓
Embedding
↓
Cosine similarity against document vectors
↓
Top-K
```

---

## BM25

BM25 was tested as a lexical retrieval system.

It is useful when exact terms matter.

Dense retrieval and BM25 provide complementary signals:

```text
BM25 → lexical similarity
Dense retrieval → semantic similarity
```

---

## Hybrid Retrieval

Dense and BM25 results were combined using Reciprocal Rank Fusion.

Conceptually:

```text
Dense Ranking
      +
BM25 Ranking
      ↓
RRF
      ↓
Combined Ranking
```

---

## Cross-Encoder Reranking

The initial retrieval stage produces a relatively broad candidate set.

The cross-encoder then evaluates each:

```text
(query, chunk)
```

pair together.

The final production configuration uses:

```text
Dense Top 20
↓
Cross-Encoder
↓
Top 5
```

This gave the strongest evaluation results while remaining simpler than the hybrid alternatives.

---

# 12. Additional RAG Techniques Evaluated

The experimental notebook also includes:

## Query Rewriting

Transforms the original query into a potentially easier retrieval representation.

---

## Context-Aware Query Rewriting

Uses conversation/context information while rewriting the query.

---

## Query Decomposition

Breaks complex questions into smaller subquestions.

Always-on decomposition did not improve aggregate retrieval results sufficiently, so it was not included in the production baseline.

---

## HyDE

Hypothetical Document Embeddings generate a hypothetical answer/document and use that representation for retrieval.

Important design decision:

> HyDE-generated text is used for retrieval representation only and is not treated as factual evidence.

---

## Metadata Filtering

Allows retrieval to be limited based on document metadata.

---

## Parent-Child Retrieval

Small child chunks are used for precise retrieval while larger parent chunks provide additional context.

The objective is:

```text
small chunks → retrieval precision
large chunks → context completeness
```

---

## Context Compression

Retrieved information can be reduced to retain only the portions most relevant to the query.

---

# 13. Retrieval Evaluation

A manually labeled evaluation dataset was created with:

```text
12 total questions
10 answerable
2 intentionally unanswerable
```

Examples included questions about:

- Transformer decoder masking
- autoregressive generation
- encoder vs decoder behavior
- scaled dot-product attention
- multi-head attention
- positional encoding
- the Platonic Representation Hypothesis
- representational alignment
- astronomy as a PRH testbed

Unanswerable examples included questions such as:

```text
What was the exact USD cost to train the original Transformer?
```

and an unrelated financial-crisis question.

---

# 14. Retrieval Metrics

Three metrics were implemented manually.

## Hit@K

Measures whether at least one relevant result appears in the first K retrieved results.

## Recall@K

Measures what fraction of all known relevant chunks were retrieved.

## Reciprocal Rank / MRR

Rewards systems that rank the first relevant result near the top.

---

# 15. Retrieval Ablation Results

## Dense Retrieval

```text
Hit@5   = 0.900
Recall  = 0.667
MRR     = 0.683
```

## BM25

```text
Hit@5   = 0.700
Recall  = 0.517
MRR     = 0.408
```

## Dense + BM25 + RRF

```text
Hit@5   = 0.900
Recall  = 0.667
MRR     = 0.675
```

## Dense + Cross-Encoder Reranker

```text
Hit@5   = 0.900
Recall  = 0.717
MRR     = 0.783
```

## Hybrid + Cross-Encoder Reranker

```text
Hit@5   = 0.900
Recall  = 0.717
MRR     = 0.783
```

---

# 16. Retrieval Architecture Decision

The strongest result was:

```text
Dense Retrieval
+
Cross-Encoder Reranking
```

Hybrid retrieval plus reranking produced essentially the same metrics.

Therefore, the simpler architecture was selected:

```text
Query
↓
Dense Top 20
↓
Cross-Encoder Reranking
↓
Top 5
```

This reflects an important engineering principle followed throughout the project:

> More components do not automatically produce a better RAG system.

Complexity was added only when evaluation showed a benefit.

---

# 17. Retrieval Validation

The local production pipeline was compared against the evaluated Colab implementation.

For the question:

```text
Why is masking necessary in the Transformer decoder?
```

Dense Top-20 retrieval produced:

```text
[11, 12, 130, 10, 18, 19, 134, 103, 34, 206,
 126, 38, 128, 141, 7, 2, 23, 93, 85, 50]
```

Cross-encoder reranking produced:

```text
[19, 12, 18, 206, 11]
```

This matched the previously evaluated notebook baseline.

The most relevant retrieved chunks included the Transformer paper sections explaining that decoder masking:

- prevents attention to subsequent positions
- preserves the autoregressive property
- ensures predictions for position `i` depend only on known outputs at earlier positions

---

# 18. Generation Evaluation

Generation was evaluated separately from retrieval because strong retrieval does not guarantee a correct final answer.

Two generator sizes were tested.

---

## Qwen2.5-1.5B-Instruct

With the original prompt, provisional manual evaluation was approximately:

```text
Completeness: ~0.675
Faithfulness: ~0.81
Correctness:  ~0.75
Abstention:    1.00
```

After improving the grounding prompt:

```text
Completeness: ~0.71
Faithfulness: ~0.84
Correctness:  ~0.83
Abstention:    1.00
```

However, important failures remained.

### Example: scaled dot-product attention

The model incorrectly explained the `sqrt(dk)` scaling even though the correct evidence had been retrieved.

### Example: representational alignment

The model reconstructed an incorrect formula by combining information from multiple chunks.

These errors showed that retrieval was not the only bottleneck.

---

# 19. Generator Model Decision

The generator was upgraded to:

```text
Qwen2.5-7B-Instruct
```

using quantized inference on Google Colab.

With the same retrieval results and grounded prompt, provisional manual evaluation improved to approximately:

```text
Completeness: ~0.83
Faithfulness: ~0.97
Correctness:  ~0.95
Abstention:    1.00
```

The larger model:

- correctly explained the attention scaling mechanism
- stopped inventing unsupported formulas
- used retrieved evidence more reliably
- preserved the ability to abstain

This experiment showed that:

> Generator capability was a major bottleneck after retrieval had already become sufficiently strong.

---

# 20. Grounded Generation

The production prompt instructs the model to answer based only on the provided context.

If the documents do not support the requested information, the model should abstain.

Example:

### Question

```text
What was the exact USD cost to train the original Transformer?
```

### Result

```text
I do not know based on the provided documents.
```

This behavior was verified in a real end-to-end test.

---

# 21. Remote Model Architecture

Running Qwen2.5-7B locally was not practical on the available CPU-only development machine.

Therefore the system was split into two environments.

## Local Machine

Runs:

```text
Index
Embedding model
Retriever
Reranker
FastAPI application
```

## Google Colab GPU

Runs:

```text
Qwen2.5-7B-Instruct
FastAPI /generate endpoint
```

The connection is:

```text
Local RAG Pipeline
      ↓ HTTPS
ngrok Tunnel
      ↓
Colab FastAPI
      ↓
Qwen2.5-7B
```

---

# 22. Remote Generation API

The Colab model server exposes:

```text
POST /generate
```

Expected request structure:

```json
{
  "question": "Why is masking necessary?",
  "context": [
    {
      "text": "...",
      "document": "...",
      "page": 3
    }
  ]
}
```

The endpoint returns:

```json
{
  "question": "...",
  "answer": "..."
}
```

---

# 23. API Authentication

The generation API is protected with an API key.

The client sends:

```text
X-API-Key
```

in the HTTP header.

Secrets are stored in environment variables rather than source code.

The repository includes:

```text
.env.example
```

but the real:

```text
.env
```

is ignored by Git.

Example configuration:

```env
MODEL_SERVER_URL=https://your-ngrok-url.ngrok-free.app
MODEL_API_KEY=your_api_key
```

Never commit real API keys.

---

# 24. Local Generation Client

`src/generation.py` sends the final query and retrieved context to the remote model server.

It handles:

- JSON request construction
- `X-API-Key` authentication
- HTTP timeout
- connection failures
- upstream server failures
- response parsing

A request timeout of:

```text
60 seconds
```

is currently used.

---

# 25. Model Server Error Handling

A custom exception is used:

```text
ModelServerError
```

It separates dependency failures from unexpected local application failures.

Examples classified as model-server failures include:

- connection refused
- request timeout
- unavailable ngrok tunnel
- upstream HTTP `5xx`
- ngrok returning a `404` HTML page when the tunnel/backend is unavailable

These failures are translated into:

```text
HTTP 503 Service Unavailable
```

instead of incorrectly exposing them as internal application bugs.

---

# 26. Unified RAG Pipeline

The production `RAGPipeline` loads:

- embedding model
- reranker
- saved corpus
- saved embeddings

once during initialization.

The query flow is:

```text
RAGPipeline.query(question)
      ↓
retrieve(question)
      ↓
Dense Top 20
      ↓
Rerank Top 5
      ↓
Build context
      ↓
Remote generation
      ↓
Build sources
      ↓
Return result
```

The embedding model and reranker are not reloaded for every query.

---

# 27. Local FastAPI Service

The complete RAG application is exposed through:

```text
POST /query
```

Request:

```json
{
  "question": "Why is masking necessary in the Transformer decoder?"
}
```

Example response:

```json
{
  "question": "Why is masking necessary in the Transformer decoder?",
  "answer": "To preserve the auto-regressive property, masking is necessary in the Transformer decoder to prevent positions from attending to subsequent positions.",
  "sources": [
    {
      "page": 5,
      "document": "data\\1706.03762v7.pdf"
    },
    {
      "page": 3,
      "document": "data\\1706.03762v7.pdf"
    }
  ],
  "latency": {
    "retrieval": 2.69,
    "generation": 3.86,
    "total": 6.55
  }
}
```

Multiple retrieved chunks can currently produce repeated document/page entries in `sources`.

This is cosmetic and does not affect retrieval or generation behavior.

---

# 28. Latency Measurement

The production pipeline measures:

```text
retrieval latency
generation latency
total query latency
```

Timing uses:

```python
time.perf_counter()
```

Example real end-to-end result:

```text
Retrieval:  2.691 s
Generation: 3.860 s
Total:      6.551 s
```

These numbers depend on:

- local CPU performance
- network latency
- Colab GPU availability
- ngrok routing
- model inference speed

---

# 29. API Error Handling

The local API currently handles several important cases.

## Empty Question

Request:

```json
{
  "question": "   "
}
```

Response:

```text
400 Bad Request
```

```json
{
  "detail": "Question cannot be empty."
}
```

---

## Remote Model Unavailable

If Colab/ngrok/model inference is unavailable:

```text
503 Service Unavailable
```

```json
{
  "detail": "Model server unavailable"
}
```

---

## Unexpected Internal Failure

Unexpected errors return:

```text
500 Internal Server Error
```

```json
{
  "detail": "Internal server error"
}
```

Low-level exception details are intentionally not returned to API clients.

---

# 30. Automated Testing

The API uses `pytest` and FastAPI's `TestClient`.

Tests deliberately mock the RAG pipeline when appropriate so unit tests do not depend on:

- Colab
- ngrok
- GPU availability
- external network access

Current automated API tests cover:

### 1. Empty Question

Expected:

```text
400
```

### 2. Model Server Unavailable

`rag_pipeline.query()` is monkeypatched to raise:

```text
ModelServerError
```

Expected:

```text
503
```

### 3. Successful Query

The pipeline is mocked with a valid response.

Expected:

```text
200
```

The response is validated for:

```text
question
answer
sources
latency
```

### 4. Unexpected Internal Failure

The pipeline is mocked to raise a generic exception.

Expected:

```text
500
```

Current automated result:

```text
4 passed
```

Tests can be run with:

```bash
python -m pytest tests/test_api.py -v
```

---

# 31. Real Integration Testing

In addition to mocked API tests, the complete system was tested end to end.

The integration path included:

```text
Local FastAPI
↓
RAGPipeline
↓
Dense retrieval
↓
Cross-encoder reranking
↓
Top-5 context
↓
ngrok
↓
Colab FastAPI
↓
Qwen2.5-7B
↓
Final answer
```

---

# 32. Real Answerable Integration Test

Question:

```text
Why is masking necessary in the Transformer decoder?
```

Example result:

```text
To preserve the auto-regressive property, masking is necessary in the Transformer decoder to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
```

The response was correctly grounded in retrieved Transformer paper chunks.

---

# 33. Real Abstention Integration Test

Question:

```text
What was the exact USD cost to train the original Transformer?
```

Result:

```text
I do not know based on the provided documents.
```

Measured latency:

```text
Retrieval:  2.6914542 s
Generation: 3.8597552 s
Total:      6.5512221 s
```

This verifies that the final production path preserves the abstention behavior observed during the evaluation stage.

---

# 34. Building the Index

Place PDF files inside:

```text
data/
```

Then run:

```bash
python -m scripts.build_index
```

The indexing pipeline performs:

```text
PDF loading
↓
cleaning
↓
sentence-aware chunking
↓
embedding generation
↓
artifact persistence
```

Expected artifacts:

```text
artifacts/corpus.json
artifacts/embeddings.npy
```

---

# 35. Running the Local API

From the project root:

```bash
uvicorn api.main:app --reload
```

The API should start at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 36. Example Query

Using Swagger or another HTTP client:

```http
POST /query
```

Body:

```json
{
  "question": "Why is masking necessary in the Transformer decoder?"
}
```

The pipeline automatically performs retrieval, reranking and generation.

The user does not provide context manually.

---

# 37. Installation

Create and activate a virtual environment.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For tests, ensure pytest is installed:

```bash
python -m pip install pytest
```

---

# 38. Environment Configuration

Create:

```text
.env
```

using:

```text
.env.example
```

Example:

```env
MODEL_SERVER_URL=https://your-model-server-url
MODEL_API_KEY=your-model-api-key
```

The real `.env` must never be committed.

---

# 39. Colab Environment

The remote GPU environment uses a separate dependency file:

```text
requirements-colab.txt
```

The Colab environment runs:

- Transformers
- PyTorch
- Accelerate
- bitsandbytes
- FastAPI
- Uvicorn
- pyngrok

The Qwen model and model-serving API remain separate from the CPU-oriented local application.

---

# 40. ngrok

ngrok is used only as a development tunnel between the local machine and the temporary Colab FastAPI server.

The public URL can change between Colab sessions.

When a new URL is generated, update:

```env
MODEL_SERVER_URL=...
```

in the local `.env`.

The ngrok authtoken is stored securely in Colab Secrets rather than hardcoded into the notebook.

---

# 41. Design Decisions

## Why build RAG manually first?

Frameworks can hide important details.

Implementing retrieval manually made it possible to understand:

- embedding representation
- similarity functions
- candidate retrieval
- rank fusion
- reranking
- context selection
- generation failures
- evaluation

This creates a better foundation for later learning LangChain and LlamaIndex.

---

## Why dense retrieval instead of BM25?

Dense retrieval clearly outperformed BM25 on the current corpus.

```text
Dense Hit@5: 0.900
BM25 Hit@5:  0.700
```

---

## Why not keep hybrid retrieval?

Hybrid retrieval did not improve results after reranking.

```text
Dense + reranker:
Recall = 0.717
MRR    = 0.783

Hybrid + reranker:
Recall = 0.717
MRR    = 0.783
```

The simpler solution was therefore selected.

---

## Why retrieve 20 and rerank to 5?

Dense retrieval is relatively cheap and can produce a broader candidate set.

The cross-encoder is more expensive but more accurate.

Therefore:

```text
cheap broad retrieval → 20
expensive precise reranking → 5
```

provides a useful precision/latency tradeoff.

---

## Why Qwen2.5-7B instead of Qwen2.5-1.5B?

The 1.5B model sometimes failed even when retrieval was correct.

The 7B model substantially improved:

- reasoning
- faithfulness
- correctness
- resistance to unsupported reconstruction

The experiment demonstrated the importance of separately evaluating retrieval and generation.

---

## Why remote inference?

The development machine is CPU-only.

Running 7B inference locally would make iteration unnecessarily slow.

Google Colab provides GPU access while the local machine remains responsible for the rest of the application.

---

## Why persist embeddings?

Embedding generation should not be repeated every time the API starts.

Persistence allows:

```text
build index once
↓
load artifacts on startup
↓
query repeatedly
```

---

## Why custom error types?

`ModelServerError` makes it possible to distinguish:

```text
dependency unavailable
```

from:

```text
local application bug
```

This allows correct HTTP behavior:

```text
503 → dependency unavailable
500 → internal failure
```

---

# 42. Important Lessons From the Project

Several practical lessons emerged from the experiments.

### 1. Better retrieval is not always more retrieval