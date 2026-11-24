# Hybrid RAG

Hybrid RAG is a Python retrieval-augmented generation application for asking questions about PDF documents. It converts PDFs with Docling, splits the Markdown into chunks, stores embeddings in LanceDB, combines vector and full-text retrieval, reranks candidates with a cross-encoder, and generates grounded answers in a Gradiocle chat interface.

![Screenshot](Screenshot.png)

## Langsmith trace

![Screenshot](Screenshot1.png)

## Evaluation metrics

```json
{
  "queries_evaluated": 67,
  "k": 5,
  "similarity": {
    "recall_at_5": 0.42288557213930345,
    "precision_at_5": 0.11343283582089553,
    "mrr": 0.38656716417910447,
    "ndcg_at_5": 0.3686124628832584
  },
  "sparse": {
    "recall_at_5": 0.7450248756218905,
    "precision_at_5": 0.18507462686567164,
    "mrr": 0.714179104477612,
    "ndcg_at_5": 0.6948259860837872
  },
  "mmr": {
    "recall_at_5": 0.46766169154228854,
    "precision_at_5": 0.11940298507462686,
    "mrr": 0.39875621890547264,
    "ndcg_at_5": 0.380454226411776
  },
  "hybrid": {
    "recall_at_5": 0.7189054726368159,
    "precision_at_5": 0.18507462686567164,
    "mrr": 0.7082089552238806,
    "ndcg_at_5": 0.6753480847363774
  }
}
```

## Features

- PDF document ingestion with Docling
- Markdown chunking with deterministic document and chunk IDs
- Persistent LanceDB vector and full-text storage
- Hugging Face embeddings
- LanceDB hybrid search with reciprocal-rank fusion
- Cross-encoder reranking
- Streaming answers from OpenRouter, OpenAI, Ollama, or another LangChain chat provider
- A citation-aware chat prompt
- Retrieval evaluation utilities for similarity, sparse, MMR, and hybrid search
- Optional legacy ChromaDB vector-store support

## Tech Stack

- Python 3.14
- LangChain
- LanceDB
- Sentence Transformers
- Hugging Face Sentence Transformers
- Docling
- Gradio
- `uv`

## Project Structure

```text
src/naive_rag/
├── config/
│   └── setting.py          # Environment-backed application settings
├── ingestion/
│   └── ingestion_service.py
├── persistance/
│   └── vector_store.py     # LanceDB and legacy ChromaDB stores
├── model/
│   └── model.py            # Embedding and chat model factories
├── retrival/
│   ├── hybrid_retriever.py # LanceDB hybrid search and reranking
│   └── retriver_service.py # Retrieval chain and answer prompt
├── evalutation/
│   ├── generate_eval_questions.py
│   └── retrieval_evaluation.py
├── source/
│   └── 2611.14027v1.pdf     # Example source document
└── main.py                 # Gradio application entry point
```

## Prerequisites

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- An API key if the selected chat provider requires one
- Enough disk space for embedding models and the LanceDB database

## Installation

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/vpariha/hybrid-rag.git
cd hybrid-rag
uv sync
```

## Configuration

Create a `.env` file in the repository root. The application loads environment variables with `python-dotenv`.

```dotenv
# LanceDB vector and hybrid search store
LANCEDB_URI=./lancedb
LANCEDB_TABLE_NAME=documents

# Optional legacy ChromaDB settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=my_collection

# Retrieval
TOP_K=5
RERANK_CANDIDATE_K=20
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2

# Embeddings
EMBEDDING_MODEL_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Chat model
CHAT_MODEL_PROVIDER=openrouter
CHAT_MODEL_NAME=openrouter/free
CHAT_MODEL_TEMPERATURE=0.5

# Add the key required by your selected provider
OPENROUTER_API_KEY=your_api_key
```

Other provider-specific environment variables, such as `OPENAI_API_KEY`, can be added when changing `CHAT_MODEL_PROVIDER`.

> Never commit `.env` or API keys. The repository ignores `.env`.

### Supported Configuration Fields

| Variable                   | Purpose                                            | Example                                  |
| -------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `LANCEDB_URI`              | LanceDB storage URI                                | `./lancedb`                              |
| `LANCEDB_TABLE_NAME`       | LanceDB table name                                 | `documents`                              |
| `CHROMA_PERSIST_DIRECTORY` | Directory used for persistent vector storage       | `./chroma_db`                            |
| `CHROMA_COLLECTION_NAME`   | ChromaDB collection name                           | `my_collection`                          |
| `TOP_K`                    | Number of chunks retrieved per question            | `5`                                      |
| `RERANK_CANDIDATE_K`       | Candidate count loaded for reranking configuration | `20`                                     |
| `RERANKER_MODEL_NAME`      | Cross-encoder used for reranking                   | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| `EMBEDDING_MODEL_PROVIDER` | LangChain embeddings provider                      | `huggingface`                            |
| `EMBEDDING_MODEL_NAME`     | Embedding model identifier                         | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHAT_MODEL_PROVIDER`      | LangChain chat provider                            | `openrouter`                             |
| `CHAT_MODEL_NAME`          | Chat model identifier                              | `openrouter/free`                        |
| `CHAT_MODEL_TEMPERATURE`   | Chat model sampling temperature                    | `0.5`                                    |

## Ingest Documents

Place PDF files in `src/naive_rag/source/`, then run the project module. Ingestion is currently disabled in `main()`; invoke `IngestionService.ingest_data_using_dolcing()` explicitly for a first-time setup:

```bash
uv run python -c "from naive_rag.ingestion.ingestion_service import IngestionService; IngestionService().ingest_data_using_dolcing('src/naive_rag/source')"
```

The ingestion process:

1. Discovers all PDF files below the source directory.
2. Converts each PDF to Markdown with Docling.
3. Generates stable document and chunk hashes.
4. Stores chunks in batches of 10 in LanceDB.

Re-running ingestion reuses the same deterministic chunk IDs.

## Run the Application

Start the Gradio chat interface:

```bash
uv run python -m naive_rag.main
```

Open the local URL printed by Gradio, place a question in the chat input, and receive a streamed response based on the retrieved document context.

## Retrieval Evaluation

The evaluation utilities can generate a labeled question dataset and compare dense similarity, sparse/full-text search, maximum marginal relevance (MMR), and hybrid retrieval.

Generate evaluation questions from ingested chunks by calling `generate_eval_question(context, question_count)` from `generate_eval_questions.py`. Results are appended to `evaluation.json`.

Run retrieval metrics:

```bash
uv run python -m naive_rag.evalutation.retrieval_evaluation
```

The evaluation reports:

- Recall@K
- Precision@K
- Mean Reciprocal Rank
- NDCG@K

All four retrieval strategies are evaluated with `K=5` by default.

## How It Works

```text
PDFs
  -> Docling Markdown conversion
  -> Markdown chunk splitting
  -> Hugging Face embeddings
  -> Persistent LanceDB table
  -> Vector and full-text search
  -> Reciprocal-rank fusion
  -> Cross-encoder reranking
  -> Top-K hybrid retrieval
  -> Prompt with retrieved context
  -> Streaming chat model response
```

Answers are intended to be grounded only in retrieved document context and include the source metadata available with each chunk.
