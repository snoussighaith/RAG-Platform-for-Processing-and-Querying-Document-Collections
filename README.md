# RAG Platform for Processing and Querying Document Collections

A configurable Retrieval-Augmented Generation (RAG) project focused on document preparation, retrieval benchmarking, hybrid search, and neural reranking. The repository provides a Docker-based environment and evaluation assets to compare retrieval strategies before relying on an LLM to answer questions over a document collection.

> **Project status:** the retrieval and reranking benchmark has been implemented and evaluated. Local Ollama models have been downloaded and tested for availability; end-to-end answer generation, source-groundedness checks, and faithfulness evaluation are still work in progress.

## Table of contents

- [Overview](#overview)
- [What this project explores](#what-this-project-explores)
- [Retrieval pipeline](#retrieval-pipeline)
- [Evaluation and current results](#evaluation-and-current-results)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Current limitations and next steps](#current-limitations-and-next-steps)

## Overview

Retrieval-Augmented Generation combines information retrieval with text generation. Instead of asking a language model to answer from its parameters alone, a RAG system first retrieves relevant passages from a document collection and then provides those passages as context to the model.

This project studies the retrieval stage in detail. It compares lexical, semantic, and hybrid retrieval methods, then measures whether a Cross-Encoder reranker improves the relevance of the retrieved passages.

The experiments use a BBC News document collection containing **1,440 articles across five categories**, after cleaning and duplicate removal.

## What this project explores

- Document cleaning, normalization, deduplication, and metadata preservation.
- Text chunking and the effect of chunk size and overlap on retrieval.
- Lexical retrieval with BM25.
- Semantic retrieval with Sentence-Transformer embeddings.
- Hybrid retrieval with Reciprocal Rank Fusion (RRF).
- Candidate reranking with a Cross-Encoder.
- Quantitative evaluation using information-retrieval metrics.
- A reproducible, containerized development environment with Docker Compose and Ollama.

## Retrieval pipeline

```text
Document collection
        |
        v
Cleaning, normalization and deduplication
        |
        v
Chunking + source metadata
        |
        +-----------------------+
        |                       |
        v                       v
   BM25 retrieval         Dense retrieval
   (lexical search)       (embeddings)
        |                       |
        +-----------+-----------+
                    |
                    v
          Hybrid fusion (RRF)
                    |
                    v
       Candidate passages / top-k
                    |
                    v
       Cross-Encoder reranking
                    |
                    v
       Ranked passages for context
                    |
                    v
       LLM answer generation
       (integration in progress)
```

### 1. Document preparation and chunking

The BBC News corpus is cleaned and deduplicated, while metadata is retained to trace retrieved chunks back to their source articles.

Three chunking strategies are compared:

| Strategy | Chunking rule |
|---|---|
| `fixed_100_o20` | Fixed windows of 100 words with an overlap of 20 words |
| `fixed_200_o40` | Fixed windows of 200 words with an overlap of 40 words |
| `sent_150` | Sentence-based segmentation with a maximum target of 150 words |

### 2. Lexical retrieval: BM25

BM25 ranks documents using term-frequency and inverse-document-frequency signals. It is used as a lexical baseline.

The benchmark uses `rank_bm25` with:

- `k1 = 1.5`
- `b = 0.75`

### 3. Dense retrieval

Dense retrieval represents text as numerical embeddings and compares query and passage vectors in a semantic space.

The evaluated embedding models include:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-small`

The experiments compare embedding configurations and account for model-specific query/document prefixes where applicable.

### 4. Hybrid retrieval: Reciprocal Rank Fusion

BM25 and dense retrieval capture different relevance signals. Reciprocal Rank Fusion combines their ranked lists without requiring their raw scores to be directly comparable.

For a document \(d\), the RRF score is:

\[
\operatorname{RRF}(d) = \sum_{r \in R(d)} \frac{1}{60 + \operatorname{rank}_r(d)}
\]

where \(R(d)\) is the set of ranked lists containing document \(d\).

### 5. Neural reranking: Cross-Encoder

A Cross-Encoder jointly evaluates a query and a candidate passage. It is applied to an initial candidate set (top 20) to refine the ranking.

The reranker used in the experiments is:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The benchmark compares six retrieval configurations:

1. BM25
2. Dense retrieval
3. RRF
4. BM25 + reranking
5. Dense retrieval + reranking
6. RRF + reranking

## Evaluation and current results

The retrieval evaluation uses a question set containing 50 questions. Four questions are classified as out-of-corpus and excluded from retrieval scoring, leaving **42 in-corpus questions** for the reported retrieval evaluation.

| Experiment | Observed result |
|---|---:|
| BM25 baseline | Recall@5 = **0.976** |
| Dense retrieval + Cross-Encoder | Mean MRR improvement of **+0.094** vs. dense retrieval alone, across six configurations |
| RRF | Mean MRR improvement of **+0.065** vs. dense retrieval alone |
| RRF + reranking | MRR improvement of **+0.039** vs. BM25 in the primary comparison |

These results should be interpreted within the current evaluation set and experimental configurations. The observed improvements do not establish that every hybrid or reranking configuration is statistically superior to every baseline.

### Evaluation methodology

The project uses standard information-retrieval metrics:

- **Recall@k:** whether relevant documents appear within the first *k* retrieved results.
- **MRR (Mean Reciprocal Rank):** rewards systems that place the first relevant result near the top.
- **nDCG (Normalized Discounted Cumulative Gain):** evaluates ranking quality while accounting for relevance and position.
- **Latency:** measures retrieval response time.

Bootstrap confidence intervals are used in the evaluation workflow to examine uncertainty in metric differences.

## Technology stack

| Area | Tools and methods |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Lexical retrieval | BM25, `rank_bm25` |
| Embeddings | Sentence-Transformers, Hugging Face |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Local LLM runtime | Ollama |
| Models tested locally | Qwen 2.5 3B Instruct, Llama 3.2 3B |
| Environment | Docker, Docker Compose |
| Configuration | YAML |

## Repository structure

The repository currently includes the following top-level resources:

```text
.
├── data/
│   └── eval/                  # Retrieval evaluation questions / assets
├── results/                   # Experiment outputs
├── scripts/                   # Project scripts
├── config.yaml                # Model and retrieval configuration
├── Dockerfile                 # RAG service image
├── docker-compose.yml         # Main service orchestration
├── docker-compose.gpu.yml     # GPU-oriented Compose configuration
├── requirements.txt           # Python dependencies
├── requirements.lock.txt      # Pinned environment snapshot
└── README.md
```

The exact contents of `scripts/`, `data/eval/`, and `results/` may evolve as experiments are extended.

## Prerequisites

- Docker Desktop or Docker Engine with the Docker Compose plugin.
- Git.
- Sufficient disk space for Python dependencies, embedding models, and local LLM model files.
- An NVIDIA GPU is optional. The base environment can run CPU-based embedding and retrieval workloads; GPU acceleration depends on the host and the selected Compose configuration.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/snoussighaith/RAG-Platform-for-Processing-and-Querying-Document-Collections.git
cd RAG-Platform-for-Processing-and-Querying-Document-Collections
```

### 2. Configure the environment

Review `config.yaml` and the environment variables referenced by `docker-compose.yml`. If the Compose configuration expects a `.env` file, create it from the values required by that configuration before starting the services. Do not commit secrets or private tokens.

### 3. Build and start the services

```bash
docker compose up -d --build
```

Check the service status:

```bash
docker compose ps
```

View logs when troubleshooting:

```bash
docker compose logs -f
```

### 4. Check Ollama and available models

If the Ollama service is enabled in the Compose configuration, list the models available inside the service:

```bash
docker compose exec ollama ollama list
```

Pull the local models used during the initial experiments:

```bash
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama pull llama3.2:3b
```

Check that the RAG container can reach the Ollama API:

```bash
docker compose exec rag curl http://ollama:11434/api/tags
```

> Model downloads can require several gigabytes of disk space. The exact memory and inference performance depend on the host machine.

### 5. Run project scripts

The project scripts are located in `scripts/`. Inspect the available script names and their command-line arguments before running them:

```bash
docker compose exec rag ls -la scripts
```

Run a script using the service's Python environment, following the arguments documented in that script:

```bash
docker compose exec rag python scripts/<script_name>.py
```

Replace `<script_name>.py` with an existing script in the repository. The exact execution sequence depends on the current script interfaces and the data available locally.

### 6. Stop the services

```bash
docker compose down
```

To remove Compose-managed volumes as well (including cached model data), use this only if you intentionally want to delete those persisted files:

```bash
docker compose down -v
```

## Configuration

The main experiment settings are defined in `config.yaml`. Review this file before running experiments, especially:

- embedding model selection;
- query/document prefixes required by some embedding models;
- retrieval `top_k`;
- reranker model;
- local LLM model selection;
- generation parameters, when the generation stage is enabled.

Keep experiment settings explicit so that results can be reproduced and compared fairly.

## Reproducing the experiments

For a meaningful comparison:

1. Use the same corpus version and document preprocessing rules.
2. Keep the evaluation question set and relevance judgments unchanged across retrieval methods.
3. Record the chunking strategy and embedding model for each run.
4. Keep retrieval depth and reranking candidate depth consistent.
5. Record Recall@k, MRR, nDCG, and latency for every configuration.
6. Report confidence intervals when comparing metric differences.
7. Separate in-corpus questions from out-of-corpus questions.
8. Record model versions, configuration files, and random seeds with each experiment.

## Current limitations and next steps

The current work is primarily a retrieval and reranking benchmark. The following improvements are planned or remain to be completed:

- Complete the end-to-end connection between retrieved passages and local LLM answer generation.
- Add source citations to generated answers and verify that claims are supported by retrieved passages.
- Evaluate answer faithfulness, relevance, and correctness separately from retrieval quality.
- Add a repeatable command-line evaluation entry point and document exact script arguments.
- Improve experiment tracking and persist complete configuration metadata with every result.
- Extend the evaluation set and test additional document collections.
- Document resource usage and compare CPU/GPU execution where relevant.

## Notes

- Retrieval metrics evaluate the quality of retrieved evidence; they do **not** by themselves measure the factual correctness of generated answers.
- Local model availability does not mean that the complete generation and evaluation pipeline is already implemented.
- Results are specific to the current corpus, question set, chunking strategies, and model configurations.

## Author

**Repository:** [RAG Platform for Processing and Querying Document Collections](https://github.com/snoussighaith/RAG-Platform-for-Processing-and-Querying-Document-Collections)

---

*Built as a hands-on project to study and evaluate document retrieval techniques for Retrieval-Augmented Generation systems.*
