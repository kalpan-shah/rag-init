#  Movie Search Engine

A foundational implementation of **Retrieval-Augmented Generation (RAG)** principles. This project demonstrates two core retrieval methods: **Keyword Search** (BM25/TF-IDF style) and **Semantic Search** (Vector-based).

## Objective of this project
Understanding RAG

---

## Getting Started
### 1. Clone the Repository

```bash
git clone https://github.com/kalpan-shah/rag-init.git
cd rag-init

```

### 2. Prerequisites

This project uses `uv` for ultra-fast Python package and project management.

* **Install uv** 
* **Sync Environment:**
```bash
uv sync
```

### 3. Data Setup

You will need to download the movie dataset and configure the stop words for the keyword search engine.

* **Dataset:** Download the [Course Movies JSON](https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json) and place it in the project root in a designated `data/` folder.
* **Stop Words:** Ensure the [stop word](https://www.boot.dev/lessons/8afe99b4-d351-4794-becc-08a4661bc2df) list is available to the application. This helps the keyword search focus on meaningful terms by filtering out common words (e.g., "the", "and", "is").

---

## 🛠️ Usage

As of Now the project provides two CLI tools to interact with the movie database.

### Keyword Search

Uses exact word matching and frequency analysis to find relevant movies.

```bash
uv run cli/keyword_search_cli.py [command] [args]
```

### Semantic Search

Uses embeddings to find movies based on meaning and intent, even if the exact keywords don't match.

```bash
uv run cli/semantic_search_cli.py [command] [args]
```

---

## Project Structure

* `cli/`: Command-line interface scripts for searching.
* `data/`: Directory storing dataset and stop words.
* `pyproject.toml`: Project dependencies and configuration managed by `uv`.

---

> **Note:** When running the semantic search for the first time, the system may download a pre-trained embedding model. Ensure you have a stable internet connection.