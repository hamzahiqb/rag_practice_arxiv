# rag_practice_arxiv

RAG practice project using arXiv papers.

## Getting Started

1. Activate the environment:

```bash
source .venv/bin/activate
# or use uv run to run commands directly
```

2. Launch JupyterLab:

```bash
uv run jupyter lab
```

3. Run tests:

```bash
uv run pytest
```

4. Lint code:

```bash
uv run ruff check .
```

## Project Structure

```
rag_practice_arxiv/
├── data/              # Raw and processed data
├── notebooks/         # Jupyter notebooks
├── src/
│   └── rag_practice_arxiv/
│       └── __init__.py
├── tests/             # Unit tests
├── .gitignore
├── pyproject.toml     # Project config & dependencies
└── README.md
```
