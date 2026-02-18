# sc-ops
A lightweight collection of reusable utility functions for single-cell RNA-seq, ATAC-seq, and multimodal analysis

## Pre-commit

This repo uses `pre-commit` to run code-quality checks (Ruff lint + import sorting) and formatting before commits.

```bash
uv sync --group dev
pre-commit install
```

To run on all files:

```bash
pre-commit run --all-files
```
