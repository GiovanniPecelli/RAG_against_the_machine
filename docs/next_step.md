# Next Steps

This document tracks the work that remains. Semantic embeddings, BM25 and
semantic retrieval, RRF fusion, dataset-search mode selection, and the
embedding progress bar are implemented. The indexer now loads the embedding
model only when semantic indexing is requested, and the CLI limits chunks to
2,000 characters.

## Outstanding Work

### 1. Expose retrieval mode in single-query search

`search_dataset` accepts `bm25`, `semantic`, and `hybrid`. The single-query
`search` CLI command still uses its default BM25 mode. Add a `mode` argument to
`RAGCLI.search` and pass it through to `Retriever.retrieve`, so all three
strategies can be tried on one question from the CLI.

### 2. Finish graceful input-error handling

`search_dataset` now checks the dataset structure and skips malformed question
records. Review the other CLI paths, especially `answer_dataset` and
`evaluate`, for missing fields, wrong JSON types, and missing or unreadable
files. Invalid input should produce a clear message and should not end the CLI
with an unhandled traceback, as required by the subject.

### 3. Run lint and fix reported issues

`flake8` and `mypy` are now declared as development dependencies. Run the
required Makefile lint target in the project environment and resolve any
reported issues. The strict lint target is optional.

### 4. Complete retrieval and performance evaluation

Run the public documentation and code datasets in all three modes, evaluate
each result set against its answered dataset, and record the actual metrics
below. Also record BM25 indexing time and retrieval time for 200 questions.

The subject targets are Recall@5 of at least 80% for documentation and 50% for
code, indexing within 5 minutes, and retrieval of 200 questions within 90
seconds.

| Dataset | Metric | BM25 | Semantic | Hybrid (RRF) |
| :--- | :--- | :---: | :---: | :---: |
| Documentation | Recall@1 | TBD | TBD | TBD |
| Documentation | Recall@3 | TBD | TBD | TBD |
| Documentation | Recall@5 | TBD | TBD | TBD |
| Documentation | Recall@10 | TBD | TBD | TBD |
| Code | Recall@1 | TBD | TBD | TBD |
| Code | Recall@3 | TBD | TBD | TBD |
| Code | Recall@5 | TBD | TBD | TBD |
| Code | Recall@10 | TBD | TBD | TBD |
| Performance | BM25 indexing time | TBD | — | — |
| Performance | Retrieval time for 200 questions | TBD | TBD | TBD |

Example commands for generating the six result sets:

```bash
# Documentation questions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode bm25 --k 10 --save_directory data/output/search_results_bm25/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode semantic --k 10 --save_directory data/output/search_results_semantic/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode hybrid --k 10 --save_directory data/output/search_results_hybrid/UnansweredQuestions

# Code questions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode bm25 --k 10 --save_directory data/output/search_results_bm25/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode semantic --k 10 --save_directory data/output/search_results_semantic/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode hybrid --k 10 --save_directory data/output/search_results_hybrid/UnansweredQuestions
```

Evaluate each generated file with `evaluate`, using the matching file under
`data/datasets/AnsweredQuestions/` as the ground truth.

## Deferred: Bonus 3 Incremental Indexing and Artifact Consistency

The current indexer rebuilds the full corpus. Revisit this section when
implementing incremental indexing:

- Track which chunks and embeddings belong to each source file.
- When a file changes, update its chunks and embeddings while reusing those
  for unchanged files.
- Keep global BM25 statistics consistent with the complete corpus.
- Detect stale or mismatched artifacts, such as `index.pkl` paired with
  vectors from a different chunk set, and explain how to rebuild or update.
- Choose a manifest or equivalent metadata to verify that the index and vector
  files describe the same corpus version.
