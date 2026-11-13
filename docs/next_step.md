# Next Steps

This document tracks the work that remains. Semantic embeddings, BM25 and
semantic retrieval, RRF fusion, mode selection in single-query and dataset
search, and the embedding progress bar are implemented. The indexer now loads
the embedding model only when semantic indexing is requested, and the CLI
limits chunks to 2,000 characters.

## Required Checks Completed

The BM25 retrieval targets and performance limits were checked in this
workspace. Recall was evaluated with the project's `evaluate` command against
the answered public datasets. The timing measurements include CLI startup and
result-file writing where applicable.

| Requirement | Subject target | Measured result | Status |
| :--- | :--- | :--- | :---: |
| Documentation Recall@5 | At least 0.800 | 0.840 (100 questions) | Pass |
| Code Recall@5 | At least 0.500 | 0.808 (99 questions) | Pass |
| BM25 indexing | At most 5 minutes | 5.4 seconds (1,969 files, 13,349 chunks) | Pass |
| BM25 retrieval | At most 90 seconds for 200 questions | 8.8 seconds | Pass |

## Remaining Work

### 1. Bonus 3: Incremental Indexing and Artifact Consistency

The current indexer rebuilds the full corpus. The implementation should:

- Save a content hash for every source file so a later `index` run can identify
  added, changed, unchanged, and removed files.
- Re-chunk changed and new files only; reuse chunks and embeddings for
  unchanged files and discard data belonging to removed files.
- Rebuild global BM25 statistics from the final combined chunk set, because
  BM25 statistics depend on the whole corpus.
- Keep `vectors.npy` rows aligned with `all_chunks`. A BM25-only rebuild must
  not leave an old vector file that semantic or hybrid mode can mistake for
  vectors belonging to the new index.
- Persist enough metadata to verify that `index.pkl` and `vectors.npy` belong
  to the same corpus version, and save/update them consistently.

### 2. Run lint and fix reported issues

`flake8` and `mypy` are declared as development dependencies. Run the required
Makefile lint target after the remaining implementation work and resolve any
reported issues. The strict lint target is optional.
