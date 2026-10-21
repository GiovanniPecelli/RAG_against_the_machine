# Next Steps & Implementation Guide: RAG Against the Machine

Questo documento riassume lo stato attuale del progetto, l'unica funzionalità mancante richiesta dal subject della 42 e la guida passo-passo per completarla prima della consegna e della peer-evaluation.

---

## 1. Stato Attuale del Progetto (95% Completato)

| Componente | Stato | Dettagli |
| :--- | :---: | :--- |
| **Data Models** (`model.py`) |  Completato | Tutti gli 8 modelli Pydantic validati e conformi al subject. |
| **Chunking** (`chunker.py`) |  Completato | Strategie distinte Python e Markdown, limite $\le 2000$ caratteri, tracciamento offset assoluti. |
| **BM25 Engine** (`bm25.py`) |  Completato | BM25 custom con parametri di Robertson ($k_1=1.5, b=0.75$) senza librerie esterne. |
| **Indexer** (`indexer.py`) |  Completato | Scansione corpus, tokenizzazione, persistenza in `data/processed/index.pkl`. |
| **Retriever** (`retriever.py`) |  Completato | Singola query e batch search su dataset (`StudentSearchResults`). |
| **Generator** (`generator.py`) |  Completato | `Qwen/Qwen3-0.6B`, budget dinamico dei token, rimozione `<think>`, export `StudentSearchResultsAndAnswer`. |
| **CLI Dispatcher** (`src/__main__.py`) |  Completato | Sintassi ufficiale `uv run python -m src <command>` pienamente funzionante. |
| **Quality & Makefile** |  Completato | Tutte le regole esposte (`install`, `data`, `run`, `debug`, `clean`, `lint`, `lint-strict`). |
| **CLI `evaluate`** | ⏳ **Da Implementare** | Manca il comando di calcolo e report del Recall@k. |

---

## 2. Cosa Manca: Il Comando CLI `evaluate`

Il subject richiede (Capitolo VI.6 e Capitolo VII):
```bash
uv run python -m src evaluate \
  --student_search_results_path <path_ai_risultati_trovati> \
  --dataset_path <path_alle_risposte_corrette_AnsweredQuestions>
```

### Requisiti del Subject (Sezione VII.1.1):
1. **Overlap / IoU (Intersection over Union)**:
   Una fonte trovata (`retrieved_source`) è considerata corretta se si trova nello **stesso file** (`file_path` identico) e ha una sovrapposizione di caratteri con la fonte di riferimento (`ground_truth`) con **IoU $\ge 0.05$**:
   $$\text{IoU} = \frac{\text{Intersezione tra gli intervalli di caratteri}}{\text{Unione tra gli intervalli di caratteri}}$$
2. **Recall@k**:
   Per ogni domanda, calcola quante fonti corrette sono state recuperate tra le prime $k$ restituite:
   $$\text{Recall@k} = \frac{\text{Fonti corrette recuperate nei top-}k}{\text{Totale fonti corrette per la domanda}}$$
3. **Formato Output a Schermo** (Sezione VI.7.1, pagina 18):
   ```text
   Evaluation Results
   ========================================
   Recall@1: 0.xxx  Recall@3: 0.xxx  Recall@5: 0.xxx  Recall@10: 0.xxx
   ```
4. **Target Minimi di Superamento**:
   - **Documentazione**: Recall@5 $\ge 80\%$
   - **Codice**: Recall@5 $\ge 50\%$

---

## 3. Guida Passo-Passo all'Implementazione di `evaluate`

### Passo A: La Funzione di Calcolo IoU
La formula per calcolare l'IoU tra due intervalli di caratteri `[start1, end1]` e `[start2, end2]`:

```python
def compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculates Intersection over Union (IoU) between two character spans."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)

    union = (end1 - start1) + (end2 - start2) - intersection
    if union <= 0:
        return 0.0
    return intersection / union
```

### Passo B: Funzione di Calcolo Recall@k
Confronta le fonti predette con quelle di riferimento per una singola domanda:

```python
def is_source_recalled(pred_source: dict, gt_source: dict, min_iou: float = 0.05) -> bool:
    """Verifies same file_path and character overlap IoU >= 0.05."""
    if pred_source.get("file_path") != gt_source.get("file_path"):
        return False
    iou = compute_iou(
        pred_source.get("first_character_index", 0),
        pred_source.get("last_character_index", 0),
        gt_source.get("first_character_index", 0),
        gt_source.get("last_character_index", 0),
    )
    return iou >= min_iou
```

### Passo C: Metodo CLI in `src/rag_against_the_machine/__main__.py`
Aggiungere il metodo `evaluate` alla classe `RAGCLI`:

```python
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        """
        Evaluate retrieval Recall@k against ground-truth AnsweredQuestions dataset.
        """
        import json
        from pathlib import Path

        p_pred = Path(student_search_results_path)
        p_gt = Path(dataset_path)

        if not p_pred.exists():
            print(f"Error: Prediction file '{student_search_results_path}' does not exist.")
            return
        if not p_gt.exists():
            print(f"Error: Ground-truth file '{dataset_path}' does not exist.")
            return

        try:
            with open(p_pred, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            with open(p_gt, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception as e:
            print(f"Error reading evaluation datasets: {e}")
            return

        # Mappa le predizioni per question_id
        preds_by_id = {
            item["question_id"]: item.get("retrieved_sources", [])
            for item in pred_data.get("search_results", [])
        }

        # Mappa i ground-truth per question_id
        gt_questions = gt_data.get("rag_questions", [])
        if not gt_questions:
            print("Error: No questions found in ground-truth dataset.")
            return

        recalls: dict[int, list[float]] = {k: [] for k in k_values}

        for gt_item in gt_questions:
            q_id = gt_item.get("question_id")
            gt_sources = gt_item.get("sources", [])
            if not gt_sources:
                continue

            retrieved = preds_by_id.get(q_id, [])

            for k in k_values:
                top_k = retrieved[:k]
                found_count = 0
                for gt_src in gt_sources:
                    if any(is_source_recalled(pred, gt_src) for pred in top_k):
                        found_count += 1
                recalls[k].append(found_count / len(gt_sources))

        # Stampa i risultati nel formato richiesto dal subject
        print("\nEvaluation Results")
        print("=" * 40)
        formatted = "  ".join(
            f"Recall@{k}: {sum(recalls[k]) / len(recalls[k]):.3f}"
            for k in k_values
        )
        print(formatted + "\n")
```

---

## 4. Procedura di Verifica e Test Completa

Una volta implementato `evaluate`, esegui l'intero flusso dall'inizio alla fine:

### 1. Indicizzazione
```bash
uv run python -m src index --max_chunk_size 2000
```
*Verifica che l'indicizzazione completi in meno di 5 minuti e crei `data/processed/index.pkl`.*

### 2. Ricerca sul Dataset Documentazione
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 3. Valutazione Recall@k (Il tuo comando e la Moulinette)
```bash
# Con il tuo comando CLI:
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# Con la moulinette ufficiale del subject:
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```
*Target richiesto: Recall@5 $\ge 0.800$ (80%).*

### 4. Generazione Risposte
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
*Verifica che il file `dataset_docs_public.json` dentro `data/output/search_results_and_answer/UnansweredQuestions/` sia valido secondo `StudentSearchResultsAndAnswer`.*

---

## 5. Controllo Qualità e Linter (`make lint`)

Prima della consegna finale:
```bash
make lint
make lint-strict
```

Assicurati che:
- `flake8 .` restituisca zero errori (nessun import inutilizzato, lunghezza righe $\le 79$ caratteri).
- `mypy .` validi i tipi senza errori.

---

## 6. Checklist Finale Pre-Consegna

- [ ] `uv run python -m src <command>` risponde per tutti i 6 comandi (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`).
- [ ] Nessun file binario pesante, modello o output committato nel repository Git (`.gitignore` corretto).
- [ ] `README.md` completo di login studente, descrizione e formule matematiche.
- [ ] `make clean` rimuove cache e file temporanei senza eliminare la `.venv`.
