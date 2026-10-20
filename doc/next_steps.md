# Guida Dettagliata all'Implementazione: RAG Against the Machine

Questo documento contiene tutti i dettagli architetturali, le spiegazioni tecniche e i passi concreti per completare la parte di **Generazione LLM** (`answer`, `answer_dataset`) e di **Valutazione** (`evaluate`).

---

## 1. Stato Attuale (Verificato al 100%)

I seguenti moduli sono completati, testati e a norma Flake8/Mypy:
- **`model.py`**: Tutti gli 8 modelli Pydantic ufficiali.
- **`chunker.py`**: Chunking Markdown/Python con protezione righe > 2000 caratteri e offset corretti.
- **`tokenizer.py`**: Gestione CamelCase e snake_case per massimizzare il Recall.
- **`bm25.py`**: Algoritmo Robertson BM25 da zero ($k_1=1.5$, $b=0.75$, smoothed IDF, no division by zero).
- **`indexer.py`**: Scansione ordinata deterministica, chunking e salvataggio `data/processed/index.pkl`.
- **`retriever.py`**: Ranking top-$k$, gestione di $k \le 0$, file mancanti e JSON corrotti senza traceback.
- **`__main__.py`**: Comandi `index`, `search` e `search_dataset` funzionanti e blindati.

---

## 2. FASE 1: Dipendenze per l'LLM

Nel file `pyproject.toml`, assicurarsi che la sezione `dependencies` contenga:

```toml
dependencies = [
    "fire>=0.7.1",
    "pydantic>=2.14.1",
    "tqdm>=4.70.1",
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "accelerate>=0.28.0",
]
```

E sincronizzare l'ambiente da terminale:
```bash
uv sync
```

---

## 3. FASE 2: Modulo `generator.py` (Architettura e Dettagli Chiave)

Il file `src/rag_against_the_machine/generator.py` conterrà la classe `Generator`.

### Dettaglio A: La Cache dei File in Memoria (`_file_cache`)
- **Il problema:** `MinimalSource` contiene solo la provenienza (`file_path`, `first_character_index`, `last_character_index`), non il testo grezzo. Quando recuperiamo i $k=5$ chunk migliori, spesso 3 o 4 provengono dallo **stesso file** (es. `lora.md`), solo a paragrafi differenti. Senza cache, rileggeremmo lo stesso file dal disco 3 o 4 volte. In `answer_dataset` (200 domande), faremmo oltre 1.000 accessi al disco.
- **La soluzione:** Creare un dizionario `self._file_cache: dict[str, str] = {}`.
  - Se il file non è nel dizionario, lo leggiamo da disco una sola volta e lo salviamo (`_file_cache[path] = content`).
  - Se il file è già nel dizionario, leggiamo istantaneamente dalla RAM (in nanosecondi).

```python
def extract_snippets(
    self, sources: list[MinimalSource]
) -> list[tuple[MinimalSource, str]]:
    extracted: list[tuple[MinimalSource, str]] = []
    for src in sources:
        file_path = src.file_path
        # Se non è in cache, leggilo dal disco una sola volta
        if file_path not in self._file_cache:
            p = Path(file_path)
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    self._file_cache[file_path] = f.read()
            except Exception:
                continue

        content = self._file_cache[file_path]
        # Ritaglio della fetta esatta di testo
        snippet = content[src.first_character_index : src.last_character_index]
        if snippet.strip():
            extracted.append((src, snippet))
    return extracted
```

---

### Dettaglio B: Costruzione del Prompt con ChatML nativo
- **Perché `tokenizer.apply_chat_template`:** Il modello `Qwen/Qwen3-0.6B` è stato addestrato usando il formato ChatML (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n`). Non dobbiamo formattare i tag a mano: usare `apply_chat_template` garantisce che il modello riceva i token di controllo esatti.
- **Vincoli del Subject per l'Answer:**
  1. *Coerente e comprensibile.*
  2. *Grounded:* basata **esclusivamente** sul contesto fornito, vietando allucinazioni.
  3. *On point:* risponde esattamente alla domanda.

```python
def format_prompt(
    self, query: str, snippets: list[tuple[MinimalSource, str]]
) -> str:
    context_blocks = []
    for i, (src, text) in enumerate(snippets, start=1):
        context_blocks.append(f"[Source {i}: {src.file_path}]\n{text.strip()}")
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a technical assistant specializing in the vLLM codebase.\n"
        "Answer the user question based ONLY on the provided context snippets.\n"
        "If the context does not contain enough information, state that clearly.\n"
        "Be factual, concise, and do not hallucinate."
    )

    user_prompt = f"Context snippets:\n{context_str}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
```

---

### Dettaglio C: Inferenza con Greedy Decoding (`do_sample=False`)
- **Perché `do_sample=False`:**
  - Nel RAG tecnico non vogliamo creatività o fantasia; vogliamo precisione e risposte deterministiche (la stessa domanda deve dare sempre la stessa risposta).
  - Su CPU, il campionamento casuale (`temperature > 0`, `top_p`) è più lento e rischia allucinazioni. Greedy decoding sceglie sempre il token più probabile: è più rapido e rigoroso.
- **Perché `max_new_tokens=256`:**
  - Una risposta tecnica a una domanda di codice richiede raramente più di 150-250 token. Mettere un limite impedisce che il modello su CPU si blocchi in loop infiniti.
- **Come estrarre solo la risposta generata:**
  - `model.generate` restituisce sia i token del prompt sia quelli nuovi.
  - Per ottenere solo la risposta, calcoliamo la lunghezza dell'input (`input_len = inputs["input_ids"].shape[1]`) e prendiamo lo slice: `outputs[0][input_len:]`.

```python
def generate_answer(
    self,
    query: str,
    sources: list[MinimalSource],
    max_new_tokens: int = 256,
) -> str:
    snippets = self.extract_snippets(sources)
    if not snippets:
        return "No relevant context found to answer the question."

    prompt = self.format_prompt(query, snippets)
    inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()
```

---

## 4. FASE 3: I Comandi CLI in `__main__.py`

### 1. `answer <query> --k <int>`
- Flusso:
  1. Controlla `query` e `k`.
  2. `retriever = Retriever.load(processed_dir)`.
  3. `sources = retriever.retrieve(query, k=k)`.
  4. `generator = Generator(model_name=model_name)`.
  5. `answer = generator.generate_answer(query, sources)`.
  6. Stampa a terminale la risposta e l'elenco delle fonti usate.

### 2. `answer_dataset --student_search_results_path <path> --save_directory <dir>`
- Flusso:
  1. Apre il file JSON prodotto in precedenza da `search_dataset` (`StudentSearchResults`).
  2. Inizializza `generator = Generator()`.
  3. Per ciascun elemento `item` in `search_results`:
     - Prende `item.question` e `item.retrieved_sources`.
     - Chiama `generator.generate_answer(item.question, item.retrieved_sources)`.
     - Crea un oggetto `MinimalAnswer` contenente `question_id`, `question`, `retrieved_sources` e `answer`.
  4. Avvolge la lista nel modello `StudentSearchResultsAndAnswer(search_results=answers, k=k)`.
  5. Salva il file JSON formattato dentro `--save_directory`.

---

## 5. FASE 4: Comando di Valutazione `evaluate`

Il subject richiede di calcolare il nostro **Recall@k** rispetto ai dataset con le risposte corrette (`AnsweredQuestions`).

### La Formula del Recall@k con IoU (Intersection over Union)
Nel Capitolo VII.1.1:
- Una sorgente recuperata è corretta se si trova nello **stesso file** (`file_path` identico) e l'intervallo di caratteri ha un **$\text{IoU} \ge 0.05$**.
- Formula IoU tra due intervalli $[A_1, A_2]$ e $[B_1, B_2]$:
  $$\text{Intersezione} = \max(0, \min(A_2, B_2) - \max(A_1, B_1))$$
  $$\text{Unione} = \max(A_2, B_2) - \min(A_1, B_1)$$
  $$\text{IoU} = \frac{\text{Intersezione}}{\text{Unione}}$$

- **Soglie minime obbligatorie da superare:**
  - **$\ge 80\%$ Recall@5** sui file di documentazione (`dataset_docs_public.json`).
  - **$\ge 50\%$ Recall@5** sui file di codice (`dataset_code_public.json`).

---

## 6. FASE 5: Controlli Finali per la Moulinette

1. **Entrypoint `python -m src`**:
   Creare `src/__main__.py`:
   ```python
   import fire
   from rag_against_the_machine.__main__ import RAGCLI

   if __name__ == "__main__":
       fire.Fire(RAGCLI)
   ```
   Così `uv run python -m src <comando>` funzionerà come richiesto dalla valutazione.

2. **File `.flake8` alla radice**:
   ```ini
   [flake8]
   max-line-length = 79
   exclude = .venv,data,tests/data,.git,__pycache__
   ```
   Così `flake8 .` controllerà solo il codice sorgente scritto da te senza scansionare le migliaia di file di vLLM.

3. **Verifica Finale:**
   - `make lint`
   - `make lint-strict`
