from .retriever import Retriever
from .indexer import Indexer

from pathlib import Path
from typing import Any
import fire
import json

from rag_against_the_machine import retriever


class RAGCLI:
    """RAG against the machine CLI."""

    @staticmethod
    def _compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
        """Calculate the Intersection over Union"""
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)
        union = (end1 - start1) + (end2 - start2) - intersection
        return (intersection / union) if union > 0 else 0.0

    @staticmethod
    def _is_source_recalled(
        pred: dict[str, Any], gt_src: dict[str, Any], min_iou: float = 0.05
    ) -> bool:
        # IF are not part of the same file ...
        if pred.get("file_path") != gt_src.get("file_path"):
            return False
        iou = RAGCLI._compute_iou(
            pred.get("first_character_index", 0),
            pred.get("last_character_index", 0),
            gt_src.get("first_character_index", 0),
            gt_src.get("last_character_index", 0),
        )
        return iou >= min_iou
    
    def index(
        self,
        max_chunk_size: int = 2000,
        processed_dir: str = "data/processed",
        raw_dir: str = "data/raw/vllm-0.10.1",
        semantic: bool = False,
    ) -> None:
        """ingest the file and build the BM25 index."""
        if max_chunk_size <= 0:
            print("Error: max_chunk_index must be greater than 0")
            return

        raw_path = Path(raw_dir)
        if not raw_path.exists():
            print(f"Error: '{raw_dir}' does not exist.")
            return

        print(
            f"Indexing started in '{raw_dir}' "
            f"(chunk max: {max_chunk_size})"
        )
        indexer = Indexer(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_chunk_size=max_chunk_size,
        )
        indexer.build(semantic)

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> None:
        """Search for i top-k snipped for a single question."""
        if not query or not query.strip():
            print("Error: The query can't be empty")
            return

        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        print(f"Serach for: '{query}' (top {k})...")

        # Load retriever and get the result ["search" is an alias]
        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        results = retriever.search(query, k=k)

        # Results is a list[MinimalSource]
        for i, source in enumerate(results, start=1):
            print(
                f"[{i}] {source.file_path} "
                f"(chars {source.first_character_index} "
                f"to {source.last_character_index})"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        ),
        processed_dir: str = "data/processed",
    ) -> None:
        """Call the retriever to search for the entire dataset"""

        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        retriever.search_dataset(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single query using the retrieved context."""
        if not query or not query.strip():
            print("Error: The query can't be empty.")
            return

        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        sources = retriever.retrieve(query, k)

        from .generator import Generator

        generator = Generator(model_name=model_name)
        answer_text = generator.generate_answer(query, sources)
        print(f"\n--- Answer ---\n{answer_text}\n")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """
        Generate answers for a dataset, producing a
        StudentSearchResultsAndAnswer JSON file.
        """
        from .generator import Generator

        generator = Generator(model_name=model_name)
        generator.generate_answer_dataset(
            student_search_results_path,
            save_directory,
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k_values: tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        """Evaluate retrieval Recall@k against ground-truth dataset."""

        # Prediction: program's predicted results
        p_pred = Path(student_search_results_path)

        # Ground Truth: reference data
        p_gt = Path(dataset_path)

        if not p_pred.exists():
            print(f"Error: '{student_search_results_path}' does not exist.")
            return
        if not p_gt.exists():
            print(f"Error: '{dataset_path}' does not exist.")
            return

        try:
            with open(p_pred, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            with open(p_gt, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception as e:
            print(f"Error reading datasets: {e}")
            return

        pred_quest_id: dict = {}
        for pred_item in pred_data.get("search_results", []):
            question_id = pred_item["question_id"]
            pred_sources = pred_item.get("retrieved_sources", [])
            pred_quest_id[question_id] = pred_sources

        # RECALLS save the points for each k
        #{                                                                                                           
        #    1:  [0.0, 1.0, 0.5, 0.0, ...] Recall@1 for each quests
        #    3:  [0.5, 1.0, 0.5, 1.0, ...] Recall@3 for each quests
        #    5:  [1.0, 1.0, 1.0, 1.0, ...] Recall@5 for each quests
        #    10: [1.0, 1.0, 1.0, 1.0, ...] Recall@10 for each quests
        #}
        recalls: dict[int, list[float]] = {k: [] for k in k_values}

        gt_questions = gt_data.get("rag_questions", [])
        for gt_item in gt_questions:
            gt_quest_id = gt_item.get("question_id")
            gt_sources = gt_item.get("sources", [])
            if not gt_sources:
                continue

            # Retrieve the sources found for the question: "gt_quest_id"
            retrieved = pred_quest_id.get(gt_quest_id, [])

            for k in k_values:
                top_k = retrieved[:k]
                
                # Count the correct sources found
                found = 0
                for gt_src in gt_sources:
                    for pred in top_k:
                        if self._is_source_recalled(pred, gt_src):
                            found += 1
                            break
                # ADD the score for the current k
                recalls[k].append(found / len(gt_sources))

                if k == 5 and found == 0:
                    print(f"\n[FAIL @ 5] ID: {gt_quest_id}")
                    print(f"  Domanda: {gt_item.get('question')}")
                    print(f"  Atteso: {gt_sources[0].get('file_path')}")
                    if top_k:
                        print(f"  Trovato: {top_k[0].get('file_path')}")

        print("\nEvaluation Results")
        print("=" * 40)
        formatted = "  ".join(
            f"Recall@{k}: {sum(recalls[k]) / len(recalls[k]):.3f}"
            for k in k_values
        )
        print(formatted + "\n")


if __name__ == "__main__":
    fire.Fire(RAGCLI())