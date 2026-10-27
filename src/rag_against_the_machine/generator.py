from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
)
from pathlib import Path
from tqdm import tqdm
import torch
import json
import os

from .model import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Loads tokenizer and Qwen/Qwen3-0.6B model onto available device.

        .to(self.device):
            The generator uses the gpu with cuda if available
            otherwise the cpu
        """
        self._file_cache: dict[str, str] = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.auto_tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(model_name)
        )

        # bfloat16 have the scope to reduce the data heavy
        # modern LLMs (including Qwen3) were trained natively in bfloat16
        self.auto_model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
            ).to(self.device)
        )

        # Force PyTorch to use all available CPU cores
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)

    def extract_snippets(
            self, sources: list[MinimalSource]
    ) -> list[tuple[MinimalSource, str]]:
        """Extracts context from sources"""
        extracted: list[tuple[MinimalSource, str]] = []
        for src in sources:
            file_path = src.file_path

            # if is not in the cache, read it by the disk
            if file_path not in self._file_cache:
                p = Path(file_path)
                if not p.exists():
                    continue
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        content = f.read()
                        self._file_cache[file_path] = content
                except Exception:
                    continue

            content = self._file_cache[file_path]
            snippet = content[
                src.first_character_index:src.last_character_index
            ]

            # strip() avoid extraction of empty snippet
            if snippet.strip():
                extracted.append((src, snippet))
        return extracted

    def format_prompt(
            self,
            query: str,
            snippets: list[tuple[MinimalSource, str]]
    ) -> str:
        """
        Qwen's tokenizer will automatically turn the messages list into
        its special internal tokens

        apply_chat_template() takes:
            messages: the prompt "system" and "user"
            tokenize: if TRUE return the token(numbers) -> here False
            add_generation_prompt add <|im_start|>assistant
            that say to the model that the prompt is finished
        """
        context_block = []
        for i, (src, text) in enumerate(snippets, start=1):
            context_block.append(
                f"[Source {i}: {src.file_path}]\n{text.strip()}"
            )

        context_str = "\n\n".join(context_block)

        system_prompt = (
            "You are a technical assistant specializing in the vLLM "
            "codebase.\n"
            "Answer the user question based ONLY on the provided context "
            "snippets.\n"
            "If the context does not contain enough information, "
            "state that clearly.\n"
            "Answer directly in 1 to 2 sentences without filler."
            "Be factual, concise, and do not hallucinate."
        )

        user_prompt = (
            f"Context snippets: \n{context_str}\n\nQuestion: {query}"
        )

        # specifications in the docstring
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # specifications in the docstring
        prompt = self.auto_tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True
        )

        # "<think>\n</think>\n":
        # say to the model that thought are already ended
        return prompt + "<think>\n</think>\n"

    def generate_answer(
            self,
            query: str,
            sources: list[MinimalSource],
            max_new_tokens: int | None = None,
    ) -> str:
        """
        Generates an answer based on retrieved sources and the user query.

        max_new_tokens have a flaxible determination with a min value=384
            This choice prevents responses from entering a loop and
            avoids the truncation of responses.
        return_tensors="pt" & .to(self.device):
            The tokenizer converts text into PyTorch tensors ("pt").
            .to(self.device) moves the tensors to GPU/CPU to match the model.

        self.auto_model.generate():
            with torch.no_grad():
                Disables gradient calculation during inference to halve memory
                usage and speed up generation.
            **inputs:
                Unpacks the dictionary of tensors (input_ids, attention_mask)
                as keyword arguments to model.generate().
                    data structure example:
                    inputs = {
                        "input_ids": tensor([[151644, 8948, 279, 102]]),
                        "attention_mask": tensor([[1, 1, 1, 1]])
                    }
            max_new_tokens:
                Limits the maximum number of newly generated tokens in the
                reply.
            do_sample=False:
                Uses greedy decoding for deterministic, factual answers.
        """
        if max_new_tokens is None:
            max_new_tokens = 180

        # Send to model 3 sources to reduce the waiting time
        snippets = self.extract_snippets(sources[:3])
        if not snippets:
            return "No relevant context found to answer the question."

        prompt = self.format_prompt(query, snippets)
        inputs = self.auto_tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            # generate() is a loop that every time pass:
            # the PROMPT and what is already generated to the llm
            # to generate the NEXT token
            # do_sample: Always picks the word with the highest probability
            #    IF True: the llm is in creative mode.
            #    it could use random and temperature properties
            outputs = self.auto_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # outputs[0] = [ all PROMPT + generated ANSWER ]
        # shape[1]: return the matrix dimension, [0] rows - [1] columns
        input_len = inputs["input_ids"].shape[1]

        # The generated tokens are after the input prompt
        generated_tokens = outputs[0][input_len:]

        # skip_special_tokens: skip all technical tag (es: <|im_end|>)
        answer = self.auto_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if "</think>" in answer:
            answer = answer.split("</think>")[-1]

        elif "<think>" in answer:
            return (
                "Error: Token budget exhausted during reasoning. "
                "Please increase max_new_tokens."
            )

        answer = answer.strip()
        if not answer:
            return "Unable to produce an answer from the retrieved sources."

        return answer

    def generate_answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
    ) -> None:
        """
        Reads search results JSON, generates answers, and saves
        StudentSearchResultsAndAnswer.
        """
        dataset_path: Path = Path(student_search_results_path)
        if not dataset_path.exists():
            print(f"Error: File not found at '{student_search_results_path}'")
            return

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Malformed JSON in '{student_search_results_path}'")
            return
        except Exception as e:
            print(f"Error reading '{student_search_results_path}': {e}")
            return

        answers: list[MinimalAnswer] = []

        for q_item in tqdm(data.get("search_results", [])):
            question_id = q_item["question_id"]
            question = q_item["question"]

            sources: list[MinimalSource] = []
            for source in q_item.get("retrieved_sources", []):
                sources.append(
                    MinimalSource(
                        file_path=source["file_path"],
                        first_character_index=source["first_character_index"],
                        last_character_index=source["last_character_index"],
                    )
                )

            answer = self.generate_answer(query=question, sources=sources)

            answers.append(
                MinimalAnswer(
                    question_id=question_id,
                    question=question,
                    retrieved_sources=sources,
                    answer=answer,
                )
            )

        # ====================================================================
        #                   Saving Output in save_directory
        # ====================================================================
        k = data.get("k", 5)
        output_data = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=k,
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / dataset_path.name

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {output_file}")