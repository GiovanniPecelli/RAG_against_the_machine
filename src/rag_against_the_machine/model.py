from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A source file path and character span returned by retrieval."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question record without a reference answer or source list."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question record with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing answered and unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved source locations for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved source locations and a generated answer for one question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search output containing per-question results and requested top-k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Answer-generation output containing results and requested top-k."""

    search_results: List[MinimalAnswer]
    k: int
