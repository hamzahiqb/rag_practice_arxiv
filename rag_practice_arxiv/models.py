from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Frequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"


class InputType(StrEnum):
    KEYWORDS = "keywords"
    CATEGORY = "category"
    ARXIV_ID = "arxiv_id"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


@dataclass
class SearchRequest:
    query: str
    input_type: InputType
    frequency: Frequency = Frequency.WEEKLY
    max_results: int = 100


@dataclass
class RankedPaper:
    paper: Paper
    score: float
    reasoning: str


@dataclass
class AgentState:
    user_input: str = ""
    frequency: Frequency = Frequency.WEEKLY
    search_request: SearchRequest | None = None
    papers: list[Paper] = field(default_factory=list)
    ranked_papers: list[RankedPaper] = field(default_factory=list)
    status: str = ""
