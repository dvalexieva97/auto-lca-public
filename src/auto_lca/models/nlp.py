from typing import List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DocumentRank:
    """A ranked document with its score and rank."""

    query: str
    text: str
    rank: int
    score: float
    pid: Optional[str] = None
    request_id: Optional[str] = None
    inserted_at: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Topic:
    """A topic extracted from a paper."""

    topic_id: int
    representation: List[str]
    count: int
    paper_ids: List[str] = field(default_factory=list)
    ranks: Optional[List[DocumentRank]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
