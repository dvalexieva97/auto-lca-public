import re
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
from auto_lca.shared.util import hash_sha256
from auto_lca.models.enum import AcademicSearchEngine
from auto_lca.models.nlp import Topic


@dataclass
class PaperIds:
    """A class to hold various identifiers for a paper."""

    doi: Optional[str] = None
    semantic_scholar: Optional[str] = None
    mag: Optional[str] = None
    openalex: Optional[str] = None

    acl: Optional[str] = None
    arxiv: Optional[str] = None
    corpus_id: Optional[str] = None
    dblp: Optional[str] = None
    pubmed: Optional[str] = None
    pubmed_central: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def normalize_doi(doi: str) -> str:
        """Normalizes the DOI by removing 'https://doi.org/' or 'DOI:' prefix (case-insensitive, only at start)."""
        doi = doi.strip().lower()
        return re.sub(
            r"^(https://doi\.org/|doi:)", "", doi, flags=re.IGNORECASE
        ).strip()


# TODO Add mapping class which maps keys to Paper attributes
# using an enum mapping per paper source (e.g. Semantic Scholar, OpenAlex, etc.)


@dataclass
class Paper:
    title: str
    # Unique ID for the paper, built from DOI:
    pid: Optional[str] = None
    search_engine: Optional[AcademicSearchEngine] = None
    journal: Optional[str] = None
    search_term: Optional[str] = None
    publication_year: Optional[str] = None
    version: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata_url: Optional[str] = None
    pdf_url: Optional[str] = None

    # ids
    ids: Optional[PaperIds] = field(default_factory=PaperIds)

    request_id: Optional[str] = None
    scraped_at: Optional[str] = None
    inserted_at: Optional[str] = None

    authors: Optional[List[str]] = field(default_factory=list)
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    fields_of_study: Optional[List[str]] = field(default_factory=list)
    keywords: Optional[List[str]] = field(default_factory=list)
    reference_count: Optional[int] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None
    blob_url: Optional[str] = None
    error: Optional[str] = None
    pdf_size_mb: Optional[float] = None
    exec_time: Optional[float] = None

    # NLP:
    full_text: Optional[str] = None
    summary: Optional[str] = None
    topics: Optional[List[Topic]] = None

    @classmethod
    def generate_request_id(
        cls, timestamp, academic_search_engine: AcademicSearchEngine, search_term=None
    ):
        """Generates a unique request ID for the
        paper based on the current time and a prefix.
        """
        if not search_term:
            search_term = ""
        return hash_sha256(f"{academic_search_engine}{search_term}{timestamp}")

    def __post_init__(self):
        self.pid = self.build_id_from_doi()

    def to_dict(self):
        dic = asdict(self)
        if dic.get("search_engine"):
            dic["search_engine"] = dic["search_engine"].value
        # Ensure repeated fields are always lists
        if dic.get("fields_of_study") is None:
            dic["fields_of_study"] = []
        return dic

    def dump(self, file_path: str):
        """
        Dumps the paper instance as a row in a JSONL file.
        """

        paper_data = asdict(self)

        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(paper_data, f, ensure_ascii=False)
            f.write("\n")

        print(f"Paper appended to {file_path}")

    def __eq__(self, other):
        return self.pid == other.pid

    def __hash__(self):
        return hash(self.pid)

    @classmethod
    def deduplicate(cls, papers):
        """Deduplicates papers by ID"""
        return list(set(papers))

    @classmethod
    def doi_to_id(cls, doi: str):
        """Converts a DOI to a unique paper ID."""
        return hash_sha256(f"{PaperIds.normalize_doi(doi)}")

    def build_id_from_doi(self):
        """Builds a unique ID for the paper based on its DOI."""
        if not self.ids.doi:
            # TODO Log warning
            # raise ValueError(
            #     f"DOI is required to build the paper ID. Current paper: {asdict(self)}"
            # )
            if not self.ids.semantic_scholar:
                raise ValueError(
                    f"Paper ID cannot be built without DOI or Semantic Scholar ID. Current paper: {asdict(self)}"
                )
            self.ids.doi = f"NULL_DOI_{self.ids.semantic_scholar}"
        return self.doi_to_id(self.ids.doi)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Paper instance from a dict, handling nested structures like ids and topics.
        Ignores unknown fields.
        """
        # Handle nested ids
        ids_data = data.get("ids")
        if isinstance(ids_data, dict):
            data["ids"] = PaperIds(**ids_data)
        elif ids_data is None:
            data["ids"] = PaperIds()

        # Handle topics if present
        if "topics" in data and isinstance(data["topics"], list):
            data["topics"] = [
                Topic(**t) if isinstance(t, dict) else t for t in data["topics"]
            ]

        # Only pass known fields to the constructor
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


@dataclass
class PaperBatch:
    """A batch of papers to be processed together."""

    request_id: Optional[str] = None
    inserted_at: Optional[str] = None
    topics: List[Topic] = field(default_factory=list)

    def dump(self, file_path: str):
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
            f.write("\n")
        print(f"Paper batch appended to {file_path}")
