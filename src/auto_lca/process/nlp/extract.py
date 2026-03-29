import re

import pymupdf
import tiktoken

from auto_lca.process.nlp.config import NLPConfig
from auto_lca.process.nlp.document_rank import DocumentRanker
from auto_lca.shared.constants import (
    PAPER_SECTIONS,  # TODO These need to be aligned with concept config sections
)
from auto_lca.shared.file_manager import FileManager

# Get encoding once:
encoding_name = "o200k_base"
encoding = tiktoken.get_encoding(encoding_name)
MIN_SIMILARITY_THRESHOLD = 0.5


class TextExtractor(FileManager, DocumentRanker):
    # Class to extract text and tables from pdf

    # TODO Add to config and fix path:
    def __init__(self, config: NLPConfig = None) -> None:
        super().__init__()
        if not config:
            config = NLPConfig()
        self.config = config
        self.section_names = PAPER_SECTIONS
        self.ranker = DocumentRanker(self.config)

    @classmethod
    def extract_text_from_pdf(cls, pdf_path):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        try:
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        if page_text:
                            text += page_text
                    except Exception as e:
                        print(
                            f"Warning: Error extracting text from page {page_num} of {pdf_path}: {e}"
                        )
                        continue
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract text from PDF {pdf_path}: {e}"
            ) from e
        return text

    @classmethod
    def extract_tables_from_pdf(cls, pdf_path):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """

        tables = []
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                tabs = page.find_tables()
                if tabs.tables:
                    extracted = [tab.to_pandas() for tab in tabs.tables]
                    print(extracted)
                    tables += extracted
        return tables

    def cutoff_string_by_max_tokens(
        self, text, encoding=encoding, max_allowed_tokens=None
    ) -> str:
        """Returns the number of tokens in a text string."""
        if not max_allowed_tokens:
            max_allowed_tokens = self.config.MAX_INPUT_TOKEN_COUNT
        encoded = encoding.encode(text)
        if len(encoded) > max_allowed_tokens:
            text = encoding.decode(encoded[:max_allowed_tokens])
        return text

    @classmethod
    def is_main_header(cls, line: str) -> bool:

        # max_header_letters = 80 # TODO Clean
        # max_header_words = 5
        line = line.strip()

        # skip empty lines or lines that are too short
        if len(line) < 3:
            return False

        # Case 1: Numbered headers ("1. Introduction", "2) Methods", etc.)
        match = re.match(r"^(\d+)[\.\)]?\s+[A-Z][A-Za-z]", line)
        if match:
            section_num = int(match.group(1))
            if 1 <= section_num <= 10:
                return True
            else:
                return False

        # Case 2: Uppercase headers ("INTRODUCTION", "METHODS AND MATERIALS")
        if re.match(r"^[A-Z][A-Z\s\-]{3,}$", line):
            # avoid cases like "H I G H L I G H T S" (spaced-out letters)
            if not re.match(r"^([A-Z]\s+){2,}[A-Z]$", line):
                return True

        # Case 3: Title-case single-word headers ("Introduction", "Acknowledgments", etc.)
        if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$", line):
            # but only if short and not a sentence
            if len(line.split()) <= 4 and not line.endswith("."):
                return True

        pattern = r"(?i)\b([A-Za-z]+)\s*(?:and|&)\s*([A-Za-z]+)\b"
        if re.match(pattern, line):
            return True

        return False

    @classmethod
    def filter_potential_headers(cls, text):
        """Detects potential paper headers in text"""

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        headers = [line for line in lines if cls.is_main_header(line)]

        return headers

    def extract_headers_from_text(self, text, section_names: list[str] = None):
        """
        Finds the main headers in a scientific text
        by finding the closest match to a list of desired headers (section_names)
        Returns: a Dict like:
        {'Abstract': [{'query': 'Abstract', 'rank': 1,
        'score': 1.2, 'text': 'ABSTRACT', 'cosine_similarity': 1.00}]}
        """
        if not section_names:
            section_names = self.section_names
        potential_headers = self.filter_potential_headers(text)

        results = self.ranker.rank_documents_bulk_queries(
            queries=section_names,
            documents=potential_headers,
            count_top_ranks=1,
        )
        for sname, result_list in results.items():
            res = result_list[0]
            clean_q = self.clean_title(res["query"])
            clean_t = self.clean_title(res["text"])
            res["cosine_similarity"] = self.compute_cosine_similarity(clean_q, clean_t)
            if res["cosine_similarity"] < MIN_SIMILARITY_THRESHOLD:
                res["fuzzy_cosine_similarity"] = self.compute_fuzzy_cosine_similarity(
                    clean_q, clean_t
                )
            else:
                res["fuzzy_cosine_similarity"] = None
        return results

    def extract_section(self, section_title: str, section_titles: list[str], text: str):
        """
        Extracts the content of a given section from a text based on its title
        and the next section title in the document.

        Args:
            section_title (str): The section title to extract.
            section_titles (List[str]): All detected section titles in order of appearance.
            text (str): The full document text.

        Returns:
            str: The extracted text belonging to that section (empty string if not found).
        """

        section_titles = list(dict.fromkeys(section_titles))

        # Escape regex special characters in titles (e.g. parentheses, dots)
        def escape(t):
            return re.escape(t.strip())

        # Find the current and next section patterns
        try:
            idx = section_titles.index(section_title)
        except ValueError:
            print(f"ValueError:section title not found: {section_title}")
            return None

        current_pattern = escape(section_title)
        next_pattern = None
        if idx + 1 < len(section_titles):
            next_pattern = escape(section_titles[idx + 1])

        # Construct regex for capturing everything between current and next section
        if next_pattern:
            pattern = rf"{current_pattern}(.*?)(?={next_pattern})"
        else:
            # Last section: capture until the end of text
            pattern = rf"{current_pattern}(.*)$"

        # Flags: DOTALL to include newlines, IGNORECASE for robustness
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return None

        section_text = match.group(1).strip()
        return section_text

    @classmethod
    def clean_title(cls, title: str):
        """Cleans title from numbers"""
        title = re.sub(r"^\d+[\.\)]?\s*", "", title)
        return title.strip()

    def extract_sections_from_text(self, text):
        """
        Extracts the main headers and
        their sections from a text
        Returns: a dict like {'section_name':
        {'section_text':str,
        'section_title': str,
        'cosine_similarity': float}}
        In case the cosine_similarity of the closest matched header
        text is too low, no text is extracted
        """
        results_dict = {}
        headers_dict = self.extract_headers_from_text(text)
        section_titles = [
            res[0]["text"]
            for res in headers_dict.values()
            if (
                res[0]["cosine_similarity"] >= MIN_SIMILARITY_THRESHOLD
                or (
                    res[0]["fuzzy_cosine_similarity"]
                    and res[0]["fuzzy_cosine_similarity"] >= MIN_SIMILARITY_THRESHOLD
                )
            )
        ]

        for k, res_list in headers_dict.items():
            dic = res_list[0]  # Get first (and only) result
            section_title = dic["text"]
            cosine = dic["cosine_similarity"]
            fuzzy_cosine = dic["fuzzy_cosine_similarity"]
            if not fuzzy_cosine:
                fuzzy_cosine = dic["cosine_similarity"]
            if cosine >= MIN_SIMILARITY_THRESHOLD or (
                fuzzy_cosine and fuzzy_cosine >= MIN_SIMILARITY_THRESHOLD
            ):
                section_text = self.extract_section(section_title, section_titles, text)
            else:
                section_text = None
            results_dict[k] = {
                "section_title": section_title,
                "cosine_similarity": cosine,
                "fuzzy_cosine_similarity": fuzzy_cosine,
                "section_text": section_text,
            }

        # Extract post-references supplementary content (tables, figures, appendices)
        post_refs_content = self._extract_post_references_content(text)
        if post_refs_content:
            print(
                f"Found {len(post_refs_content)} characters of post-references supplementary content"
            )
            # Append to Methods and Results sections if they exist
            for section_name in ["Methods", "Results"]:
                if (
                    section_name in results_dict
                    and results_dict[section_name]["section_text"]
                ):
                    original_text = results_dict[section_name]["section_text"]
                    results_dict[section_name]["section_text"] = (
                        original_text + "\n\n" + post_refs_content
                    )
                    print(f"Appended post-references content to {section_name} section")

        return results_dict

    def _extract_post_references_content(self, text):
        """
        SIMPLE logic: Extract anything that appears AFTER the References section ends.

        1. Find "References" header
        2. Find where references citations actually end
        3. Return everything after that (if there's anything substantial)

        Returns empty string if nothing substantial exists after references.
        """
        # Find the references section header
        refs_match = re.search(r"\b(references|bibliography)\b", text, re.IGNORECASE)
        if not refs_match:
            return ""

        refs_start = refs_match.start()
        after_refs = text[refs_start:]

        # Find where references end by looking for the LAST reference citation
        # References typically end with the last author citation (ends with year and period/newlines)
        # Look for the last occurrence of a typical reference pattern

        # Common reference patterns:
        # - "Author, A., Year. Title..."
        # - "Author, A. (Year)..."
        # - URLs/DOIs at the end

        # Find all lines that look like references (have author names, years, etc.)
        lines = after_refs.split("\n")
        last_ref_line_idx = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Skip empty lines and the References header itself
            if not line_stripped or i == 0:
                continue

            # Check if this looks like a reference citation
            # References typically have: year in parens, DOI/URL, or specific punctuation patterns
            has_year = bool(re.search(r"\b(19|20)\d{2}\b", line_stripped))
            has_doi_url = bool(
                re.search(r"(doi|http|www\.)", line_stripped, re.IGNORECASE)
            )
            has_authors = bool(re.search(r"[A-Z][a-z]+,\s*[A-Z]\.", line_stripped))

            # If line has reference characteristics, mark it as a potential last reference line
            if has_year or has_doi_url or has_authors:
                last_ref_line_idx = i

        # Everything after the last reference line is post-references content
        if last_ref_line_idx == 0:
            return ""  # Couldn't find end of references

        # Get content after last reference
        post_refs_lines = lines[last_ref_line_idx + 1 :]
        post_refs_content = "\n".join(post_refs_lines).strip()

        # Must have substantial content (at least 500 chars)
        if len(post_refs_content) < 500:
            return ""

        # Clean up
        post_refs_content = re.sub(
            r"MANUSCRIPT\s+ACCEPTED\s+ACCEPTED\s+MANUSCRIPT",
            "",
            post_refs_content,
            flags=re.IGNORECASE,
        )

        return post_refs_content.strip()
