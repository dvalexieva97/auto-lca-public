import re
from typing import Callable, Optional

import pandas as pd

from auto_lca.process.nlp.document_rank import DocumentRanker


class DataSimilarity:
    def __init__(self):
        self.ranker = DocumentRanker()
        # Pre-initialize the model to avoid meta tensor issues during parallel execution
        # This ensures the model is fully loaded before any threads try to use it
        # Force actual encoding to ensure weights are loaded (not just meta tensors)
        try:
            encoder = self.ranker._get_bi_encoder()
            # Do a dummy encode to force full model loading
            encoder.encode(["dummy"], convert_to_tensor=False)
        except Exception:
            # If initialization fails, let it fail later during actual use
            # This prevents blocking if there's a real issue
            pass
        # small epsilon for numeric comparisons
        self._numeric_tol = 1e-9

    @classmethod
    def _is_null(cls, val):
        if type(val) in [list, dict]:
            if not val:
                return True
            else:
                return False
        else:
            return True if pd.isna(val) or str(val).strip() == "" else False

    @classmethod
    def _safe_str(cls, val):
        # convert NaN/None to empty string, otherwise cast to str
        return "" if cls._is_null(val) else str(val)

    @staticmethod
    def _parse_value_and_unit(val: object) -> tuple[float | None, str | None]:
        """
        Try to split a raw value into a numeric part and a unit suffix.
        Returns (numeric_value, unit) where either can be None if parsing fails.
        """
        if DataSimilarity._is_null(val):
            return None, None

        if isinstance(val, (int, float)):
            return float(val), None

        text = str(val).strip()
        # Simple pattern: number followed by optional unit text
        match = re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([^\d\s].*)?$", text)
        if match:
            number = match.group(1)
            unit = match.group(2).strip() if match.group(2) else None
            try:
                return float(number), unit
            except Exception:
                return None, unit

        return None, None

    def text_similarity(self, val1, val2, fuzzy_flag=False):
        """Calculates similarity ratio between two texts"""

        similarity = None
        if fuzzy_flag:
            similarity = self.ranker.compute_fuzzy_cosine_similarity(
                self._safe_str(val1), self._safe_str(val2)
            )

        if not fuzzy_flag or similarity is None:
            similarity = self.ranker.compute_cosine_similarity(
                self._safe_str(val1), self._safe_str(val2)
            )
        return similarity

    def _numeric_similarity(self, val1, val2):
        """Return 1.0 if numeric values match within tolerance, else 0.0."""
        try:
            f1 = float(val1)
            f2 = float(val2)
            return 1.0 if abs(f1 - f2) <= self._numeric_tol else 0.0
        except Exception:
            return None

    def compare(
        self,
        val1,
        val2,
        llm_judge: Optional[Callable[[str, str], float | bool]] = None,
    ) -> dict:
        """
        Compute a set of similarity metrics for two values.
        Returns a dict with keys:
        - cosine / cosine_distance
        - fuzzy_cosine / fuzzy_cosine_distance (if enabled)
        - clean_string_equal (1.0 if lowercase stripped values are equal, else 0.0)
        - numeric_exact (if both parse as numeric)
        - numeric_distance (absolute diff when both parse as numeric)
        - llm_meaning_equal (optional LLM judge for semantic meaning equality; returns float in [0,1])
        """
        metrics: dict[str, float | None] = {}

        # Fast path: if both are null, return early
        if self._is_null(val1) and self._is_null(val2):
            return {
                "cosine": 1.0,
                "fuzzy_cosine": 1.0,
                "clean_string_equal": 1.0,
                "numeric_exact": 1.0,
                "numeric_distance": None,
                "llm_meaning_equal": 1.0 if llm_judge else None,
            }

        # Fast path: exact string match (case-insensitive)
        str1 = self._safe_str(val1)
        str2 = self._safe_str(val2)
        if str1.lower() == str2.lower() and str1:
            cosine_sim = 1.0
        else:
            # Only compute cosine similarity if values differ
            cosine_sim = self.ranker.compute_cosine_similarity(str1, str2)

        metrics["cosine"] = cosine_sim

        # Always calculate fuzzy cosine similarity
        if cosine_sim == 1.0 or str1.lower() == str2.lower():
            fuzzy_sim = 1.0
        else:
            fuzzy_sim = self.ranker.compute_fuzzy_cosine_similarity(str1, str2)
            # If fuzzy returns None (both words length 1), use normal cosine
            if fuzzy_sim is None:
                fuzzy_sim = cosine_sim

        metrics["fuzzy_cosine"] = fuzzy_sim

        # Clean string equality check (lowercase, stripped)
        clean_str1 = str1.lower().strip()
        clean_str2 = str2.lower().strip()
        metrics["clean_string_equal"] = (
            1.0 if clean_str1 == clean_str2 and clean_str1 else 0.0
        )

        # Numeric handling (keep numeric_exact and numeric_distance, but remove value/unit splitting)
        num1, _ = self._parse_value_and_unit(val1)
        num2, _ = self._parse_value_and_unit(val2)

        metrics["numeric_exact"] = (
            1.0
            if num1 is not None and num2 is not None and num1 == num2
            else 0.0 if num1 is not None and num2 is not None else None
        )

        if num1 is not None and num2 is not None:
            metrics["numeric_distance"] = abs(num1 - num2)
        else:
            metrics["numeric_distance"] = None

        # Optional LLM judge for semantic equivalence (meaning equality)
        # Only call LLM if both values are non-null and cosine similarity is below threshold
        if llm_judge:
            # Check if both values are non-null
            both_non_null = not self._is_null(val1) and not self._is_null(val2)
            # Check if cosine similarity is below threshold (0.95)
            cosine_below_threshold = cosine_sim is not None and cosine_sim < 0.95

            if both_non_null and cosine_below_threshold:
                try:
                    judge_val = llm_judge(self._safe_str(val1), self._safe_str(val2))
                    if judge_val is None:
                        metrics["llm_meaning_equal"] = None
                    elif isinstance(judge_val, (int, float)):
                        metrics["llm_meaning_equal"] = float(judge_val)
                    elif isinstance(judge_val, bool):
                        metrics["llm_meaning_equal"] = 1.0 if judge_val else 0.0
                    else:
                        metrics["llm_meaning_equal"] = 1.0 if judge_val else 0.0
                except Exception:
                    metrics["llm_meaning_equal"] = None
            else:
                # Automatically assign True for null values or high cosine similarity
                metrics["llm_meaning_equal"] = 1.0
        else:
            metrics["llm_meaning_equal"] = None

        return metrics

    def equality_ratio(self, val1, val2, fuzzy_flag=False):
        # if both null/NaN -> equal
        """
        Compute equality ratio between two values of various types.
        Args:
            val1: First value.
            val2: Second value."""

        if self._is_null(val1) and self._is_null(val2):
            return 1.0

        # try numeric comparison if both convertible to float
        try:
            f1 = float(val1)
            f2 = float(val2)
        except Exception:
            f1 = f2 = None

        if f1 is not None and f2 is not None:
            return 1.0 if f1 == f2 else 0.0

        # fallback to string similarity
        return self.text_similarity(val1, val2, fuzzy_flag)
