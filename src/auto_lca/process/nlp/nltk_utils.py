"""Centralized NLTK data management.

This module handles all NLTK data downloads and provides a clean interface
for accessing NLTK resources like stopwords.
"""

import logging
from typing import Set

import nltk

logger = logging.getLogger(__name__)

# Track which resources have been downloaded
_downloaded_resources: Set[str] = set()


def ensure_nltk_data(*resource_names: str) -> None:
    """Ensure NLTK data resources are downloaded.

    Args:
        *resource_names: Names of NLTK data resources to download (e.g., 'stopwords', 'punkt')

    Example:
        >>> ensure_nltk_data('stopwords', 'punkt')
    """
    for resource_name in resource_names:
        if resource_name in _downloaded_resources:
            continue

        # Check if resource is already available by trying to access it
        resource_available = False
        try:
            if resource_name == "stopwords":
                # Check if stopwords corpus is available
                from nltk.corpus import stopwords

                stopwords.words("english")
                resource_available = True
            elif resource_name == "punkt":
                # Check if punkt tokenizer is available
                nltk.data.find("tokenizers/punkt")
                resource_available = True
            else:
                # Generic check - try to find the resource
                try:
                    nltk.data.find(resource_name)
                    resource_available = True
                except LookupError:
                    pass
        except (LookupError, ImportError):
            # Resource not found, will download below
            pass

        if not resource_available:
            # Resource not found, download it
            logger.info(f"Downloading NLTK resource: {resource_name}")
            try:
                nltk.download(resource_name, quiet=True)
                _downloaded_resources.add(resource_name)
            except Exception as e:
                logger.warning(
                    f"Failed to download NLTK resource '{resource_name}': {e}"
                )
                # Try alternative download method
                try:
                    nltk.download(resource_name, quiet=True, raise_on_error=True)
                    _downloaded_resources.add(resource_name)
                except Exception as e2:
                    logger.error(
                        f"Failed to download '{resource_name}' after retry: {e2}"
                    )
                    raise
        else:
            _downloaded_resources.add(resource_name)


def get_stopwords(language: str = "english") -> Set[str]:
    """Get stopwords for a given language.

    Args:
        language: Language code (default: "english")

    Returns:
        Set of stopwords

    Example:
        >>> stopwords = get_stopwords("english")
        >>> "the" in stopwords
        True
    """
    ensure_nltk_data("stopwords")
    from nltk.corpus import stopwords

    return stopwords.words(language)


ensure_nltk_data("stopwords")
_nltk_stopwords = get_stopwords()

stopwords = list(set(_nltk_stopwords + ["&"]))
