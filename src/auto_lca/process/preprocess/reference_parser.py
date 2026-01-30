import re
import time
from difflib import SequenceMatcher

import requests

CROSSREF_API = "https://api.crossref.org/works"


class ReferenceParser:
    def __init__(self) -> None:
        pass

    @classmethod
    def parse_reference(cls, reference_str):
        pattern = re.compile(
            r"(?P<authors>.+?)\s+\((?P<year>\d{4})\)\.\s+"
            r"(?P<title>.+?)\.\s+"
            r"(?P<journal>.+?),\s+"
            r"(?P<volume>\d+)"
            r"(?:\((?P<issue>\d+)\))?,\s+"
            r"(?P<pages>\d+-\d+)\."
        )

        match = pattern.match(reference_str)
        if not match:
            return None
        return match.groupdict()

    @classmethod
    def extract_doi(cls, reference_str):
        doi_pattern = re.compile(r"10.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
        match = doi_pattern.search(reference_str)
        return match.group(0) if match else None

    @classmethod
    def find_reference(cls, reference_str: str, delay=1, min_similarity=0.6):
        # Basic parameter extraction
        reference = cls.parse_reference(reference_str)
        if not reference:
            # TODO log
            return None
        authors = reference.get("authors", "").split("; ")[0].split()
        first_author = authors[0] if authors else ""
        issn = reference.get("issn")
        year = reference.get("year")
        filters = [
            f"issn:{issn}" if issn else None,
            f"from-pub-date:{year}" if year else None,
        ]
        filter_str = ",".join([f for f in filters if f])
        params = {
            "query.title": reference["title"],
            "query.author": first_author,
            "filter": filter_str,
            "rows": 5,
        }

        time.sleep(delay)  # Polite API usage
        response = requests.get(CROSSREF_API, params=params).json()
        data = response.get("message", {}).get("items", [])
        best_match = None
        highest_similarity = 0
        for item in data:
            title_sim = SequenceMatcher(
                None, reference["title"].lower(), item.get("title", [""])[0].lower()
            ).ratio()

            if title_sim > highest_similarity:
                highest_similarity = title_sim
                best_match = item
        # TODO Raise exception?
        return (
            best_match if best_match and highest_similarity > min_similarity else None
        )


# # Example
# ref = (
#     "Berton, M., Cesaro, G., Gallo, L., Pirlo, G., Ramanzin, M., Tagliapietra, F., & Sturaro, E. "
#     "(2016). Environmental impact of a cereal-based intensive beef fattening system according "
#     "to a partial Life Cycle Assessment approach. Livestock Science, 190, 81-88."
# )
# ref = """Wiedemann, S., Davis, R., McGahan, E., Murphy, C., & Redding, M. (2016). Resource use and greenhouse gas emissions from grain-finishing beef cattle in seven Australian feedlots: a life cycle assessment. Animal Production Science, 57(6), 1149-1162."""

# ref = """Berton, M., Cesaro, G., Gallo, L., Pirlo, G., Ramanzin, M., Tagliapietra, F., & Sturaro, E. (2016).
# Environmental impact of a cereal-based intensive beef fattening system according to a partial Life Cycle Assessment approach.
# Livestock Science, 190, 81-88."""
# ref = """Zira, S., Röös, E., Rydhmer, L., & Hoffmann, R. (2023). Sustainability assessment of economic, environmental and social impacts, feed-food competition and economic robustness of dairy and beef farming systems in South Western Europe. Sustainable Production and Consumption, 36, 439-448."""
# # reference=sample_ref

# import pandas as pd

# path = "src/auto_lca/data/input/Inventory_07_29.xlsx - Inventory (original).csv"
# df = pd.read_csv(path)
# refs = list(df["BASIC INFORMATION"].dropna().unique())[:10]
# res = []
# for ref in refs:
#     best_match = ReferenceParser.find_reference(ref)
#     if best_match:
#         best_match["title"]
#         best_match["author"]
#         doi = best_match["DOI"]
#         print(f"DOI: {doi}")
#     else:
#         doi = None
#     res.append(doi)

# new_df = pd.DataFrame(res)
# new_df.to_csv("parsed_dois.csv")
# df["doi"] = res
