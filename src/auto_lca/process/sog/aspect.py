# # Definion of Aspects for scientific papers


from contextgem import Aspect

from auto_lca.process.sog.concept import (
    BASIC_DAIRY,
    EXRA_DAIRY,
    GENERAL_CONCEPTS,
    IMPACT_CATEGORIES,
)

ASPECT_TO_CONCEPT_MAP = {
    "Abstract": [],
    "Introduction": [],
    "Methods": GENERAL_CONCEPTS + BASIC_DAIRY + EXRA_DAIRY,
    "Results": IMPACT_CATEGORIES,
    "Discussion": [],
    "Conclusion": [],
    "Supplementary Information": [],
}


aspect_definitions = [
    # {"name": "Abstract", "description": "Abstract of the study", "concepts": []},
    # {
    #     "name": "Introduction",
    #     "description": "Background and objectives",
    # },
    {
        "name": "Methods",
        "description": "Study design and methodology",
    },
    {
        "name": "Results",
        "description": "Findings and data",
    },
    # {"name": "Discussion", "description": "Interpretation of results", "concepts": []},
    # {"name": "Conclusion", "description": "Summary and implications", "concepts": []},
    # {
    #     "name": "Supplementary Information",
    #     "description": "Additional data and materials",
    #     "concepts": [],
    # },
]

ASPECTS = []
for aspect_def in aspect_definitions:
    aspect = Aspect(
        name=aspect_def["name"],
        description=aspect_def["description"],
        add_justifications=False,
    )
    ASPECTS.append(aspect)
