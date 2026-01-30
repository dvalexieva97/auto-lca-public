# To see available models:
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# TODO Add Async limiter - 150 requests per minute per model per region (for gemini-2.5-pro)
import random
import time
from collections import defaultdict
from copy import deepcopy
from functools import wraps

import tiktoken
from contextgem import (
    Document,
    DocumentLLM,
    DocumentLLMGroup,
)
from contextgem.internal.exceptions import LLMAPIError
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from auto_lca.process.nlp.config import NLPConfig, SogConfig
from auto_lca.process.nlp.extract import TextExtractor
from auto_lca.process.sog.aspect import ASPECT_TO_CONCEPT_MAP, ASPECTS
from auto_lca.process.sog.concept import scenarios
from auto_lca.process.sog.concept_dynamic import (
    build_concepts_and_aspects_from_config,
    build_concepts_from_config_scenario_constraint,
)
from auto_lca.shared.error import SOGError
from auto_lca.shared.schema_helper import SCENARIO_NAME

MAX_RETRIES = 8
num_retries_failed_request = 1  # By contextgem
BASE_DELAY = 5  # seconds
MAX_CONCEPTS_PER_CALL = 6  # 4
USE_CONCURRENCY = True
OVERWRITE_EXISTING = True
encoding = tiktoken.get_encoding(NLPConfig.TOKEN_ENCODING_MODEL)
SYSTEM_MESSAGE = """You are an expert in lifecycle assessment (LCA) and sustainability reporting. You goal is to extract structured data from research papers.
- For any field which could be measured, prioritize finding numeric values in the text (e.g. Midpoint indicators, Herd size etc.)
- Always include units for numeric values; if unknown, write 'units not stated'
- Do not hallucinate. If something does not exist in the text, do not make it up.
"""


class StructuredOutputGenerator(TextExtractor):
    """
    A class to generate structured outputs from documents using LLMs.
    """

    def __init__(
        self,
        config: SogConfig = None,
    ):
        """
        Initializes the StructuredOutputGenerator.
        """
        super().__init__()
        if not config:
            config = SogConfig()
        self.config = config
        self.model_config = config.model_config
        self.initialize_llms(self.model_config)
        self.encoding = encoding

    def initialize_llms(self, model_config):
        reasoner_model = self.model_config.REASONER_MODEL
        extractor_model = self.model_config.EXTRACTOR__MODEL
        fallback_model = self.model_config.FALLBACK_MODEL
        api_base = self.model_config.API_BASE
        api_key = self.model_config.API_KEY

        if not reasoner_model:
            raise ValueError("Reasoner model must be provided for SOG!")
        if not extractor_model:
            raise ValueError("Extractor model must be provided for SOG!")

        if fallback_model:
            fallback_llm = DocumentLLM(
                model=fallback_model,
                api_key=api_key,
                role="reasoner_text",
                is_fallback=True,
                api_base=api_base,
                num_retries_failed_request=num_retries_failed_request,
                system_message=SYSTEM_MESSAGE,
                # TODO Validate
                temperature=0.1,  # More deterministic
                top_p=0.95,
            )
        else:
            fallback_llm = None
        llm_reasoner = DocumentLLM(
            model=reasoner_model,
            api_key=api_key,
            role="reasoner_text",
            fallback_llm=fallback_llm,
            api_base=api_base,
            num_retries_failed_request=num_retries_failed_request,
            system_message=SYSTEM_MESSAGE,
            # TODO Validate
            temperature=0.1,  # More deterministic
            top_p=0.95,
        )
        llm_extractor = DocumentLLM(
            model=extractor_model,
            api_key=api_key,
            role="extractor_text",
            api_base=api_base,
            num_retries_failed_request=num_retries_failed_request,
            system_message=SYSTEM_MESSAGE,
            # TODO Validate
            temperature=0.1,  # More deterministic
            top_p=0.95,
        )
        llm_group = DocumentLLMGroup(llms=[llm_extractor, llm_reasoner])
        # If extractor provided, add extractor to group
        # Else, use only llm
        self.llm = llm_group if llm_group else llm_reasoner

    def pdf_paths_to_texts(pdf_paths):
        """
        Extracts text from a list of PDF paths.

        Args:
            pdf_paths (List[str]): List of paths to PDF files.

        Returns:
            List[str]: List of extracted texts from the PDFs.
        """
        # TODO Return dict of texts organized by intro / methods / results / discussion / conclusion
        texts = [self.extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]
        return texts

    def extract_scenarios(self, doc):
        """Extracts LCA scenarios (production systems) from paper.
        Scenarios are extracted from full doc, as they can be
        found anywhere in the paper.
        Returns: list of extracted scenarios:
        list[dict['Scenario Name':str, 'Scenario Description':str]]
        Side effect: resets doc concepts
        """
        doc.concepts = [scenarios]
        doc = self.extract_all_with_retry(
            doc,
            use_concurrency=USE_CONCURRENCY,
            overwrite_existing=OVERWRITE_EXISTING,
        )
        extracted_scenarios = doc.concepts[0].extracted_items[0].value["items"]
        if not doc.concepts or not extracted_scenarios:
            raise SOGError("No scenarios extracted from doc.")
        doc.concepts = []
        return extracted_scenarios

    def with_retry(func):
        """
        Decorator for retrying a function call with exponential backoff and jitter
        on LLMAPIError or ResourceExhausted.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"Attempt #{attempt} for {func.__name__}")
                try:
                    return func(*args, **kwargs)
                except (
                    LLMAPIError,
                    ResourceExhausted,
                    ServiceUnavailable,
                ) as e:

                    if attempt == MAX_RETRIES:
                        raise

                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    delay += random.uniform(0, delay / 2)  # jitter
                    print(f"Retrying after {delay:.2f}s due to {type(e).__name__}...")
                    time.sleep(delay)

        return wrapper

    @with_retry
    def extract_all_with_retry(
        self,
        doc,
        use_concurrency=USE_CONCURRENCY,
        overwrite_existing=OVERWRITE_EXISTING,
    ):
        """
        Handles 'Resource Exhausted' errors
        """

        return self.llm.extract_all(
            doc,
            use_concurrency=use_concurrency,
            overwrite_existing=overwrite_existing,
            # max_paragraphs_to_analyze_per_call=30,  # TODO Seems like when using this flag, it runs much much slower
        )

    @with_retry
    def extract_concepts_from_aspect_with_retry(
        self,
        aspect,
        doc,
        from_concepts=None,
        use_concurrency=USE_CONCURRENCY,
        overwrite_existing=OVERWRITE_EXISTING,
    ):
        """
        Handles 'Resource Exhausted' errors
        """

        return self.llm.extract_concepts_from_aspect(
            aspect,
            doc,
            from_concepts=None,
            use_concurrency=use_concurrency,
            overwrite_existing=overwrite_existing,
            max_items_per_call=MAX_CONCEPTS_PER_CALL,
            # max_paragraphs_to_analyze_per_call=30,  # TODO
        )

    @with_retry
    def extract_aspects_from_doc_with_retry(
        self,
        doc,
        use_concurrency=USE_CONCURRENCY,
        overwrite_existing=OVERWRITE_EXISTING,
    ):
        """
        Handles 'Resource Exhausted' errors
        """
        return self.llm.extract_aspects_from_document(
            doc,
            use_concurrency=use_concurrency,
            overwrite_existing=overwrite_existing,
            max_items_per_call=1,  # TODO Dehardcode
            # max_paragraphs_to_analyze_per_call=30,  # TODO
        )

    def extract_concepts_by_scenario(self, doc, extracted_scenarios):
        """
        Loops over extracted scenarios and
        extracts all concepts accordingly:
        1. Build concepts and aspects from config
        2. Split into two subdicts:
            - concepts which are on the scenario level ("by_scenario")
            - global concepts ("not_by_scenario")
        3. Extract non-scenario based aspects
        4. Loop over all scenarios,
            reassign scenario-based concepts by aspect,
            find all concepts by scenario
        Returns:
        results_list: list[dict] of extracted concepts
        tokens_dict: dict of approximate token usage
        """
        results_list = []
        tokens_dict = {}

        aspect_dict, non_aspect_concepts, aspect_concept_map = (
            build_concepts_and_aspects_from_config(
                config_path=self.config.CONCEPT_CONFIG_PATH,
            )
        )

        # 1. Extract non-scenario aspects and their concepts first
        doc.aspects = list(aspect_dict.values())
        # doc.concepts = non_aspect_concepts
        print(
            "Starting extraction of aspects and concepts not by scenarios:",
            [a.name for a in doc.aspects],
        )
        for aspect in doc.aspects:
            aspect.concepts = aspect_concept_map["not_by_scenario"][aspect.name]

        # doc = self.extract_all_with_retry(doc) # TODO DEL - This doesn't work then
        # with overriding the concepts
        doc.aspects = self.extract_aspects_from_doc_with_retry(doc)
        # TODO DEL
        extracted_non_scenario_concepts = self.unnest_concepts(doc)
        extracted_non_scenario_concepts = [
            x for x in extracted_non_scenario_concepts if "Scenarios" not in x.keys()
        ]
        # TODO DEL
        tokens_dict["aspect_tokens"] = self.count_token_usage_and_reset()

        # TODO Check if we want to get them here
        # scenarios = [x for x in doc.concepts if x.name == "Scenarios"]
        # extracted_scenarios = [
        #     x.value for scenario in scenarios for x in scenario.extracted_items
        # ]
        # doc.aspects = list(aspect_map["by_scenario"].values())  # TODO Enum

        for aspect in doc.aspects:
            aspect_name = aspect.name
            aspect.concepts = aspect_concept_map["by_scenario"].get(aspect_name, [])
            # aspect.concepts = asp.concepts if asp else []
            if not aspect.concepts:
                # TODO Log warning, not error
                raise ValueError(
                    f"No concepts found for aspect: {aspect.name}. "
                    f"Available aspects: {aspect_concept_map["by_scenario"].keys()}"
                )

        concepts_cache = {
            aspect.name: deepcopy(aspect.concepts) for aspect in doc.aspects
        }
        for scenario_dict in extracted_scenarios:
            scenario_name = scenario_dict["Scenario Name"]
            print(f"Processing scenario: {scenario_name}...")

            for aspect in doc.aspects:
                aspect_concepts = deepcopy(concepts_cache[aspect.name])
                if not aspect_concepts:
                    continue

                for concept in aspect_concepts:
                    concept.description = (
                        f"{concept.description} - for scenario '{scenario_name}'"
                    )
                aspect.concepts = aspect_concepts
                # 2. Per scenario, per aspect, extract concepts.
                aspect_concepts = self.extract_concepts_from_aspect_with_retry(
                    aspect=aspect,
                    doc=doc,
                    from_concepts=aspect.concepts,
                )
                aspect.concepts = (
                    aspect_concepts  # TODO Why do we override? Just for the unnest?
                )

            extracted = [{"Scenario": scenario_dict}] + self.unnest_concepts(doc)
            tokens_dict[scenario_dict["Scenario Name"]] = (
                self.count_token_usage_and_reset()
            )

            results_dict = {
                "concepts": extracted_non_scenario_concepts + extracted,
            }
            results_list.append(results_dict)
            print(f"Processed scenario: {scenario_name}")
        return results_list, tokens_dict

    def _extract_concepts_by_scenario_no_aspects(self, doc, extracted_scenarios):
        """
        Loops over extracted scenarios and
        extracts all concepts accordingly
        """
        results_list = []
        tokens_dict = {}

        # 1. Extract aspects first
        # doc.aspects = ASPECTS
        # extracted_aspects = self.extract_aspects_from_doc_with_retry(doc)
        # doc.aspects = extracted_aspects
        # tokens_dict["aspect_tokens"] = self.count_token_usage_and_reset()

        for scenario_dict in extracted_scenarios:
            scenario_name = scenario_dict["Scenario Name"]
            print(f"Processing scenario: {scenario_name}...")
            concepts = deepcopy(
                [y for x in ASPECT_TO_CONCEPT_MAP.values() for y in x if x]
            )
            for concept in concepts:
                # concept.structure["Scenario"] = scenario_description
                concept.description = (
                    f"{concept.description} - for scenario '{scenario_name}'"
                )
            doc.concepts = concepts
            doc = self.extract_all_with_retry(doc)
            extracted = [{"Scenario": scenario_dict}] + self.unnest_concepts(doc)
            tokens_dict[scenario_dict["Scenario Name"]] = (
                self.count_token_usage_and_reset()
            )  # TODO Enum

            results_dict = {
                "concepts": extracted,
            }
            results_list.append(results_dict)
            print(f"Processed scenario: {scenario_name}")
        return results_list, tokens_dict

    @classmethod
    def get_extracted_items(cls, concepts):
        """
        Gets extracted items from a concept
        """
        return [{x.name: y.value} for x in concepts for y in x.extracted_items]

    @classmethod
    def unnest_concepts(cls, doc):
        """
        Unnests concepts nested in aspects
        """
        unnested = []
        for aspect in doc.aspects:
            unnested += cls.get_extracted_items(aspect.concepts)

        return unnested + cls.get_extracted_items(doc.concepts)

    def count_token_usage_and_reset(self):
        """
        Counts total token usage
        Note: This method is not perfectly accurate, but conte
        Side effect: resets token usage
        """
        usages = self.llm.get_usage()
        total_count = 0
        for u in usages:
            calls = u.usage.calls
            if not calls:
                continue
            prompt_str = str(calls)
            prompt_token_count = len(self.encoding.encode(prompt_str))
            total_count += prompt_token_count
        print(f"Total token count: {total_count}")
        self.llm.reset_usage_and_cost()
        return total_count

    def _sog_pipeline_by_scenario(self, pdf_path: str):
        """
        Extracts concepts from a PDF file by:
        0. Extract text from pdf
        1. Extracting all scenarios from full text
        2. Extracting all concepts by scenario one by one
        (Looping over each scenario and running the concepts)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[dict]: List of extracted concepts.
        """
        text = self.extract_text_from_pdf(pdf_path)
        doc = Document(
            raw_text=text,
        )
        extracted_scenarios = self.extract_scenarios(doc)
        scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        scenario_tokens = self.count_token_usage_and_reset()
        print(f"{len(scenario_names)} scenarios extracted.")
        results, tokens_dict = self.extract_concepts_by_scenario(
            doc, extracted_scenarios
        )
        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum([v for v in tokens_dict.values()])

        return results, tokens_dict

    def sog_pipeline(self, pdf_path: str):
        """
        sog_pipeline_by_scenario_low_resource

        Extracts concepts from a PDF file by:
        0. Extract text from pdf
        1. Find and filter relevant paper sections
            with basic NLP, used for SOG
        2. Extract all scenarios from filtered text
        3. Extract concepts by scenario:
            3.1
        2. Extracting all concepts by scenario one by one
        (Looping over each scenario and running the concepts)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            results: List[dict]: List of extracted concepts.
            tokens_dict: dict: dict of approximate token usage per extraction step

        """
        full_text = self.extract_text_from_pdf(pdf_path)
        sections_dict = self.extract_sections_from_text(full_text)
        text = "\n".join([dic["section_text"] for dic in sections_dict.values()])

        doc = Document(
            raw_text=text,
        )
        # extracted_scenarios = self.extract_scenarios(doc)
        # TODO Dehardcode
        extracted_scenarios = [
            {SCENARIO_NAME: "Baseline Scenario (BS)", "scenario_description": ""},
            {
                SCENARIO_NAME: "Third Daily Milking Strategy (3MS)",
                "scenario_description": "",
            },
            {
                SCENARIO_NAME: "Anaerobic Digestion Strategy (ADS)",
                "scenario_description": "",
            },
        ]
        scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        scenario_tokens = self.count_token_usage_and_reset()
        print(f"{len(scenario_names)} scenarios extracted.")
        results, tokens_dict = self.extract_concepts_by_scenario(
            doc, extracted_scenarios=extracted_scenarios
        )
        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum([v for v in tokens_dict.values()])

        return results, tokens_dict

    def _sog_pipeline_at_once(self, pdf_path: str):
        """
        Extracts concepts from a PDF file by:
        0. Extract text from pdf
        1. Extracting all scenarios from full text
        2. Rebuilding all concepts to include the scenarios in the schema definition
        3. Extracting each concept for all scenarios
        (Looping over each scenario and running the concepts)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[dict]: List of extracted concepts.
        """
        tokens_dict = {}
        text = self.extract_text_from_pdf(pdf_path)
        doc = Document(
            raw_text=text,
        )
        extracted_scenarios = self.extract_scenarios(doc)
        scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        scenario_tokens = self.count_token_usage_and_reset()

        concepts, aspect_map = build_concepts_from_config_scenario_constraint(
            scenarios=scenario_names,
            config_path=self.config.CONCEPT_CONFIG_PATH,
        )

        print(f"{len(scenario_names)} scenarios extracted.")

        doc = self.assign_concepts_by_aspect(doc, concepts, aspect_map)
        doc = self.extract_all_with_retry(doc)

        tokens_dict["concept_tokens"] = self.count_token_usage_and_reset()
        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum([v for v in tokens_dict.values()])
        results = self._get_extracted([doc])
        return results, tokens_dict

    def _sog_pipeline_at_once_low_resource(self, pdf_path: str):
        """
        Extracts concepts from a PDF file by:
        0. Extract text from pdf
        01. Extract text sections and rejoin them to shorten text and
        ignore unnecessary information.
        1. Extracting all scenarios from full text
        2. Rebuilding all concepts to include the scenarios in the schema definition
        3. Extracting each concept for all scenarios
        (Looping over each scenario and running the concepts)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[dict]: List of extracted concepts.
        """
        tokens_dict = {}
        full_text = self.extract_text_from_pdf(pdf_path)
        sections_dict = self.extract_sections_from_text(full_text)
        text = "\n".join([dic["section_text"] for dic in sections_dict.values()])
        doc = Document(
            raw_text=text,
        )

        extracted_scenarios = self.extract_scenarios(doc)
        scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        scenario_tokens = self.count_token_usage_and_reset()

        concepts, aspect_map = build_concepts_from_config_scenario_constraint(
            scenarios=scenario_names,
            config_path=self.config.CONCEPT_CONFIG_PATH,
        )

        print(f"{len(scenario_names)} scenarios extracted.")

        doc = self.assign_concepts_by_aspect(doc, concepts, aspect_map)
        doc = self.extract_all_with_retry(doc)

        tokens_dict["concept_tokens"] = self.count_token_usage_and_reset()
        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum([v for v in tokens_dict.values()])
        results = self._get_extracted([doc])
        return results, tokens_dict

    def _sog_pipeline_low_resource(self, pdf_path: str):
        """
        Extracts concepts from a PDF file by:
        0. Extract text from pdf
        1. Extracting all scenarios from full text
        2. Rebuilding all concepts to include the scenarios in the schema definition
        3. Extracting each concept for all scenarios
        (Looping over each scenario and running the concepts)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[dict]: List of extracted concepts.
        """
        tokens_dict = {"concept_tokens": 0}
        text = self.extract_text_from_pdf(pdf_path)
        sections_dict = self.extract_sections_from_text(text)
        doc = Document(raw_text=text)

        # extracted_scenarios = self.extract_scenarios(doc)
        # scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        # scenario_tokens = self.count_token_usage_and_reset()

        # TODO DEL - test
        # extracted_scenarios = [{SCENARIO_NAME: "Baseline scenario"}]
        scenario_tokens = 0
        # scenario_names = [x[SCENARIO_NAME] for x in extracted_scenarios]
        scenario_names = [
            "Baseline Scenario (BS)",
            "Third Daily Milking Strategy (3MS)",
            "Anaerobic Digestion Strategy (ADS)",
        ]
        # TODO DEL
        print(f"{len(scenario_names)} scenarios extracted.")
        concepts, aspect_map = build_concepts_from_config_scenario_constraint(
            scenarios=scenario_names,
            config_path=self.config.CONCEPT_CONFIG_PATH,
        )
        docs = []
        for section_name, dic in sections_dict.items():
            print(f"Processing '{section_name}'..")
            if section_name not in aspect_map.keys():
                print(f"Nothing to fetch from section '{section_name}'. Skipping.")
                # TODO Log - Ignoring section 'section_name'
                continue

            section_text = dic["section_text"]  # TODO Enum
            if not section_text:
                # TODO Log warning
                section_text = text
            section_doc = Document(raw_text=section_text)
            target_aspect_map = {section_name: aspect_map.pop(section_name)}

            section_doc = self.assign_concepts_by_aspect(
                doc, concepts, target_aspect_map
            )
            section_doc = self.extract_all_with_retry(section_doc)
            docs.append(deepcopy(section_doc))
            tokens_dict["concept_tokens"] += self.count_token_usage_and_reset()
            print(f"Processed '{section_name}'..")

        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum([v for v in tokens_dict.values()])

        results = self._get_extracted(docs)
        return results, tokens_dict

    def assign_concepts_by_aspect(self, doc, concepts, aspect_map):
        """Assigns reformatted concepts where each concept gets
        nested per scenario.
        (e.g. {"Midpoints": {"CO2":float}} -->
        {"Midpoints": {"Scenario 1": {"CO2":float}, "Scenario 2": {"CO2":float}}})
        """
        doc.aspects = [x for x in ASPECTS if x.name in aspect_map.keys()]
        if not doc.aspects:
            raise ValueError(
                f"No aspects found between aspects {ASPECTS} and map: {aspect_map.keys()}"
            )
        for aspect in doc.aspects:
            aspect_concepts = deepcopy(aspect_map[aspect.name])
            if not aspect_concepts:
                continue
            aspect_concepts = [x for x in concepts if x in aspect_map[aspect.name]]
            if not aspect_concepts:
                raise ValueError(f"No concepts found for aspect {aspect.name}")
                # continue

            aspect.concepts = aspect_concepts

        return doc

    def assign_concepts_with_scenarios(self, doc, scenario_names):
        """Assigns reformatted concepts where each concept gets
        nested per scenario.
        (e.g. {"Midpoints": {"CO2":float}} -->
        {"Midpoints": {"Scenario 1": {"CO2":float}, "Scenario 2": {"CO2":float}}})
        """
        doc.aspects = ASPECTS
        for aspect in doc.aspects:
            # TODO Update
            aspect_concepts = deepcopy(ASPECT_TO_CONCEPT_MAP[aspect.name])
            if not aspect_concepts:
                continue
            for concept in aspect_concepts:
                concept.structure = {name: concept.structure for name in scenario_names}

            aspect.concepts = aspect_concepts

        return doc

    @classmethod
    def explode_extracted_by_keys(cls, data_list: list[dict], keys: list[str]):
        """
        Explodes nested dictionaries under specified keys into flat records.

        Example:
            Input:
                [
                    {
                        "Study details": {
                            "Baseline Scenario": {...},
                            "Third Milking Strategy": {...},
                        }
                    }
                ]
            Output:
                [
                    {
                        "Scenario name": "Baseline Scenario",
                        "data": {
                            "Study details": {...}
                        }
                    },
                    {
                        "Scenario name": "Third Milking Strategy",
                        "data": {
                            "Study details": {...}
                        }
                    }
                ]
        """
        exploded = []

        for data in data_list:
            for master_key, item in data.items():
                for key in keys:
                    nested = item.get(key)
                    if not nested:
                        continue
                    new_entry = {SCENARIO_NAME: key, "data": {master_key: nested}}
                    exploded.append(new_entry)

        results = cls._merge_exploded_by_scenario(exploded)
        return results

    @classmethod
    def _merge_exploded_by_scenario(
        cls, exploded: list[dict], scenario_key: str = SCENARIO_NAME
    ):
        """
        Merge multiple exploded entries that share the same Scenario name.
        """
        merged = defaultdict(lambda: {SCENARIO_NAME: None, "data": {}})

        for entry in exploded:
            scenename = entry.get(scenario_key)
            if not scenename:
                continue

            merged_entry = merged[scenename]
            merged_entry["Scenario name"] = scenename

            # Merge all nested data dictionaries
            for section, section_data in entry.get("data", {}).items():
                merged_entry["data"][section] = section_data

        # Convert dict -> list
        return list(merged.values())

    @classmethod
    def _get_extracted(cls, docs: list[Document]):
        """Gets extracted concept results from doc:
        both from aspects and directly from doc"""
        extracted = []
        # Fetch extracted concepts from each aspect
        for doc in docs:
            for aspect in doc.aspects:
                extracted += [
                    {x.name: y.value}
                    for x in aspect.concepts
                    for y in x.extracted_items
                ]
            # Fetch extracted concepts from doc
            extracted += [
                {x.name: y.value} for x in doc.concepts for y in x.extracted_items
            ]
        return extracted
