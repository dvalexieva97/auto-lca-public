import functools
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from mistralai import Mistral
from pydantic import BaseModel, Field, create_model

from auto_lca.process.nlp.extract import TextExtractor
from auto_lca.process.preprocess.config_parser import ensure_json_config
from auto_lca.process.sog.concept_dynamic import (
    build_concepts_and_aspects_from_config,
)
from auto_lca.shared.schema_helper import SCENARIO_NAME


def retry_with_backoff(
    max_retries=5, base_delay=1.0, max_delay=30.0, retriable_exceptions=(Exception,)
):
    """
    Decorator for exponential backoff with jitter.
    Retries on provided exception classes (e.g., rate-limit errors).
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except retriable_exceptions as e:
                    # Check if it's a rate limit error (common for Mistral)
                    is_last_attempt = attempt == max_retries - 1
                    if is_last_attempt:
                        raise

                    delay = min(max_delay, base_delay * (2**attempt))
                    delay = delay * (0.7 + random.random() * 0.6)  # jitter

                    print(
                        f"[retry_with_backoff] Attempt {attempt+1}/{max_retries}, "
                        f"error: {e}, sleeping {delay:.2f}s"
                        f"{str(e)}"
                    )

                    time.sleep(delay)

        return wrapper

    return decorator


class MistralStructuredExtractor(TextExtractor):
    """
    Structured output generator using Mistral's native structured outputs API.
    More token-efficient than contextgem wrapper.
    """

    def __init__(self, config_path: str = "src/auto_lca/process/sog/config.json"):
        super().__init__()
        # Convert CSV to JSON if needed, or use JSON directly
        self.config_path = ensure_json_config(config_path)
        self._temp_config_path = (
            self.config_path if self.config_path != config_path else None
        )
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        self.client = Mistral(api_key=self.api_key)
        self.model = (
            "mistral-small-latest"  # or "mistral-large-latest" for better accuracy
        )
        self._concept_models_cache = {}  # Cache Pydantic models

    @retry_with_backoff(
        max_retries=6, base_delay=1.0, max_delay=20.0, retriable_exceptions=(Exception)
    )
    def _safe_mistral_parse(self, *args, **kwargs):
        """
        Wrapper for Mistral API calls with retry/backoff.
        """
        return self.client.chat.parse(*args, **kwargs)

    def _parse_type_str_to_python_type(self, type_str: str):
        """Convert type string to Python type for Pydantic"""
        type_str = type_str.strip()

        # Handle Optional types
        if "| None" in type_str:
            base_type = type_str.replace("| None", "").strip()
            return Optional[self._parse_type_str_to_python_type(base_type)]

        # Handle List types
        if type_str.startswith("List["):
            inner = type_str[5:-1].strip()
            return List[self._parse_type_str_to_python_type(inner)]

        # Basic types
        type_map = {
            "str": str,
            "float": float,
            "int": int,
            "bool": bool,
        }

        if type_str in type_map:
            return type_map[type_str]

        # Default to str if unknown
        return str

    def _python_type_to_pydantic_field(self, python_type: Any) -> Any:
        """Convert Python type (from JsonObjectConcept.structure) to Pydantic field type"""
        # If it's already a type, use it directly
        if python_type in (str, int, float, bool):
            return Optional[python_type]

        # Handle Optional types
        origin = get_origin(python_type)
        if origin is Union:
            args = get_args(python_type)
            # Check if it's Optional (Union[SomeType, NoneType])
            if len(args) == 2 and type(None) in args:
                # Extract the non-None type
                non_none_type = next(arg for arg in args if arg is not type(None))
                if non_none_type in (str, int, float, bool):
                    return Optional[non_none_type]
                elif origin is list or get_origin(non_none_type) is list:
                    inner = (
                        get_args(non_none_type)[0] if get_args(non_none_type) else str
                    )
                    return Optional[List[inner]]
                return Optional[non_none_type]
            return python_type

        # Handle List types
        if origin is list:
            inner = get_args(python_type)[0] if get_args(python_type) else str
            return Optional[List[inner]]

        # Handle dict types
        if origin is dict:
            args = get_args(python_type)
            if args:
                return Optional[dict]
            return Optional[dict]

        # Default to Optional[str]
        return Optional[str]

    def _create_pydantic_model(
        self, concept_name: str, structure: Union[Dict[str, str], Dict[str, Any]]
    ) -> type[BaseModel]:
        """Create a Pydantic model from concept structure.

        Structure can be:
        - Dict[str, str]: field_name -> type_string (e.g., "str | None")
        - Dict[str, Any]: field_name -> Python_type (e.g., Optional[str])
        """
        # Check cache first
        if concept_name in self._concept_models_cache:
            return self._concept_models_cache[concept_name]

        # Create fields dict for Pydantic
        fields = {}
        for field_name, type_spec in structure.items():
            # Check if type_spec is a string (type string) or a Python type
            if isinstance(type_spec, str):
                # It's a type string, parse it
                python_type = self._parse_type_str_to_python_type(type_spec)
            else:
                # It's already a Python type from JsonObjectConcept
                python_type = self._python_type_to_pydantic_field(type_spec)

            fields[field_name] = (
                python_type,
                Field(default=None, description=field_name),
            )

        # Create Pydantic model dynamically
        model = create_model(concept_name, **fields)
        self._concept_models_cache[concept_name] = model
        return model

    def _create_dataclass_from_structure(
        self, concept_name: str, structure: Union[Dict[str, str], Dict[str, Any]]
    ) -> type:
        """Create a dataclass from concept structure (alternative to Pydantic).

        Note: Mistral's parse() works best with Pydantic, but this shows how
        to create dataclasses if needed for other purposes.
        """
        # Create fields dict for dataclass
        fields_dict = {}
        for field_name, type_spec in structure.items():
            if isinstance(type_spec, str):
                python_type = self._parse_type_str_to_python_type(type_spec)
            else:
                python_type = self._python_type_to_pydantic_field(type_spec)

            fields_dict[field_name] = (python_type, field(default=None))

        # Create dataclass dynamically
        return dataclass(type(concept_name, (), fields_dict))

    def extract_concept(
        self,
        text: str,
        concept_name: str,
        structure: Union[Dict[str, str], Dict[str, Any]],
        description: str,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract a single concept using Mistral's structured outputs.

        Args:
            text: Text to extract from
            concept_name: Name of the concept
            structure: Schema structure dict (field_name -> type_str or Python type)
            description: Concept description
            scenario_name: Optional scenario name for scenario-specific extraction

        Returns:
            Extracted data as dict
        """
        # Create Pydantic model (works with both type strings and Python types)
        pydantic_model = self._create_pydantic_model(concept_name, structure)

        # Build prompt
        if scenario_name:
            prompt = f"""Extract {concept_name} for scenario '{scenario_name}' from the following text.

{description}

IMPORTANT: Extract values specifically for the scenario '{scenario_name}'. If no data exists for this scenario, use null.

Text:
{text}"""
        else:
            prompt = f"""Extract {concept_name} from the following text.

{description}

If information is not explicitly stated, use null. Do not infer or guess values.

Text:
{text}"""
        usage = {}
        try:
            # Use Mistral's parse method with Pydantic model
            response = self._safe_mistral_parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in lifecycle assessment (LCA) and sustainability reporting. Extract structured data from research papers. Always include units for numeric values; if unknown, write 'units not stated'. Look into tables for data as a first priority. Do not hallucinate.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=pydantic_model,
                temperature=0.1,
                max_tokens=2048,
            )

            # Extract parsed data
            parsed_data = response.choices[0].message.parsed
            usage = dict(response.usage)

            # Convert Pydantic model to dict
            if isinstance(parsed_data, BaseModel):
                return parsed_data.model_dump(), usage
            return parsed_data, usage

        except Exception as e:
            print(f"Error extracting {concept_name}: {e}")
            # Return empty dict with null values
            return dict.fromkeys(structure.keys(), None), usage

    def extract_scenarios(self, text: str) -> List[Dict[str, str]]:
        """Extract scenarios from text"""
        from pydantic import BaseModel, Field

        class ScenarioItem(BaseModel):
            Scenario_Name: str = Field(description="Scenario name")
            Scenario_Description: str = Field(description="Scenario description")

        class ScenariosResponse(BaseModel):
            items: List[ScenarioItem]

        prompt = f"""Identify all distinct product systems, production systems, or scenarios described or analyzed in the LCA study.

Each scenario represents one version or variant of the system under study (e.g., baseline, reference, or alternative systems such as different farm types, management practices, technologies, or regions).

Scenarios can be described explicitly (e.g., "Scenario A: Baseline" / "Scenario B: Alternative") 
or implicitly (e.g., through study objectives, comparisons, or dataset descriptions such as "pasture-based farms", 
"conventional systems", "literature benchmarks", or "other LCA studies").

Include:
- Any modeled or referenced systems that are compared or evaluated.
- Any system whose results are reported separately (only those with qualitative results).
- Systems described through data sources (e.g., “survey of 10 farms”, “literature reference systems”).
- Specific farms (e.g. "Farm 1", "Farm B").
- If there are specific farms mentioned in tables, extract those farm scenarios as well. For example, the 10-20 scenarios which then get aggregated into one scenario.
Output one list item per distinct system/scenario.
Do not duplicate scenarios. Do not hallucinate.

Text:
{text}"""

        try:
            response = self._safe_mistral_parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert LCA analyst. Extract all scenarios from research papers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=ScenariosResponse,
                temperature=0.1,
            )

            scenarios = response.choices[0].message.parsed.items
            # Convert to your expected format
            return (
                [
                    {
                        SCENARIO_NAME: s.Scenario_Name,
                        "scenario_description": s.Scenario_Description,
                    }
                    for s in scenarios
                ],
                dict(response.usage),
            )
        except Exception as e:
            print(f"Error extracting scenarios: {e}")
            return ([], {})

    def extract_concepts_by_scenario(
        self,
        text: str,
        extracted_scenarios: List[Dict[str, str]],
        sections_dict: Optional[Dict] = None,
    ):
        """
        Extract concepts by scenario, matching your existing interface.

        Returns:
            results_list: List of extracted concepts per scenario
            tokens_dict: Token usage tracking
        """
        results_list = []
        tokens_dict = {
            "aspect_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }

        # Build concept maps
        _, _, aspect_concept_map = build_concepts_and_aspects_from_config(
            config_path=self.config_path,
        )

        # 1. Extract non-scenario concepts first
        print("Extracting non-scenario concepts...")
        extracted_non_scenario_concepts = []

        for aspect_name, concepts_list in aspect_concept_map["not_by_scenario"].items():
            # Get relevant section text
            if sections_dict and aspect_name in sections_dict:
                section_text = sections_dict[aspect_name].get("section_text", text)
            else:
                section_text = text

            for concept_obj in concepts_list:
                # JsonObjectConcept uses attributes, not dict keys
                concept_name = concept_obj.name
                concept_desc = concept_obj.description
                concept_structure = (
                    concept_obj.structure
                )  # This is already Python types

                # Extract concept
                extracted, tokens = self.extract_concept(
                    text=section_text,
                    concept_name=concept_name,
                    structure=concept_structure,
                    description=concept_desc,
                )

                extracted_non_scenario_concepts.append({concept_name: extracted})
                tokens_dict["aspect_tokens"]["prompt_tokens"] += tokens["prompt_tokens"]
                tokens_dict["aspect_tokens"]["completion_tokens"] += tokens[
                    "completion_tokens"
                ]
                tokens_dict["aspect_tokens"]["total_tokens"] += tokens["total_tokens"]

        # 2. Extract scenario-specific concepts
        for scenario_dict in extracted_scenarios:
            scenario_name = scenario_dict[SCENARIO_NAME]
            print(f"Processing scenario: {scenario_name}...")

            scenario_results = [{"Scenario": scenario_dict}]

            # Extract concepts per aspect
            for aspect_name, concepts_list in aspect_concept_map["by_scenario"].items():
                # Get relevant section text
                if sections_dict and aspect_name in sections_dict:
                    section_text = sections_dict[aspect_name].get("section_text", text)
                else:
                    section_text = text

                for concept_obj in concepts_list:
                    # JsonObjectConcept uses attributes, not dict keys
                    concept_name = concept_obj.name
                    concept_desc = concept_obj.description
                    concept_structure = (
                        concept_obj.structure
                    )  # This is already Python types

                    # Add scenario context to description
                    enhanced_desc = f"{concept_desc} - for scenario '{scenario_name}'"

                    # Extract concept
                    extracted, tokens = self.extract_concept(
                        text=section_text,
                        concept_name=concept_name,
                        structure=concept_structure,
                        description=enhanced_desc,
                        scenario_name=scenario_name,
                    )

                    scenario_results.append({concept_name: extracted})

            results_dict = {
                "concepts": extracted_non_scenario_concepts + scenario_results
            }
            results_list.append(results_dict)
            tokens_dict[scenario_name] = tokens

        return results_list, tokens_dict

    def _combine_results_sections(self, sections_dict: Dict) -> str:
        """
        Intelligently combine Results, Discussion, and Conclusion sections.
        These sections often contain overlapping content, so we deduplicate by paragraph.

        Returns:
            Combined and deduplicated text from results-related sections
        """
        # Get all results-related sections
        results_sections = ["Results", "Discussion", "Conclusion"]
        all_paragraphs = []
        seen_paragraphs = set()

        for section_name in results_sections:
            section_text = sections_dict.get(section_name, {}).get("section_text", "")
            if not section_text:
                continue

            # Split into paragraphs (by double newline or single newline for short sections)
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in section_text.split("\n") if p.strip()]

            for para in paragraphs:
                # Use first 100 chars as dedup key (handles minor variations)
                para_key = para[:100].lower().strip()
                if para_key and para_key not in seen_paragraphs:
                    seen_paragraphs.add(para_key)
                    all_paragraphs.append(para)

        return "\n\n".join(all_paragraphs)

    def sog_pipeline(self, pdf_path: str):
        """
        Main pipeline matching your existing interface.
        """
        # Extract text and sections
        full_text = self.extract_text_from_pdf(pdf_path)
        sections_dict = self.extract_sections_from_text(full_text)

        # Get Methods (stays separate)
        methods_text = sections_dict.get("Methods", {}).get("section_text", "")

        # Combine and deduplicate Results, Discussion, and Conclusion
        results_combined = self._combine_results_sections(sections_dict)
        sections_dict["Results"]["section_text"] = results_combined

        # Combine Methods with deduplicated results sections
        texts = [x for x in [methods_text, results_combined] if x]
        combined_text = "\n\n".join(texts)

        if not combined_text:
            print(
                f"WARNING: No sections detected, using full text instead: {pdf_path}."
            )
            combined_text = full_text
        # Join Methods and Results for scenarios which can be in either:
        sections_dict["Methods_Results"] = {
            "section_title": "Methods_Results",
            "section_text": combined_text,
        }
        # Extract scenarios
        extracted_scenarios, scenario_tokens = self.extract_scenarios(combined_text)
        if not extracted_scenarios:
            raise ValueError(f"No scenarios extracted for {pdf_path}")

        print(f"{len(extracted_scenarios)} scenarios extracted.")
        print(f"Extracted scenarios: {[x[SCENARIO_NAME] for x in extracted_scenarios]}")
        # Extract concepts
        results, tokens_dict = self.extract_concepts_by_scenario(
            text=combined_text,
            extracted_scenarios=extracted_scenarios,
            sections_dict=sections_dict,
        )

        tokens_dict["scenario_tokens"] = scenario_tokens
        tokens_dict["total"] = sum(
            [x.get("total_tokens", 0) for x in tokens_dict.values()]
        )

        return results, tokens_dict

    def process_pdfs_batch(
        self,
        pdf_paths: List[str],
        output_folder: str,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process multiple PDFs in batch.

        Args:
            pdf_paths: List of paths to PDF files to process
            output_folder: Folder to save results JSON files
            skip_existing: If True, skip PDFs that already have output files

        Returns:
            Dictionary with 'success', 'fails', and 'all_results' keys
        """
        import os
        import time

        from auto_lca.shared.util import save_list_to_json

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        all_results = {}
        fails = []
        success = []

        # Filter out already processed PDFs if skip_existing
        if skip_existing and os.path.exists(output_folder):
            existing_outputs = {
                x.replace(".json", "")
                for x in os.listdir(output_folder)
                if x.endswith(".json")
            }
            pdf_paths = [
                pdf_path
                for pdf_path in pdf_paths
                if os.path.splitext(os.path.basename(pdf_path))[0]
                not in existing_outputs
            ]

        print(f"Total PDFs for processing: {len(pdf_paths)}")

        for i, pdf_path in enumerate(pdf_paths):
            try:
                print(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_path}")
                s = time.time()

                # Extract PID from filename
                pid = os.path.splitext(os.path.basename(pdf_path))[0]
                save_path = os.path.join(output_folder, f"{pid}.json")

                # Process PDF
                results, tokens = self.sog_pipeline(pdf_path)

                e = time.time()
                exec_time = e - s

                # Prepare data to save
                to_save = {
                    "metadata": {
                        "exec_time": exec_time,
                        "pid": pid,
                        "pdf_path": pdf_path,
                        "model": self.model,
                    },
                    "tokens": tokens,
                    "results": results,
                }

                # Save results
                save_list_to_json(data_list=to_save, save_path=save_path)
                all_results[pid] = to_save
                success.append(pid)

                print(f"Token usage: {tokens.get('total', 'N/A')}")
                print(f"Execution time: {exec_time:.2f}s")
                print(f"Successfully processed {pdf_path}")
                print(f"Remaining PDFs: {len(pdf_paths) - i - 1}\n")

            except Exception as e:
                print(f"Failed to process {pdf_path}: {str(e)}")
                fails.append({"pid": pid, "pdf_path": pdf_path, "error": str(e)})

        return {
            "success": success,
            "fails": fails,
            "all_results": all_results,
        }
