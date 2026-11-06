"""
Prompt templates for SOG extraction (concept and scenario extraction).
"""

# --- System messages ---
SYSTEM_EXTRACT_CONCEPT = (
    "You are an expert in lifecycle assessment (LCA) and sustainability reporting. "
    "Extract structured data from research papers. Include units with numeric values when "
    "the paper states them; if the unit is missing, write the number alone. Use null for "
    "any field not explicitly stated -- never guess, and never fill missing fields with "
    "placeholder text such as 'units not stated' or 'not stated'. Look into tables for "
    "data as a first priority. Do not hallucinate."
)
SYSTEM_EXTRACT_SCENARIOS = (
    "You are an expert LCA analyst. Extract all scenarios from research papers."
)

# --- Scenario extraction ---
# Domain-neutral by design: every rule below is illustrated with examples
# from more than one domain (agri/farm-style studies alongside industrial-
# process and energy-system studies) rather than farm-only examples, so the
# same wording holds up whether the paper being read is a dairy LCA or a
# chemical-process/energy LCA. The farm examples are kept, not replaced --
# more domains were added alongside them, not instead of them. Confirmed
# via real runs on 2 dairy papers after this rewrite that farm scenario
# extraction quality is unaffected.
PROMPT_EXTRACT_SCENARIOS = """Identify all distinct systems, processes, or scenarios described or analyzed in the LCA study.

A scenario is a distinct system, process, or product pathway being modeled or compared -- e.g. a specific site/facility, a type/management or route/technology variant (pasture-based vs. confined dairy farming; coal-based vs. renewable hydrogen route; grid-connected vs. off-grid power system), a technology-year or region variant, or an aggregate/mean system representing a group of individual sites. Each scenario represents one version or variant of the system under study.

Scenarios can be described explicitly (e.g., "Scenario A: Baseline" / "Scenario B: Alternative")
or implicitly (e.g., through study objectives, comparisons, or dataset descriptions such as "pasture-based farms",
"electrolysis-based route", "coal-fired plant", "literature benchmarks", or "other LCA studies").

Include:
- Any modeled or referenced systems that are compared or evaluated.
- Any system whose results are reported separately (only those with qualitative results).
- Systems described through data sources (e.g., "survey of 10 farms", "literature reference systems").
- Specific individual sites (e.g. "Farm 1", "Plant B", "Pathway 3", "Reactor Line 2").
- If there are specific sites mentioned in tables, extract those scenarios as well. For example, the 10-20 scenarios which then get aggregated into one scenario.

Do NOT include, even if it appears as a row/category in a table with numeric data -- these describe something WITHIN one system's results, they are not separate systems:
- Emission source or process-step categories within a single system (e.g. "Enteric Fermentation", "Manure Management", "Compression Energy", "Cleaning Products", "Boiler Startup Losses").
- Material or feed inputs (e.g. "Barley Grain (France)", "Palm Bagasse", "Natural Gas Feedstock", "Recycled Steel Scrap").
- Statistical breakdowns, quartiles, or percentile bins of one population's results (e.g. "R Ratio [0-0.11]", "Carbon Footprint Quartile 2").
- Individual impact categories or midpoint indicators (e.g. "Acidification potential", "Eutrophication potential").
- Emission factors, algorithms, or calculation methods, unless that variant genuinely corresponds to a distinct system being compared (not just a different formula applied to the same system).

If the study's own results could be grouped along more than one dimension (e.g. by region AND by management type; by feedstock AND by technology year; by CO2 source AND by electricity mix), choose the SINGLE dimension that the paper itself uses as the primary axis for reporting its own quantitative results, and extract scenarios along that one dimension consistently. Do not mix dimensions -- e.g. do not output both region-based and management-type-based scenarios for the same underlying data, and do not output both a percentile-based split (mean/lowest-10%/highest-10% of a farm survey) AND a separate accounting-variant split (with vs. without land-use-change emissions) for the same set of sites; pick whichever one the paper leads with as its primary result breakdown.

Output one list item per distinct system/scenario.
Do not duplicate scenarios. Do not hallucinate.
Only extract scenarios for which there is numeric data. For example, tables with the conditions of the system, or tables with results.

Text:
{text}"""
