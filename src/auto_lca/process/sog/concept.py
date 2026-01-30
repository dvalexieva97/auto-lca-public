from typing import List

from contextgem import JsonObjectConcept

scenarios_description = """
Identify all distinct product systems, production systems, or scenarios described or analyzed in the LCA study. 
Each scenario represents one version or variant of the system under study (e.g., baseline, reference, or alternative systems such as different farm types, management practices, technologies, or regions).

Scenarios can be described explicitly (e.g., "Scenario A: Baseline" / "Scenario B: Alternative") 
or implicitly (e.g., through study objectives, comparisons, or dataset descriptions such as "pasture-based farms", 
"conventional systems", "literature benchmarks", or "other LCA studies").

Include:
- Any modeled or referenced systems that are compared or evaluated.
- Any system whose results are reported separately (even if only qualitatively).
- Systems described through data sources (e.g., “survey of 10 farms”, “literature reference systems”).

Output one list item per distinct system/scenario, with:
- Scenario Name: concise label (e.g., “Organic Milk System”, “Literature Reference Milk System”)
- Scenario Description: a short description of what defines that system or how it differs.
"""

scenarios = JsonObjectConcept(
    name="Scenarios",
    description=scenarios_description,
    structure={
        "items": [{"Scenario Name": str, "Scenario Description": str}],
    },
    singular_occurrence=True,
    add_references=True,
    llm_role="reasoner_text",
    add_justifications=True,
)

methods = JsonObjectConcept(
    name="Lifecycle Assessment Characteristics",
    description="Lifecycle assessment Characteristics",
    structure={
        "Functional Unit": str,
        "System Boundary": str,
        # Literal["Cradle to Gate", "Cradle to Grave"],  # TODO Enum at some point
        "Methodology": str | None,
        "Emissions model": str | None,
        "Database (background)": str | None,
        "Software": str | None,
        "Limitations": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)


study_details = JsonObjectConcept(
    name="Study details",
    description="Details about the study: location, years (data period)",
    structure={
        "Reference": str | None,
        "Country": str | None,
        "Specific location": str | None,
        "Latitude": float | None,
        "Longitude": float | None,
        "Climate": str | None,
        "Year (publication)": int | None,
        "Year (data)": List[int],
        "Activity data": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

allocation = JsonObjectConcept(
    name="Allocation",
    description="Allocation approach used in the study",
    structure={
        "Type of allocation": str | None,
        "Allocation %": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

co2_eq = JsonObjectConcept(
    name="CO2- Equivalence",
    description="CO2-equivalence factors reported in the study",
    structure={
        "CH4": float | None,
        "N2O": float | None,
        "Others": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)


environmental_aspects = JsonObjectConcept(
    name="Environmental Aspects",
    description="Environmental aspects of the study, including climate, rainfall, temperature, soil type, elevation, land use, vegetation/soil cover, solar radiation, wind, and potential evapotranspiration.",
    structure={
        "CLIMATE": str | None,
        "RAINFALL": str | None,
        "TEMPERATURE": str | None,
        "SOIL TYPE": str | None,
        "ELEVATION": str | None,
        "LAND USE": str | None,
        "VEGETATION/ SOIL COVER": str | None,
        "SOLAR RADIATION": str | None,
        "WIND": str | None,
        "POTENTIAL EVAPOTRANSPIRATION": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)


GENERAL_CONCEPTS = [
    # functional_unit,
    # system_boundary,
    methods,
    study_details,
    # environmental_aspects,
    allocation,
    co2_eq,
]

characterization = JsonObjectConcept(
    name="Characterization",
    description="Study characterization: production system, LCA duration, and other outputs/info",
    structure={
        "Production system": str | None,
        "Duration of the LCA [year]": float | None,
        "Other outputs and information": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)
####
## Animal data:
####
# New: Animals general summary
animals = JsonObjectConcept(
    name="Animals",
    description="General animal-related descriptors for the system",
    structure={
        "Breed": str | None,
        "Herd size": int | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# New: Cattle breeding (milking cows)
cattle_breeding_milking = JsonObjectConcept(
    name="Cattle breeding (milking cows)",
    description="Details for milking cows population",
    structure={
        "Number": int | None,
        "Mean body weight [kg]": float | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# New: Cattle breeding (replacement heifers)
cattle_breeding_replacement_heifers = JsonObjectConcept(
    name="Cattle breeding (replacement heifers)",
    description="Details for replacement heifers",
    structure={
        "Number": int | None,
        "Mean body weight [kg]": float | None,
        "Replacement rate": float
        | None,  # fraction or percentage (specify in justification)
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# New: Dry cows
dry_cows = JsonObjectConcept(
    name="Dry cows",
    description="Details for dry cows",
    structure={
        "Number": int | None,
        "Mean body weight [kg]": float | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# New: Other animals
other_animals = JsonObjectConcept(
    name="Other animals",
    description="Other animals and related outputs (culled, calves, etc.)",
    structure={
        "Number": int | None,
        "Culled cows": int | None,
        "Mean male and surplus female calves weight and/or number": str
        | None,  # freeform to allow weight or count
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

BASIC_DAIRY = [
    characterization,
    animals,
    cattle_breeding_milking,
    cattle_breeding_replacement_heifers,
    dry_cows,
    other_animals,
]

####
## Other Agricultural Data:
####
# Farm output
farm_output = JsonObjectConcept(
    name="Farm output",
    description="Outputs from the farm including milk and meat production metrics",
    structure={
        "Milk production/sold": str | None,
        "Milk yield": str | None,
        "Milk per ha": str | None,
        "Protein content": str | None,
        "Fat content": str | None,
        "Meat or liveweight sold and/or produced": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Breeding
breeding = JsonObjectConcept(
    name="Breeding",
    description="Breeding and reproduction metrics",
    structure={
        "Stocking rate": str | None,
        "Calving rate": str | None,
        "Mortality rate": str | None,
        "First breeding": str | None,
        "Calving interval": str | None,
        "Age at first calving": float | None,
        "Lactation info": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Pasture
pasture = JsonObjectConcept(
    name="Pasture",
    description="Pasture management, species and yields",
    structure={
        "Pasture management and/or intake": str | None,
        "Species": str | None,
        "Forage/pasture yield": str | None,
        "Forage utilization rate": str | None,
        "Seeds": str | None,
        "Grazing season and/or period": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Purchased feed
purchased_feed = JsonObjectConcept(
    name="Purchased feed",
    description="Purchased feed products and amounts",
    structure={"Purchased products": str | None},
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Feed information
feed_information = JsonObjectConcept(
    name="Feed information",
    description="Nutritional and compositional feed details",
    structure={
        "Concentrated feed": str | None,
        "Crude protein": str | None,
        "Neutral detergent fiber": str | None,
        "Gross energy": str | None,
        "Digestibility energy": str | None,
        "Feed energy conversion ratio": str | None,
        "Chemical composition": str | None,
        "DMI": str | None,
        "Yield": str | None,
        "Self sufficiency": str | None,
        "Other info": str | None,
    },
    singular_occurrence=False,
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# On-farm feeds intake
on_farm_feeds_intake = JsonObjectConcept(
    name="On-farm feeds intake",
    description="On-farm produced feed intake and totals",
    structure={
        "Hay": str | None,
        "Maize": str | None,
        "Silage": str | None,
        "Grass/pasture": str | None,
        "Others": str | None,
        "Total intake": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Farm management
farm_management = JsonObjectConcept(
    name="Farm management",
    description="Management practices and chemical usage",
    structure={
        "Farm practices and productivity": str | None,
        "Chemical products": str | None,
        "Notes on management": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Manure
manure = JsonObjectConcept(
    name="Manure",
    description="Manure management and quantities",
    structure={
        "Manure management": str | None,
        "Quantity": str | None,
        "Applied to land (yes/no) and amount": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Fertilizers
fertilizers = JsonObjectConcept(
    name="Fertilizers",
    description="Fertilizer usage on the farm",
    structure={
        "Fertilizer use": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Other requirements
other_requirements = JsonObjectConcept(
    name="Other requirements",
    description="Bedding and other material requirements",
    structure={
        "Bedding materials": str | None,  # type -> amount/unit
        "Others": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Farm land
farm_land = JsonObjectConcept(
    name="Farm land",
    description="Land area and land-use details",
    structure={
        "Farm area": str | None,
        "Pasture and/or grazing area": str | None,
        "Off-farm area": str | None,
        "Land use": str | None,
        "Area per animal (paddocks)": str | None,
        "Others": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Energy and other resources
energy_resources = JsonObjectConcept(
    name="Energy and other resources",
    description="Energy, fuel and water consumption",
    structure={
        "Electricity and heat": dict[str, str] | None,  # type -> amount/unit
        "Fuels": dict[str, str] | None,  # fuel type -> amount/unit
        "Water consumption": dict[str, str] | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# Impact categories / emissions concepts
soil_sequestration = JsonObjectConcept(
    name="Soil sequestration",
    description="Soil organic carbon sequestration metrics",
    structure={
        "Soil organic carbon sequestration (kg C/ha/year)": str | None,
        "Measurement basis / method": str | None,
        "Uncertainty / notes": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

soil_emissions = JsonObjectConcept(
    name="Soil emissions",
    description="Total and breakdown of soil-related emissions",
    structure={
        "Total soil emissions (kg CO2e)": str | None,
        "Soil N2O (kg N2O)": str | None,
        "Soil CH4 (kg CH4)": str | None,
        "Notes / measurement basis": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

emissions = JsonObjectConcept(
    name="Emissions (detailed)",
    description="Detailed emission sources including enteric, manure and energy CO2",
    structure={
        "Enteric fermentation (kg CO2e)": str | None,
        "Manure CH4 (kg CH4)": str | None,
        "Other CH4 related emissions (kg CH4)": str | None,
        "Manure N2O (kg N2O)": str | None,
        "N2O crop residues (kg N2O)": str | None,
        "Other N2O related emissions (kg N2O)": str | None,
        "Energy CO2 (kg CO2)": str | None,
        "Other CO2 related emissions (kg CO2)": str | None,
        "Net farm emissions (kg CO2e)": str | None,
        "Total emissions (kg CO2e)": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

midpoints = JsonObjectConcept(
    name="Midpoint impact categories",
    description="Reported midpoint impact results (acidification, eutrophication, carbon footprint, etc.)",
    structure={
        "Acidification potential": str | None,
        "Eutrophication potential": str | None,
        "Carbon footprint (kg CO2e)": str | None,
        "GWP": str | None,
        "Final GWP (kg CO2e/kg FPCM)": str | None,
        "Type of allocation": str | None,
        "Notes / method (GWP factors etc.)": str | None,
    },
    add_references=False,
    llm_role="reasoner_text",
    add_justifications=False,
)

# update EXRA_DAIRY to include the new concepts
EXRA_DAIRY = [
    farm_output,
    breeding,
    pasture,
    purchased_feed,
    feed_information,
    on_farm_feeds_intake,
    farm_management,
    manure,
    fertilizers,
    other_requirements,
    farm_land,
]

IMPACT_CATEGORIES = [
    energy_resources,
    soil_sequestration,
    soil_emissions,
    emissions,
    midpoints,
]
# single DAIRY_CONCEPTS assignment
DAIRY_CONCEPTS = GENERAL_CONCEPTS + BASIC_DAIRY + EXRA_DAIRY + IMPACT_CATEGORIES
