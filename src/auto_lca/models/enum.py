from enum import Enum


class BaseEnum(Enum):
    """
    Base Enum class to ensure all enums have a string representation.
    """

    def __str__(self):
        return self.value


# Search-related Enums:
class TopicExtractorBackend(BaseEnum):
    """
    Enum for the topic extractor backend.
    """

    BERTopic = "bertopic"
    TFIDF = "tfidf"


class AcademicSearchEngine(BaseEnum):
    """
    Enum for the academic search engine.
    """

    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPEN_ALEX = "OpenAlex"
    WEB_OF_SCIENCE = "Web of Science"
    LENS_ORG = "Lens.org"
    MANUAL = "Manual"


class PaperSource(BaseEnum):
    """
    Enum for the paper source.
    """

    SEMANTIC_SCHOLAR = "Semantic Scholar"
    SCIHUB = "Sci-Hub"
    MANUAL = "Manual"
    PREDOWNLOADED = "Pre-downloaded"


# Science-related Enums:
class SystemBoundary(Enum):
    """
    Enum for the system boundary in lifecycle assessment.
    """

    CRADLE_TO_GATE = "Cradle-to-Farm Gate"
    CRADLE_TO_GRAVE = "Cradle-to-Grave"
    GATE_TO_GATE = "Gate-to-Gate"
    FARM_TO_FORK = "Farm-to-Fork"
    CRADLE_TO_EXPERIMENT_GATE = "Cradle-to-Experiment Gate"
    CRADLE_TO_SLAUGHTERHOUSE_GATE = "Cradle-to-Slaughterhouse Gate"
    FINISHING_PHASE = "Finishing Phase"


class FunctionalUnitBeef(Enum):
    """
    Enum for the functional unit in lifecycle assessment.
    """

    KG_LIVE_WEIGHT = "1 kg live weight (LW)"
    KG_CONSUMED_BONELESS_EDIBLE_BEEF = "1 kg of consumed, boneless, edible beef"
    KG_CARCASS_WEIGHT = "1 kg carcass weight"
    KG_BODY_WEIGHT_GAIN = "1 kg body weight gain (BWG)"
    KG_LIVE_WEIGHT_AND_CARCASS_WEIGHT = "1 kg live weight (LW) and 1 kg carcass weight"
    KG_LIVE_WEIGHT_GAIN = "1 kg live weight gain (LWG)"
    KG_CARCASS_WEIGHT_GAIN = "1 kg carcass weight gain (CWG)"
    KG_MEAT_WEIGHT_GAIN = "1 kg meat weight gain (MWG)"
    KG_BONE_FREE_MEAT = "1 kg bone-free meat"
    KG_LIVE_BODY_WEIGHT = "1 kg live body weight (LBW)"
    KG_COLD_CARCASS_STEER_WEIGHT = "1 kg cold carcass steer weight"
    KG_COOKED_BEEF = "1 kg cooked beef"

    KG_BODY_WEIGHT_GAIN_BWG = "1 kg of body weight gained (BWG)"
    KG_LIVE_WEIGHT_LW_FINISHED_BULLS_HEIFERS_CULLED_COWS_WEANLINGS = "1 kg live weight (LW) considering finished bulls and heifers, culled cows and weanlings sold to other farmers as breeding animals"
    KG_LIVE_WEIGHT_AND_TONNE_DRY_MATTER_FORAGE = (
        "1 kg live weight (LW) and 1 t dry matter of forage self-produced"
    )
    KG_OF_LIVE_BODY_WEIGHT_LBW = "1 kg of live body weight (LBW)"
    KG_GAIN_DAY_ANIMAL_FINISHED_ANIMAL = "1 kg gain, 1 day.animal, 1 finished animal"
    KG_LWG_CARCASS_WEIGHT_GAIN_CWG_MEAT_WEIGHT_GAIN_MWG = (
        "1 kg LWG, 1 kg carcass weight gain (CWG), and 1 kg meat weight gain (MWG)"
    )
    KG_CARCASS_WEIGHT_CW = "1 kg carcass weight (CW)"
    KG_LIVE_WEIGHT_GAIN_GATE_TO_GATE = (
        "1 kg of live weight gain (LWG) - due to gate to gate approach"
    )
    LIVE_WEIGHT_PRODUCED_LWP_YEAR = "live weight produced (LWP) during one year"
    KG_BODY_WEIGHT_CARCASS_WEIGHT_EDIBLE_PORTION_BRAZILIAN_PRIMAL_CUT = "1 kg body weight, 1 kg carcass weight, 1 kg carcass edible portion of the sum of edible portions of the Brazilian primal cut"
    KG_BODY_WEIGHT_PRODUCED_BWP = "1 kg of body weight produced (BWP)"
    KG_BEEF_CARCASS_LEAVING_SLAUGHTERHOUSE = (
        "1 kg of beef carcass leaving the slaughterhouse"
    )
    KG_LIVE_WEIGHT_GAIN_FARM_GATE = "1 kg live weight gain at the farm gate"
    KG_LIVE_WEIGHT_AND_CARCASS_WEIGHT_CW = (
        "1 kg of live weight (LW), 1 kg of carcass weight (CW)"
    )
    TONNE_LIVE_WEIGHT_LW = "1 t live weight (LW)"
    MARKETED_BEEF_CALF_8_MONTHS = "1 marketed beef calf at 8 months of age"
    KG_COLD_CARCASS_STEER_WEIGHT_DUP = "1 kg of cold carcass steer weight"
    ONE_HECTARE_YEAR_TONNE_CRUDE_PROTEIN_TCP = (
        "one hectare and year (ha⋅a), and one tonne of crude protein (tCP)"
    )
    THOUSAND_KG_PROTEIN = "1000 kg protein"
