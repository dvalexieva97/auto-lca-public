from auto_lca.process.nlp.nltk_utils import stopwords
from auto_lca.process.sog.mistral_sog import MistralStructuredExtractor


def test_nltk():
    assert "while" in stopwords


def test_extractor():
    """Test the add_numbers function."""
    extractor = MistralStructuredExtractor()
    assert extractor._parse_type_str_to_python_type("float") == float
