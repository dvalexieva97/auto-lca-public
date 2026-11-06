import os

from dotenv import load_dotenv

load_dotenv()  # TODO This should be done somewhere else


class NLPConfig:
    """Configuration class for NLP processing."""

    # Topic Extraction parameters:
    BACKEND = "bertopic"
    COUNT_TOP_RANKS = 10

    # Sentence Transformer Topic Model parameters:
    N_NEIGHBORS = 15
    N_COMPONENTS = 5
    MIN_DIST = 0.0
    SIMILARITY_METRIC = "cosine"

    # TFIDF parameters:
    TFIDF_MAX_DF = 0.8
    TF_IDFIMIN_DF = 5
    DEFAULT_STOP_WORDS_LANGUAGE = "english"
    NGRAM_RANGE = (1, 2)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # For Text Ranking. Revisions pin the exact Hugging Face model files, so
    # cosine scores stay comparable across machines and re-downloads.
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
    BI_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    # Save output:
    DEFAULT_OUTPUT_FOLDER = "output/tests"

    TOKEN_ENCODING_MODEL = "cl100k_base"  # Recommended for gemini models

    MAX_INPUT_TOKEN_COUNT = 32766


class SogModelConfig:
    # Small mistral to test if good enough
    REASONER_MODEL = "mistral/mistral-small-2409"
    EXTRACTOR__MODEL = "mistral/mistral-small-2409"
    FALLBACK_MODEL = None
    API_BASE = "https://api.mistral.ai/v1"
    API_KEY = os.environ.get("MISTRAL_API_KEY")

    # If accuracy isn't sufficient, we can upgrade reasoner:
    # REASONER_MODEL = "mistral/mistral-large-2407"


class SogConfig:

    CONCEPT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "sog", "sample_config.json"
    )

    def __init__(self, model_config: SogModelConfig = None):
        if not model_config:
            model_config = SogModelConfig()
        self.model_config = model_config


# | Model                     | Size | Why it’s good                                                                              | Ollama tag             |
# | ------------------------- | ---- | ------------------------------------------------------------------------------------------ | ---------------------- |
# | **Llama 3.2 3B Instruct** | 3B   | Newest Meta release (Oct 2024). Strong at JSON consistency and reasoning for small models. | `llama3.2:3b-instruct` |
# | **Phi-3 Mini**            | 3.8B | Very strong on reasoning, summarization, extraction; small VRAM footprint.                 | `phi3:mini`            |
# | **Mistral 7B Instruct**   | 7B   | Excellent JSON reliability, still runs fine on 16 GB+ VRAM.                                | `mistral:7b-instruct`  |
# | **Gemma 2 2B**            | 2B   | Lightweight and surprisingly structured; less verbose than Phi.                            | `gemma2:2b`            |
# | **Qwen2.5 3B**            | 3B   | Hugely improved instruction and JSON adherence for its size.                               | `qwen2.5:3b`           |
