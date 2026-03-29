import os

from dotenv import load_dotenv

from auto_lca.shared.schema_helper import GCLOUD_GEMINI_API_KEY

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

    # For Text Ranking:
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
    # Save output:
    DEFAULT_OUTPUT_FOLDER = "output/tests"

    TOKEN_ENCODING_MODEL = "cl100k_base"  # Recommended for gemini models

    MAX_INPUT_TOKEN_COUNT = 32766


class SogModelConfig:

    # REASONER_MODEL = "gemini/gemini-2.5-pro"
    REASONER_MODEL = "gemini/gemini-2.5-flash"
    # REASONER_MODEL = "gemini/gemini-2.5-flash-lite"
    EXTRACTOR__MODEL = "gemini/gemini-2.0-flash-lite"
    # EXTRACTOR__MODEL = "gemini/gemini-2.5-flash-lite"
    FALLBACK_MODEL = "gemini/gemini-2.0-flash"
    API_BASE = None
    API_KEY = os.environ.get(GCLOUD_GEMINI_API_KEY)


# # TODO DEL
# class SogModelConfig:
#     REASONER_MODEL = "ollama_chat/mistral:7b-instruct"
#     FALLBACK_MODEL = None
#     # REASONER_MODEL = "ollama_chat/phi3:mini"
#     # FALLBACK_MODEL = "ollama_chat/gemma2:2b"
#     # REASONER_MODEL = "ollama_chat/deepseek-r1:1.5b"
#     EXTRACTOR__MODEL = "ollama_chat/deepseek-r1:1.5b"
#     # # EXTRACTOR__MODEL = "gemini/gemini-2.5-flash-lite"
#     # FALLBACK_MODEL = "ollama_chat/llama3.2:1b"
#     API_BASE = "http://localhost:11435"
#     API_KEY = "DUMMY_KEY"


# class SogModelConfig:
#     # TogetherAI Config
#     REASONER_MODEL = "together_ai/deepseek-ai/deepseek-v3-0324"
#     FALLBACK_MODEL = None
#     EXTRACTOR__MODEL = "together_ai/deepseek-ai/deepseek-v3-0324"
#     API_BASE = "https://api.fireworks.ai/inference/v1"
#     API_KEY = os.environ.get(TOGETHER_API_KEY)


class SogModelConfig:
    # OpenAI Config
    # REASONER_MODEL = "openai/gpt-5-mini"
    REASONER_MODEL = "openai/gpt-5.1"
    EXTRACTOR__MODEL = "openai/gpt-5-nano"
    FALLBACK_MODEL = "openai/gpt-5-mini"
    API_BASE = None
    API_KEY = os.environ.get("OPENAI_API_KEY")


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

    CONCEPT_CONFIG_PATH = "src/auto_lca/process/sog/config.json"

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
