import os
from typing import Dict
from revelium.providers.types import LocalTextEmbeddingModel, ModelInfo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db")

# Local models
MINILM_MAX_TOKENS = 512
MINILM_MODEL_PATH = 'minilm_sentence_transformer_quant.onnx'
MINILM_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/minilm_sentence_transformer_quant.onnx"

MODEL_REGISTRY: Dict[LocalTextEmbeddingModel, ModelInfo] = {
    'all-minilm-l6-v2': ModelInfo(url=MINILM_MODEL_URL, path=MINILM_MODEL_PATH),
}

# Providers
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_SYSTEM_PROMPT = "Your objective is to label prompt messages from clusters and label them, returning ClassificationResult. Labels should be one word max 3 words."
