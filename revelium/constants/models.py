import os
from typing import Dict
from revelium.providers.types import LocalTextEmbeddingModel, ModelInfo
from pathlib import Path

BASE_DIR = Path.home() / ".cache" / "revelium"

# Local models
MINILM_MAX_TOKENS = 512
MINILM_MODEL_PATH = 'minilm_sentence_transformer_quant.onnx'
MINILM_MODEL_URL = "https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/minilm_sentence_transformer_quant.onnx"

MODEL_REGISTRY: Dict[LocalTextEmbeddingModel, ModelInfo] = {
    'all-minilm-l6-v2': ModelInfo(url=MINILM_MODEL_URL, path=MINILM_MODEL_PATH),
}
