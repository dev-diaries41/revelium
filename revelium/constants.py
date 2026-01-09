import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINILM_MODEL_PATH = os.path.join(BASE_DIR, 'models/minilm_sentence_transformer_quant.onnx')
DB_DIR = os.path.join(BASE_DIR, "db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_SYSTEM_PROMPT = "Your objective is to label prompt messages from clusters and label them, returning ClassificationResult. Labels should be one word max 3 words."
DEFAULT_CHROMADB_PATH = os.path.join(DB_DIR, "revelium_chromadb")
DEFAULT_PROMPTS_PATH = os.path.join(DB_DIR, "prompts.db")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
MINILM_MAX_TOKENS = 512