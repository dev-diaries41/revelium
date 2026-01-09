import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models
MINILM_MODEL_PATH = os.path.join(BASE_DIR, 'models/minilm_sentence_transformer_quant.onnx')

# DB
DB_DIR = os.path.join(BASE_DIR, "db")
# Config
SMARTSCAN_CONFIG_PATH = os.path.join(BASE_DIR, "smartscan.json")


