import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_SYSTEM_PROMPT = "Your objective is to label prompt messages from clusters and label them, returning ClassificationResult. Labels should be one word max 3 words."
