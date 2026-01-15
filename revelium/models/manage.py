import shutil
import tempfile
import urllib.request
from typing import Optional
from pathlib import Path

from smartscan import TextEmbeddingProvider
from smartscan.providers import  MiniLmTextEmbedder

from revelium.providers.types import LocalTextEmbeddingModel
from revelium.constants import BASE_DIR
from revelium.constants.models import MODEL_REGISTRY
from revelium.providers.types import TextEmbeddingModel
from revelium.providers.embeddings.openai import OpenAITextEmbedder
from revelium.constants.models import MINILM_MAX_TOKENS
from revelium.errors import ReveliumError, ErrorCode


class ModelManager:
    DEFAULT_MODEL_DIR = BASE_DIR / "models"

    def __init__(self, root_dir: str = DEFAULT_MODEL_DIR):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, name: LocalTextEmbeddingModel, timeout: int = 30) -> Path:
        """
        Download a file from `url` into the manager's root_dir.
        - Writes to a temp file and atomically moves into place.
        - Returns the Path to the downloaded file.
        """

        target = self.get_model_path(name)
        if not str(target).startswith(str(self.root_dir)):
            raise ReveliumError("Target path is outside the configured root_dir", code=ErrorCode.INVALID_MODEL_PATH)

        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Stream download to a temp file then move atomically
        with tempfile.NamedTemporaryFile(delete=False, dir=str(self.root_dir)) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(self.get_model_download_url(name), timeout=timeout) as resp:
                chunk_size = 64 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    tmp.write(chunk)

        tmp_path.replace(target)
        return target

    def delete_model(self, name: LocalTextEmbeddingModel) -> None:
        """
        Delete a file or directory at `path`. `path` may be an absolute path or
        relative to the manager's root_dir. Safety: disallow deletion outside root_dir.
        """
        path = self.get_model_path(name)
        if not str(path).startswith(str(self.root_dir)):
            raise ReveliumError("Cannot delete file outside of root_dir", code=ErrorCode.INVALID_MODEL_PATH)

        if not path.exists():
            return

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def model_exists(self, name: LocalTextEmbeddingModel) -> bool:
        """
        Return True if a file or directory exists at `path`.
        `path` may be absolute or relative to root_dir.
        """
        resolved = self.get_model_path(name)
        return resolved.exists()
    
    def get_model_path(self, name: LocalTextEmbeddingModel) -> Path:
        model_info = MODEL_REGISTRY[name]
        return (self.root_dir / model_info['path']).resolve() if not Path(model_info['path']).is_absolute() else Path(model_info["path"]).resolve()

    def get_model_download_url(self, name: LocalTextEmbeddingModel) -> str:
        return MODEL_REGISTRY[name]['url']


    def get_text_embedder(self,model: TextEmbeddingModel, provider_api_key: Optional[str] = None) -> TextEmbeddingProvider:
        if model == ("text-embedding-3-large" or "text-embedding-3-small"):
            if provider_api_key is None:
                raise ReveliumError("Missing OpenAI API key", code=ErrorCode.MISSING_API_KEY)
            return OpenAITextEmbedder(provider_api_key, model=model)
        else:
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return MiniLmTextEmbedder(path, MINILM_MAX_TOKENS)
            path = self.get_model_path(model)
            return MiniLmTextEmbedder(path, MINILM_MAX_TOKENS)