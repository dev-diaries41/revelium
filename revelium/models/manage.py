from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
from revelium.providers.types import LocalTextEmbeddingModel
from revelium.constants import MODEL_REGISTRY


class ModelManager:
    DEFAULT_MODEL_DIR = Path.home() / ".cache" / "revelium" / "models"
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or self.DEFAULT_MODEL_DIR).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, name: LocalTextEmbeddingModel, timeout: int = 30) -> Path:
        """
        Download a file from `url` into the manager's root_dir.
        - If filename is provided, use it; otherwise derive from the URL.
        - Writes to a temp file and atomically moves into place.
        - Returns the Path to the downloaded file.
        """

        model_info = MODEL_REGISTRY[name]
        target = self.get_model_path(name)
        print(target)
        if not str(target).startswith(str(self.root_dir)):
            raise ValueError("Resolved target path is outside the configured root_dir")

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
            raise ValueError("Refusing to delete outside of root_dir")

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
