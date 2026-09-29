from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .base import EmbeddingProvider
from ..exceptions import ConfigurationError


class OpenCLIPEmbeddingProvider(EmbeddingProvider):
    """Lazy Hugging Face CLIP image encoder with normalized batch output."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", *, device: str | None = None, batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise ConfigurationError("CLIP requires the optional 'clip' dependencies") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._dimension = int(self._model.config.projection_dim)

    @property
    def dimension(self) -> int:
        self._load()
        return int(self._dimension)

    def encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        self._load()
        import torch

        results: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                vector = self._model.get_image_features(**inputs)
                if hasattr(vector, "pooler_output"):
                    vector = vector.pooler_output
                vector = torch.nn.functional.normalize(vector, dim=-1)
            results.append(vector.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(results) if results else np.empty((0, self.dimension), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode fixed semantic prompts in the same normalized CLIP space."""
        self._load()
        import torch

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = self._model.get_text_features(**inputs)
            if hasattr(vectors, "pooler_output"):
                vectors = vectors.pooler_output
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.detach().cpu().numpy().astype(np.float32)
