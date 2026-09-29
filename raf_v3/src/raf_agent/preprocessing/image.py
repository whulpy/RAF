from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps


class ImagePreprocessor:
    def __init__(self, output_dir: str | Path, *, size: tuple[int, int] = (224, 224), mode: str = "center_crop"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.mode = mode

    def process(self, raw_path: str | Path, content_sha256: str) -> Path:
        target = self.output_dir / content_sha256[:2] / f"{content_sha256}.jpg"
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if self.mode == "stretch":
                prepared = rgb.resize(self.size, Image.Resampling.LANCZOS)
            elif self.mode == "fit":
                prepared = ImageOps.contain(rgb, self.size, Image.Resampling.LANCZOS)
            else:
                prepared = ImageOps.fit(rgb, self.size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            temporary = target.with_suffix(".tmp.jpg")
            prepared.save(temporary, format="JPEG", quality=95)
            temporary.replace(target)
        return target

    @property
    def signature(self) -> str:
        value = f"image:{self.size[0]}x{self.size[1]}:{self.mode}:rgb"
        return hashlib.sha256(value.encode()).hexdigest()[:16]
