from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

from ..models import ValidationResult


class ImageValidator:
    def __init__(
        self,
        *,
        min_width: int = 32,
        min_height: int = 32,
        max_pixels: int = 80_000_000,
        allowed_formats: set[str] | None = None,
        face_detector: Callable[[Path], bool] | None = None,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.max_pixels = max_pixels
        self.allowed_formats = {value.upper() for value in (allowed_formats or {"JPEG", "PNG", "WEBP", "BMP"})}
        self.face_detector = face_detector

    def validate(self, path: str | Path) -> ValidationResult:
        path = Path(path)
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_format = (image.format or "").upper()
                if image_format not in self.allowed_formats:
                    return ValidationResult(False, "unsupported_image_format", {"format": image_format})
                if width < self.min_width or height < self.min_height:
                    return ValidationResult(False, "image_too_small", {"width": width, "height": height})
                if width * height > self.max_pixels:
                    return ValidationResult(False, "image_too_large", {"width": width, "height": height})
                image.verify()
            if self.face_detector is not None and not self.face_detector(path):
                return ValidationResult(False, "face_not_detected")
            return ValidationResult(True, details={"width": width, "height": height, "format": image_format})
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ValidationResult(False, "unreadable_image", {"error": str(exc)})
