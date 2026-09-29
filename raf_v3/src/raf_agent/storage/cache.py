from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from ..connectors.http import RobustHttpClient
from ..exceptions import SampleDownloadError
from ..models import RawSample


@dataclass(slots=True)
class CachedContent:
    path: Path
    sha256: str
    content_type: str | None
    size_bytes: int


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SampleDownloadError(f"unsupported content URL: {url}")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or default_port)}
    except socket.gaierror as exc:
        raise SampleDownloadError(f"cannot resolve content host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise SampleDownloadError(f"content URL resolves to a non-public address: {parsed.hostname}")


class ContentCache:
    def __init__(self, root: str | Path, *, max_bytes: int = 30 * 1024 * 1024, http: RobustHttpClient | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.http = http or RobustHttpClient()

    def materialize(self, *, local_path: Path | None = None, url: str | None = None) -> CachedContent:
        if (local_path is None) == (url is None):
            raise SampleDownloadError("exactly one of local_path or url is required")
        if local_path is not None:
            return self._copy_local(local_path)
        return self._download(str(url))

    def download_or_cache(self, sample: RawSample) -> CachedContent:
        return self.materialize(local_path=sample.local_path, url=sample.content_url)

    def _copy_local(self, source: Path) -> CachedContent:
        source = source.resolve()
        if not source.is_file():
            raise SampleDownloadError(f"sample file does not exist: {source}")
        if source.stat().st_size > self.max_bytes:
            raise SampleDownloadError(f"sample exceeds byte limit: {source}")
        digest = self._hash_file(source)
        suffix = source.suffix.lower() or ".bin"
        target = self._target(digest, suffix)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return CachedContent(target, digest, mimetypes.guess_type(source.name)[0], source.stat().st_size)

    def _download(self, url: str) -> CachedContent:
        current_url = url
        response = None
        for _ in range(6):
            validate_public_url(current_url)
            response = self.http.request("GET", current_url, stream=True, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise SampleDownloadError(f"redirect without Location: {current_url}")
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise SampleDownloadError(f"too many redirects: {url}")
        final_url = current_url
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > self.max_bytes:
            raise SampleDownloadError(f"sample exceeds byte limit: {url}")
        suffix = Path(urlparse(final_url).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = mimetypes.guess_extension(response.headers.get("Content-Type", "").split(";")[0]) or ".bin"
        fd, temporary = tempfile.mkstemp(prefix="raf-download-", suffix=suffix, dir=self.root)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise SampleDownloadError(f"sample exceeds byte limit while downloading: {url}")
                    digest.update(chunk)
                    handle.write(chunk)
            target = self._target(digest.hexdigest(), suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                os.unlink(temporary)
            else:
                os.replace(temporary, target)
            return CachedContent(target, digest.hexdigest(), response.headers.get("Content-Type"), size)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _target(self, digest: str, suffix: str) -> Path:
        return self.root / digest[:2] / f"{digest}{suffix}"

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
