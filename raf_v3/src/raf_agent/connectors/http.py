from __future__ import annotations

import random
import time
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from ..exceptions import SourceAuthenticationError, SourceUnavailableError


class RobustHttpClient:
    def __init__(self, *, timeout: float = 30, retries: int = 4, backoff: float = 0.5, min_interval: float = 0):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.min_interval = min_interval
        self._last_request = 0.0
        self.session = requests.Session()

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                self._last_request = time.monotonic()
                if response.status_code in (401, 403):
                    raise SourceAuthenticationError(f"authentication rejected by {url} ({response.status_code})")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt >= self.retries:
                        raise SourceUnavailableError(f"source unavailable after retries: {url} ({response.status_code})")
                    time.sleep(self._retry_delay(response, attempt))
                    continue
                response.raise_for_status()
                return response
            except SourceAuthenticationError:
                raise
            except (requests.RequestException, SourceUnavailableError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff * (2**attempt) + random.random() * 0.1)
        raise SourceUnavailableError(f"request failed after {self.retries + 1} attempts: {url}") from last_error

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                try:
                    return max(0.0, (parsedate_to_datetime(header).timestamp() - time.time()))
                except (TypeError, ValueError):
                    pass
        return self.backoff * (2**attempt) + random.random() * 0.1
