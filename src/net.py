"""
Every outbound request in this project goes through here. A request with
no timeout can hang a script indefinitely with no visible symptom (see
2026-09-04 postmortem: a background scan looked idle but had no timeout
on one of its requests.get calls).
"""
import time
import requests

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_with_retries(url, params=None, headers=None, timeout=30, retries=3, backoff=1.5):
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=merged_headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc
