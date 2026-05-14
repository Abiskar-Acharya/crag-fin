"""Resolve URL shorteners to their final destination via a HEAD request."""

from __future__ import annotations

import requests


def follow_redirect(url: str, timeout: float = 5.0) -> str:
    """Return the final URL after redirects. On error, return the input unchanged.

    Uses HEAD with `allow_redirects=True` to avoid pulling page bodies.
    """
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
    except requests.RequestException:
        return url
    return resp.url or url
