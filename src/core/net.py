"""Shared HTTP behaviour for the web runners: retry, and error classification.

Two things went wrong in the same 372-protein run, and they are opposite kinds
of failure that were being handled identically:

* ArchCandy died after ~370 of 372 jobs, and WALTZ after ~120 of 372, both with
  ``SSL: UNEXPECTED_EOF_WHILE_READING`` — the remote closing a connection
  mid-handshake. That is **transient**: the next request usually succeeds. Left
  unretried it discards hours of completed work and, because the exception
  propagated out of the whole predictor, it took the surviving per-protein
  results with it.
* Cross-Beta-Pred refused because its form is CAPTCHA-gated. That is
  **permanent**: retrying cannot help and should not be attempted.

Treating both as "the predictor failed" is what turned a recoverable network
blip into `7 failed` and 368 `missing ..._waltz.csv` merge skips.

``classify`` separates them, and ``urlopen_retry`` retries only the transient
class, with exponential backoff and jitter so a panel running several
predictors concurrently does not synchronise its retries into a burst against
one already-struggling server.
"""

from __future__ import annotations

import random
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Literal

Kind = Literal["transient", "permanent", "unknown"]


class PermanentToolError(RuntimeError):
    """The service cannot be used this way at all; retrying is pointless.

    Raised for access gates (CAPTCHA), refusals of the request shape, and
    anything else where the same request will always be rejected.
    """


#: Substrings that mark a retryable network condition. Matched on the string
#: form because urllib wraps the interesting cause several layers down.
_TRANSIENT_MARKERS = (
    "unexpected_eof_while_reading",
    "remote end closed connection",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporary failure in name resolution",
    "timed out",
    "timeout",
    "bad gateway",
    "service unavailable",
    "gateway time-out",
    "eof occurred in violation of protocol",
)

#: HTTP statuses worth retrying. 429 included: it is the server asking for a
#: pause, which is exactly what backoff provides.
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def classify(error: BaseException) -> Kind:
    """Decide whether ``error`` is worth retrying."""
    if isinstance(error, PermanentToolError):
        return "permanent"
    if isinstance(error, urllib.error.HTTPError):
        return "transient" if error.code in _TRANSIENT_STATUS else "permanent"
    if isinstance(error, (ssl.SSLError, socket.timeout, TimeoutError, ConnectionError)):
        return "transient"
    text = str(error).lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    if isinstance(error, urllib.error.URLError):
        return "transient"
    return "unknown"


def retry_call(
    call: Callable[[], object],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
):
    """Run ``call``, retrying transient failures with jittered backoff.

    ``unknown`` is retried once. A wrapper cannot enumerate every way a dozen
    academic web servers fail, and one extra attempt costs a few seconds where
    giving up costs the whole run; a persistent unknown still surfaces.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            kind = classify(error)
            last = error
            if kind == "permanent":
                raise
            budget = attempts if kind == "transient" else min(attempts, 2)
            if attempt >= budget:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # Full jitter: concurrent predictors must not retry in lockstep
            # against a server that is already refusing connections.
            delay = random.uniform(0, delay)  # noqa: S311 - not cryptographic
            if on_retry is not None:
                on_retry(attempt, error, delay)
            time.sleep(delay)
    raise last  # pragma: no cover - loop always returns or raises


def urlopen_retry(request, *, timeout: float | None = None, **kwargs):
    """``urllib.request.urlopen`` with the retry policy above.

    Returns the response BODY, not the response object: a retried request
    cannot hand back a stream that was consumed on an earlier attempt, and
    every caller here reads the body immediately anyway.
    """
    def _call():
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read(), getattr(response, "status", 200), response.geturl()

    return retry_call(_call, **kwargs)


def _is_scheme_upgrade(old_url: str, new_url: str) -> bool:
    """True when a redirect only moves http to https, same host and path."""
    a = urllib.parse.urlsplit(old_url)
    b = urllib.parse.urlsplit(new_url)
    return (
        a.scheme == "http"
        and b.scheme == "https"
        and a.hostname == b.hostname
        and a.path == b.path
        and a.query == b.query
    )


class _MethodPreservingRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects that are required to repeat the request as it was.

    urllib does not follow 307 or 308 after a POST: its redirect_request accepts
    those codes only for GET and HEAD, and otherwise raises. Those two codes
    exist precisely to say "repeat this request, method and body included", so a
    service that has moved answers a form submission with HTTP Error 307 and the
    caller sees what looks like an outage. WALTZ fails exactly this way, and no
    amount of retrying helps, because nothing about it is transient.

    301 and 302 after a POST are turned into a GET by urllib. That is long
    settled browser behaviour and is left alone, with one exception: a bare
    scheme upgrade, where a server moves http to https on the same host and
    path. The body is dropped there too, and the service then answers whatever
    an empty GET produces -- for PASTA, a policy page where a job URL was
    expected. Only that case is overridden. Changing 301 and 302 in general
    would alter how every runner behaves on redirects nobody has examined.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        method = req.get_method()
        repeat = code in (307, 308) or (
            code in (301, 302)
            and method not in ("GET", "HEAD")
            and _is_scheme_upgrade(req.full_url, newurl)
        )
        if not repeat:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        # Host and Content-Length belong to the old request; urllib recomputes
        # both. Carrying them over sends the new host the old one's values.
        carried = {
            key: value
            for key, value in req.header_items()
            if key.lower() not in ("host", "content-length")
        }
        return urllib.request.Request(
            newurl,
            data=req.data,
            headers=carried,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
            method=method,
        )


_METHOD_PRESERVING_OPENER: urllib.request.OpenerDirector | None = None


def method_preserving_opener() -> urllib.request.OpenerDirector:
    """A shared opener that keeps POST bodies across the redirects above.

    Built once: build_opener installs the default handler set, and a new opener
    per request would also mean a new connection pool and a new cookie-free
    context each time.
    """
    global _METHOD_PRESERVING_OPENER
    if _METHOD_PRESERVING_OPENER is None:
        _METHOD_PRESERVING_OPENER = urllib.request.build_opener(
            _MethodPreservingRedirect()
        )
    return _METHOD_PRESERVING_OPENER


def urlopen_keeping_method(request, *, timeout: float | None = None):
    """urlopen, but a redirected POST stays a POST where the code demands it."""
    return method_preserving_opener().open(request, timeout=timeout)
