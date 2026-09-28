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
