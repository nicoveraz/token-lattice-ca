"""Skip a real-backend test when the WEIGHTS are unavailable -- and only then.

WHY THIS IS NARROW ON PURPOSE. CI runs with `HF_HUB_OFFLINE=1`, so the four tests that construct a
real MLM/AR rule cannot load weights and must skip rather than fail. The first version of that fix
caught `Exception`, which is too wide in a way that matters here: two of the four are the GOLDEN
tests, the numerical-reproducibility regression that protects every stored result in this
repository. Under a blanket catch, a `TypeError` from a refactor of `ARRule.__init__`, a corrupt
cache, or any other genuine breakage would turn those tests from RED to SKIPPED -- and a suite that
goes green by skipping its golden tests reports the same thing whether the code is correct or the
backend is broken. That is a check that can only pass or abstain, which is not a check.

Measured rather than assumed: with `HF_HUB_OFFLINE=1` and no cache, transformers raises a plain
`OSError` ("We couldn't connect to 'https://huggingface.co' ... and couldn't find them in the
cached files"). So the availability class is I/O-shaped, and every logic error a refactor would
introduce -- TypeError, AttributeError, ValueError, KeyError, RuntimeError -- still fails loudly.

RESIDUAL, STATED. A truncated or corrupt local weight file also raises `OSError` and would still
skip. Distinguishing "absent" from "corrupt" reliably would need a checksum of the cache, which is
more machinery than the risk warrants; the conftest skip-reason allowlist bounds it instead, by
making the skip visible and counted rather than silent.
"""
import pytest

_ERRORS = [OSError]                       # covers the offline case, measured above

try:                                      # hub versions differ on whether these subclass OSError
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
    _ERRORS += [LocalEntryNotFoundError, OfflineModeIsEnabled]
except Exception:                         # pragma: no cover - depends on installed hub version
    pass

try:
    from requests.exceptions import ConnectionError as _ReqConnErr, Timeout as _ReqTimeout
    _ERRORS += [_ReqConnErr, _ReqTimeout]
except Exception:                         # pragma: no cover
    pass

UNAVAILABLE = tuple(dict.fromkeys(_ERRORS))

# The prefix every availability skip must carry, so conftest can recognise it and count it.
SKIP_PREFIX = "backend unavailable"


def load_or_skip(loader, name):
    """Run `loader()`; skip only if the weights are unavailable, and re-raise anything else."""
    try:
        return loader()
    except UNAVAILABLE as e:
        pytest.skip(f"{SKIP_PREFIX}: {name}: {type(e).__name__}: {str(e)[:120]}")
