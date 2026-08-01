"""Preregistration blocks with kill conditions, tamper detection, and quarantine.

Origin: textca F39/F45/F76. The pattern: write the outcome contract — hypotheses, frozen
parameters, kill conditions, even prediction bands — BEFORE the run; embed it in the results
file; evaluate kill conditions mechanically; and when an unregistered variant is computed
anyway (it always is), store it under quarantine keys where it cannot be quoted as a finding.
textca kept `_UNREGISTERED`/`_INFLATED` variants visible for audit rather than deleting them;
`quarantine` is that convention with a fence around it.

The self-hash makes post-hoc edits detectable: a prereg block whose hash no longer matches its
own content has been rewritten after the fact, which is the one thing a preregistration must
never survive.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

QUARANTINE_KEY = "_quarantine"
PREREG_KEY = "_preregistration"


def _canonical(d: Mapping) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class Preregistration:
    """An outcome contract. Freeze it before the data exist; embed the block in results.

    hypotheses:      named, directional statements ("H1": "lambda(post) > lambda(pre) ...").
    frozen:          every analysis parameter fixed in advance (fit windows, seed lists,
                     thresholds, the pre/post split). If it can move a verdict, freeze it.
    kill_conditions: named conditions under which the claim DIES ("K1": "delta peaks on the
                     degenerate pole"). Writing these first is what made textca's F76 null a
                     good result instead of a quiet non-mention.
    independent_unit: what the replicate actually is ("seed", "subject", "rule") — stating it
                     here is the cheap half of the F57 defense; `gatecheck.units` is the other.
    """

    name: str
    hypotheses: dict[str, str]
    frozen: dict[str, Any] = field(default_factory=dict)
    kill_conditions: dict[str, str] = field(default_factory=dict)
    independent_unit: str = ""
    note: str = ""

    def block(self) -> dict:
        body = {
            "name": self.name,
            "hypotheses": dict(self.hypotheses),
            "frozen": dict(self.frozen),
            "kill_conditions": dict(self.kill_conditions),
            "independent_unit": self.independent_unit,
            "note": self.note,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        body["sha256"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
        return body


def verify_block(block: Mapping) -> bool:
    """True iff the block's self-hash matches its content (i.e., not edited after freezing)."""
    body = {k: v for k, v in block.items() if k != "sha256"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest() == block.get("sha256")


def evaluate_kills(block: Mapping, predicates: dict[str, Callable[[], bool]]) -> dict:
    """Evaluate the registered kill conditions mechanically; refuse silent omissions.

    `predicates` maps kill-condition names to zero-argument callables returning True when the
    condition FIRED. Every registered condition must have a predicate — an unevaluated kill
    condition is a preregistration in name only.
    """
    registered = set(block.get("kill_conditions", {}))
    supplied = set(predicates)
    if registered - supplied:
        raise ValueError(f"kill conditions never evaluated: {sorted(registered - supplied)}")
    fired = {k: bool(predicates[k]()) for k in sorted(registered)}
    return {
        "fired": {k: v for k, v in fired.items() if v},
        "survived": not any(fired.values()),
        "evaluated": fired,
    }


def quarantine(results: dict, key: str, value: Any, reason: str) -> dict:
    """Store an unregistered or inflated variant where it can be audited but not quoted.

    Mutates and returns `results`. The variant lands under `_quarantine`, alongside the reason
    it does not count. Refuses to overwrite an existing quarantine entry silently — a variant
    that keeps changing is itself information.
    """
    q = results.setdefault(QUARANTINE_KEY, {})
    if key in q:
        raise KeyError(f"quarantine already holds {key!r}; use a new key, keep the history")
    q[key] = {"value": value, "reason": reason,
              "note": "unregistered/inflated variant kept for audit; not a finding"}
    return results


def assert_no_smuggling(results: Mapping, block: Mapping, *, allow: set[str] | None = None):
    """Every top-level results key must be registered, structural, or quarantined.

    The garden of forking paths grows at write time: computing one extra variant and reporting
    it alongside the registered one is how an exploratory number becomes a headline. Registered
    hypothesis names, `frozen` parameter names, underscore-prefixed structural keys, and an
    explicit `allow` set are legitimate; anything else raises.
    """
    allowed = set(block.get("hypotheses", {})) | set(block.get("frozen", {})) | (allow or set())
    offenders = [k for k in results
                 if not k.startswith("_") and k not in allowed]
    if offenders:
        raise AssertionError(
            f"unregistered top-level results keys {sorted(offenders)}: register them, "
            f"quarantine them, or pass them in `allow` with a reason in the code"
        )
