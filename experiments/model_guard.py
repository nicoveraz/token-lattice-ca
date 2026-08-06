"""Pre-flight: fail loudly BEFORE a run if a declared model cannot be obtained.

WHY THIS EXISTS. `gatecheck.cohort` catches a shrunken cohort at verdict time, which is the check
that matters for correctness. This catches it three hours earlier, which is the check that matters
for a laptop. A screen over twenty models that discovers on model seventeen that five are
unfetchable has already spent the compute; the same screen that refuses to start has spent nothing.

THE INCIDENT IT DESCENDS FROM. Five gated repositories (google/gemma-2-2b, meta-llama/Llama-3.2-3B,
google/codegemma-2b at gated=manual; sapienzanlp/Minerva-3B-base-v1.0, pfnet/plamo-3-nict-2b-base
at gated=auto) were evicted from a local cache on the reasoning that a Hugging Face cache is
reconstructible by definition. It is not: a gated repository requires per-account licence
acceptance, so eviction is irreversible without a human going to a web page. The next runs of
band_screen, band_family_census, tstar_second_target, argmax_census_hardened and
band_benchmark_range dropped those models and recomputed headline numbers over a smaller family
set -- 18 families where 22 had been registered.

Nothing in that chain was a bug. Each script caught the load failure, recorded it, and printed a
"loads failed" line. The failure was that a well-formed number came out the other end.

WHAT IT CHECKS. For each declared model: is it in the local cache, or reachable and ungated, or
reachable and gated with access already granted to this token? Anything else is a hard stop with
the reason named per model. Network is consulted only for models not already cached, and a network
failure is reported as unknown rather than as absent -- being offline is not evidence that a
repository is gone.
"""
import os
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))


class ModelsUnavailable(RuntimeError):
    """A declared model cannot be obtained, so the run would silently measure a smaller cohort."""


def _cached():
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        return set()


def audit_models(names, *, offline=False):
    """Per-model availability. Returns {name: (ok: bool, status: str)} without raising."""
    have = _cached()
    out = {}
    for n in names:
        if n in have:
            out[n] = (True, "cached")
            continue
        if offline:
            out[n] = (False, "not cached, and offline")
            continue
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(n)
            gated = getattr(info, "gated", False)
            if not gated:
                out[n] = (True, "reachable, ungated")
            else:
                # Reaching model_info on a gated repo means the token already has access; a token
                # without access raises before this point.
                out[n] = (True, f"reachable, gated={gated}, access granted")
        except Exception as e:
            kind = type(e).__name__
            if "Gated" in kind or "403" in str(e):
                out[n] = (False, f"GATED and access not granted to this token ({kind}) -- accept "
                                 f"the licence at https://huggingface.co/{n}")
            elif "RepositoryNotFound" in kind or "404" in str(e):
                out[n] = (False, f"repository not found ({kind})")
            else:
                # Offline or a transient network fault is NOT evidence the model is gone, and
                # treating it as such would block runs for the wrong reason.
                out[n] = (False, f"could not verify ({kind}) -- network, not necessarily absence")
    return out


def require_models(names, *, offline=False, tolerate=()):
    """Raise ModelsUnavailable unless every declared model can be obtained.

    `tolerate` names models whose absence was registered BEFORE the run. Pass it explicitly and
    write down why; a tolerated absence is a design decision and belongs in the preregistration,
    not in a shrug at runtime.
    """
    tol = set(tolerate)
    report = audit_models([n for n in names if n not in tol], offline=offline)
    bad = {n: why for n, (ok, why) in report.items() if not ok}
    if bad:
        lines = "\n  ".join(f"{n}: {why}" for n, why in sorted(bad.items()))
        raise ModelsUnavailable(
            f"{len(bad)} of {len(names)} declared models cannot be obtained, so this run would "
            f"measure a SMALLER COHORT than it registered and emit a well-formed number over a "
            f"denominator nobody chose:\n  {lines}\n"
            f"Restore them, or add them to `tolerate` and record the exclusion in the "
            f"preregistration before re-running.")
    return report
