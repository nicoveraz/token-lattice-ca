"""The single simulation loop, shared by every backend (Phase 1.2).

There used to be three near-identical copies of this loop (`ca.run`, `mlm_ca.run`,
`ar_ca.run`) that had already drifted apart: `ar_ca` had lost sync mode and `ca` had never
gained an external sampler. Nobody decided that; it was drift. Unifying them means one
`StubRule` null test covers all three backends BY CONSTRUCTION, which is the point of the
exercise -- the exact-zero CRN null is the guarantee every damage number depends on, and it
previously covered only the toy JAX path.

Behaviour contract (why the golden files must stay bit-identical):
  * RNG consumption order is `init (only if init_state is None)` -> `u_stream (only if
    u_stream is None)` -> `rng.permutation(N)` once per sweep in ASYNC mode only. Sync mode
    never draws a permutation. Any change here changes every downstream number.
  * Uniforms are consumed B at a time, in site-visit order, from a flat stream -- this is
    what makes twin runs sharing `u_stream` exactly coupled (CRN).
  * `lat[:, i] = ...` assigns in place, so the lattice dtype is set by `init_state` /
    `rule.random_lattice` and preserved regardless of what `sample` returns.

VISIT ORDER IS SHARED ACROSS THE BATCH BY DEFAULT, AND THAT IS A MEASUREMENT HAZARD (F57).
`rng.permutation(N)` draws ONE order per sweep and every replica in the batch follows it. For
bulk statistics that is harmless. For single-site damage spreading it is not: the AR rule is
causal-left, so damage at site j propagates only if j+1 or j+2 is visited BEFORE j; if j goes
first it resamples against an identical context with the same uniform, heals, and the run is
absorbed. That happens for 1/3 of orders -- and because the order is shared, it kills the whole
batch at once rather than a third of the replicas. 512 "replicas" then carry the statistical
weight of ONE draw of the thing that decides the outcome, and the seed-to-seed spread that
results was being read as physics. It is also the long-unexplained cause of F42's unignited runs.

`order="per_replica"` gives each replica its own permutation, so B replicas are B independent
orders. It is OPT-IN and the default is unchanged, because switching it would move every async
number in the repo; the scripts that need independence ask for it. As with `u_stream`, an
explicit `order_stream` can be supplied so twin runs share orders exactly -- CRN coupling
requires the twins to be updated in the SAME sequence, which independently drawn orders would
break.
"""
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Rule(Protocol):
    """What a backend must provide. `window` is what makes AR causal and MLM symmetric."""

    def window(self, i: int, r: int, N: int) -> np.ndarray:
        """Ring indices conditioning site i (MLM/toy: symmetric; AR: r cells to the left)."""

    def probs(self, win: np.ndarray, T: float) -> Any:
        """win (B, |window|) -> per-site distribution at temperature T."""

    def sample(self, probs: Any, u: np.ndarray) -> np.ndarray:
        """Inverse-CDF sample with EXTERNAL uniforms u (B,) -> tokens (B,)."""

    def random_lattice(self, rng, B: int, N: int) -> np.ndarray:
        """Initial (B, N) lattice of legal tokens."""


def symmetric_window(i: int, r: int, N: int) -> np.ndarray:
    """MLM / toy: the 2r+1 ring cells centred on i (centre is masked by the rule)."""
    return np.arange(i - r, i + r + 1) % N


def causal_window(i: int, r: int, N: int) -> np.ndarray:
    """AR: the r ring cells strictly to the LEFT of i."""
    return np.arange(i - r, i) % N


def run(rule: Rule, B: int = 16, N: int = 48, r: int = 2, T: float = 1.0, sweeps: int = 60,
        mode: str = "async", init: str = "random", seed: int = 0, record_every: int = 1,
        init_state=None, u_stream=None, order: str = "shared", order_stream=None, **extra):
    """Async/sync Glauber on a ring of N token cells driven by `rule`.

    Every knob exists exactly once here, so it works for every backend. `extra` is passed
    through to the rule via attributes already bound on it (e.g. the MLM special-token
    scheme), keeping the signature backend-agnostic.
    """
    if mode not in ("async", "sync"):
        raise ValueError(f"mode must be 'async' or 'sync', got {mode!r}")
    if order not in ("shared", "per_replica"):
        raise ValueError(f"order must be 'shared' or 'per_replica', got {order!r}")
    if order == "per_replica" and mode != "async":
        raise ValueError("order='per_replica' is meaningless in sync mode: every site is "
                         "updated from the same previous state, so there is no visit order")
    if order_stream is not None and order != "per_replica":
        raise ValueError("order_stream only applies to order='per_replica'")
    rng = np.random.default_rng(seed)
    lat = init_state.copy() if init_state is not None else rule.random_lattice(rng, B, N)
    snaps, activity = [lat.copy()], []
    if u_stream is None:
        u_stream = rng.random(sweeps * N * B)
    ui = 0

    def step(src, i):
        nonlocal ui
        idx = rule.window(i, r, N)
        u = u_stream[ui:ui + B]; ui += B
        return rule.sample(rule.probs(src[:, idx], T), u)

    def step_each(src, sites):
        """One update where replica b visits its OWN site sites[b] (F57, order='per_replica').

        The window is whatever `rule` defines, recovered as signed ring offsets from the rule
        itself rather than assumed -- so this stays correct for causal and symmetric windows
        alike, and for any future rule, instead of hard-coding one shape.
        """
        nonlocal ui
        idx = (sites[:, None] + _offsets[None, :]) % N
        u = u_stream[ui:ui + B]; ui += B
        return rule.sample(rule.probs(src[_rows, idx], T), u)

    if order == "per_replica":
        w0 = rule.window(0, r, N)
        _offsets = ((w0 + N // 2) % N) - N // 2       # signed ring offsets, wrap-safe
        _rows = np.arange(B)[:, None]

    for t in range(sweeps):
        prev = lat.copy()
        if mode == "async" and order == "per_replica":
            perm = (order_stream[t] if order_stream is not None
                    else np.argsort(rng.random((B, N)), axis=1))
            for k in range(N):
                sites = perm[:, k]
                lat[np.arange(B), sites] = step_each(lat, sites)
        elif mode == "async":
            for i in rng.permutation(N):        # random visit order (draws from rng)
                lat[:, i] = step(lat, i)        # in place: later sites see earlier updates
        else:
            newlat = lat.copy()
            for i in range(N):                  # sync: no permutation drawn
                newlat[:, i] = step(prev, i)    # all sites from the SAME previous state
            lat = newlat
        activity.append((lat != prev).mean(axis=1))
        if (t + 1) % record_every == 0:
            snaps.append(lat.copy())

    out = dict(snaps=np.array(snaps), activity=np.array(activity),
               final=lat, r=r, T=T, mode=mode)
    out.update(extra)
    return out
