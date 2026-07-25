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
        init_state=None, u_stream=None, **extra):
    """Async/sync Glauber on a ring of N token cells driven by `rule`.

    Every knob exists exactly once here, so it works for every backend. `extra` is passed
    through to the rule via attributes already bound on it (e.g. the MLM special-token
    scheme), keeping the signature backend-agnostic.
    """
    if mode not in ("async", "sync"):
        raise ValueError(f"mode must be 'async' or 'sync', got {mode!r}")
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

    for t in range(sweeps):
        prev = lat.copy()
        if mode == "async":
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
