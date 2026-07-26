"""Domany-Kinzel probabilistic cellular automaton -- the stochastic *discrete* ladder rung.

Why this rung exists. Every other rung of the validation ladder is the wrong regime: the
logistic map and the CML are smooth and infinitesimal (F30/F31/F37), the ECA rung is
discrete but deterministic. The DK model is the only rung that is stochastic AND discrete,
i.e. the same regime as the token instrument, and it has a *published* damage-spreading
boundary to check against.

    Domany & Kinzel, PRL 53, 311 (1984); Kinzel, Z. Phys. B 58, 229 (1985).
    P[1|0,0] = 0 ,  P[1|0,1] = P[1|1,0] = p1 ,  P[1|1,1] = p2

THE EXACT ANCHOR (Kohring & Schreckenberg, J. Phys. I France 2, 2033 (1992); extended by
Hinrichsen, Weitz & Domany, J. Stat. Phys. 88, 617 (1997)). On the p2 = 0 line,

    s'_i = (s_{i-1} XOR s_{i+1}) . theta(p1 - z_i)

so for two replicas driven by the SAME z_i the damage field d = s XOR t obeys

    d'_i = (d_{i-1} XOR d_{i+1}) . theta(p1 - z_i)

-- the damage field is *itself* a DK automaton at the same p1. That makes the whole CRN
damage machinery testable BIT-EXACTLY rather than to within a critical point's error bar,
which is what `tests/test_dk_damage_identity.py` does. The identity is special to p2 = 0;
off that line it must fail, and the test checks that too.

WHICH COUPLING IS CRN? Hinrichsen-Weitz-Domany parametrise the admissible replica couplings
by correlations alpha = <r01 r11>, beta = <r01 r10>. Drawing ONE uniform per site and
thresholding it against every probability gives alpha = min(p1,p2), beta = p1 -- their
"maximal correlation" (eq. 21). `lattice.run` draws one uniform per site per sweep, shared
by both twins, and samples by inverse CDF, so HERE, ON A BINARY ALPHABET, our CRN *is* that
member of the family.

    SCOPE -- THIS DOES NOT GENERALISE TO THE LANGUAGE-MODEL BACKENDS. Inverse-CDF sampling
    from a shared uniform is the MONOTONE (quantile) coupling. On |V|=2 the monotone and
    maximal couplings coincide, which is why the identity below is exact and why this rung
    is unaffected. On |V|>2 they come apart: p=(.5,.5,0), q=(0,.5,.5) gives maximal
    agreement 0.5 and quantile agreement 0. Since maximal coupling maximises agreement and
    so minimises damage, the LM damage numbers are NOT a lower bound over the admissible
    family -- an earlier claim to that effect had the inequality backwards and is retracted.
    Measured excess disagreement at the real operating point is small but nonzero:
    `experiments/coupling_gap.py` -> results/coupling_gap.json.

    The property inverse-CDF *does* have, and the reason to keep it, is REPLICA
    INDEPENDENCE: each replica's next state is a function of (its own state, the shared
    noise) alone, never of its twin. A maximal coupling is defined only pairwise -- it needs
    both p and q at construction and does not extend consistently to three replicas or to a
    self-consistent damage field.

Either way the underlying point stands, and it is published: damage spreading is a property
of (model, coupling), not of the model alone -- Grassberger, J. Stat. Phys. 79, 13 (1995).

UNIFORM CONVENTION. The literature writes s'=1 iff z < p; `lattice.run` samples by
inverse CDF over [1-p, p], which fires iff u > 1-p. These are the same rule under
z = 1-u: same marginal, and same coupling structure (both are a monotone threshold on one
shared uniform), so the maximal-correlation identification above is unaffected. The
convention is `u > 1-p` -- strict, matching `np.searchsorted(..., side='left')` -- so the
vectorised reference here is bit-identical to the `Rule` path, not merely equal in
distribution.

SUBLATTICES. DK lives on a diagonal lattice; site i at t+1 depends on i-1 and i+1 at t, so
on a plain ring with synchronous updates the even and odd sublattices never mix. That is
the standard reading of the model (the two sublattices are two independent DK automata
interleaved), and it is why N must be even -- an odd ring would join them through the
periodic seam and silently stop being the DK model.
"""
import numpy as np

from lattice import symmetric_window

# Published anchors. The p2=0 line genuinely disagrees across the literature; both values
# are kept rather than collapsing to whichever is convenient.
ANCHORS = {
    "site_dp":     dict(p1=0.705489, p2=0.705489, err=4e-6,
                        ref="Tretyakov & Inui, via Hinrichsen Adv. Phys. 49, 815 (2000) Tab. 1"),
    "bond_dp":     dict(p1=0.6447001, p2=0.8737620, err=1e-7,
                        ref="Jensen, via Hinrichsen Adv. Phys. 49, 815 (2000) Tab. 1"),
    "compact_dp":  dict(p1=0.5, p2=1.0, err=0.0, ref="exact (Kinzel 1983)"),
    "w18_zp":      dict(p1=0.801, p2=0.0, err=2e-3,
                        ref="Zebende & Penna, J. Stat. Phys. 74, 1273 (1994)"),
    "w18_hwd":     dict(p1=0.8087, p2=0.0, err=5e-4,
                        ref="Hinrichsen, Weitz & Domany, J. Stat. Phys. 88, 617 (1997)"),
}
DP_DELTA = 0.159464          # DP survival exponent, P(t) ~ t^-delta at criticality
DP_BETA = 0.276486           # DP density exponent


def dk_probs(left, right, p1, p2):
    """P[s'=1 | left, right] for the DK rule. Arrays in, array out."""
    both = (left == 1) & (right == 1)
    one = left != right
    return np.where(both, p2, np.where(one, p1, 0.0))


def dk_step(s, u, p1, p2):
    """One synchronous DK sweep. `s` (..., N) of 0/1, `u` (..., N) uniforms. Vectorised.

    Uses `u > 1-p`, matching the inverse-CDF path in `lattice.run` bit for bit (see the
    module docstring on the uniform convention).
    """
    left = np.roll(s, 1, axis=-1)
    right = np.roll(s, -1, axis=-1)
    p = dk_probs(left, right, p1, p2)
    return (u > 1.0 - p).astype(s.dtype)


def dk_run(s0, u, p1, p2, sweeps):
    """Reference simulator: `sweeps` synchronous DK steps from `s0` (B, N).

    `u` is the FLAT uniform stream `lattice.run` would consume, so that a run here and a
    run through the `Rule` path can be compared bit for bit. `lattice.run` consumes B
    uniforms per site, sites in order, sweeps outermost -- i.e. index t*N*B + i*B + b --
    which is `u.reshape(sweeps, N, B)` transposed to (sweeps, B, N).
    """
    B, N = s0.shape
    need = sweeps * N * B
    if u.size < need:
        raise ValueError(f"u_stream too short: need {need}, got {u.size}")
    us = u[:need].reshape(sweeps, N, B).transpose(0, 2, 1)
    s = s0.copy()
    snaps = [s.copy()]
    for t in range(sweeps):
        s = dk_step(s, us[t], p1, p2)
        snaps.append(s.copy())
    return np.array(snaps)


class DKRule:
    """DK as a `lattice.Rule`, so it runs through the SAME loop as every model backend.

    That is the point: the bit-exact damage identity below therefore tests the loop the
    language-model numbers are produced by, not a separate reimplementation of it.

    The window is the r=1 symmetric window `[i-1, i, i+1]`; the centre is ignored, since a
    DK site's next value depends only on its two neighbours. `T` is accepted and ignored --
    DK has no temperature, its parameters *are* the rule.
    """

    def __init__(self, p1, p2):
        if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
            raise ValueError(f"p1, p2 must be probabilities, got {p1}, {p2}")
        self.p1, self.p2 = float(p1), float(p2)

    def window(self, i, r, N):
        if r != 1:
            raise ValueError(f"DK is a two-input rule: r must be 1, got {r}")
        return symmetric_window(i, 1, N)

    def probs(self, win, T=None):
        w = np.asarray(win)
        p = dk_probs(w[:, 0], w[:, 2], self.p1, self.p2)
        return np.stack([1.0 - p, p], axis=1)

    def sample(self, probs, u):
        cdf = np.cumsum(probs, axis=-1)
        cdf = cdf / cdf[:, -1:]
        return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))],
                        dtype=np.int64)

    def random_lattice(self, rng, B, N):
        _require_even(N)
        return rng.integers(0, 2, size=(B, N)).astype(np.int64)


def _require_even(N):
    if N % 2:
        raise ValueError(f"N must be even: an odd ring joins DK's two sublattices "
                         f"through the periodic seam and is no longer the DK model (got {N})")


def seed_state(B, N, dtype=np.int64):
    """A single active site at the ring centre -- the standard DP seed initial condition."""
    _require_even(N)
    s = np.zeros((B, N), dtype=dtype)
    s[:, N // 2] = 1
    return s


def survival_from_seed(p1, p2, n_trials=1000, N=2048, steps=800, seed=0, chunk=250):
    """P(t): fraction of seed runs still active at time t. The DP order parameter.

    The cone from a single seed grows by one site per step, so `N > 2*steps` keeps it off
    the periodic boundary -- checked, not assumed. Trials run in chunks to bound memory.
    """
    if N <= 2 * steps:
        raise ValueError(f"cone would wrap: need N > 2*steps ({N} <= {2 * steps})")
    alive = np.zeros(steps + 1, dtype=np.int64)
    done = 0
    rng = np.random.default_rng(seed)
    while done < n_trials:
        b = min(chunk, n_trials - done)
        s = seed_state(b, N, dtype=np.int8)
        alive[0] += b
        live = np.ones(b, dtype=bool)
        for t in range(steps):
            s = dk_step(s, rng.random((b, N)), p1, p2)
            live &= s.any(axis=1)
            alive[t + 1] += int(live.sum())
            if not live.any():
                break
        done += b
    return alive / float(n_trials)


def damage_survival_from_seed(p1, p2, n_trials=1000, N=2048, steps=800, seed=0, chunk=250):
    """P_damage(t): fraction of CRN twin pairs whose damage is still alive at time t.

    Twins share one uniform per site per step (CRN; on this binary alphabet that is the
    maximal-correlation coupling -- see the scope note in the module docstring);
    they differ initially at exactly one site. This is the damage-spreading analogue of
    `survival_from_seed`, and on the p2=0 line the two must coincide by the exact identity
    in the module docstring.
    """
    if N <= 2 * steps:
        raise ValueError(f"cone would wrap: need N > 2*steps ({N} <= {2 * steps})")
    alive = np.zeros(steps + 1, dtype=np.int64)
    done = 0
    rng = np.random.default_rng(seed)
    while done < n_trials:
        b = min(chunk, n_trials - done)
        a = rng.integers(0, 2, size=(b, N)).astype(np.int8)
        c = a.copy()
        c[:, N // 2] ^= 1                      # single-site damage
        alive[0] += b
        live = np.ones(b, dtype=bool)
        for t in range(steps):
            u = rng.random((b, N))             # ONE stream, shared by both twins
            a = dk_step(a, u, p1, p2)
            c = dk_step(c, u, p1, p2)
            live &= (a != c).any(axis=1)
            alive[t + 1] += int(live.sum())
            if not live.any():
                break
        done += b
    return alive / float(n_trials)
