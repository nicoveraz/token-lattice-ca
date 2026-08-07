"""Sub-alphabet token lattices: the shared construction for #105, #106, #107.

WHAT THIS IS. The ring CA, with `p(x_i | x_{i-2}, x_{i-1})` RENORMALISED over a small token
sub-alphabet instead of the full vocabulary. No new model, no training, no new corpus -- the only
change is the support the conditional is sampled from.

WHY A SHARED MODULE. The three issues differ only in which sub-alphabet they use and what they
measure on it; the construction is identical. Putting it in one place means the ordering test
(#105), the |V|=2 rung (#106) and the velocity rung (#107) are provably the same lattice, which is
the whole basis for comparing them. A per-script copy would make that a hope.

THE ORDERING POINT, WHICH IS THE REASON #105 EXISTS. `ar_ca.sample_device` is inverse-CDF against a
shared uniform: `(cdf < u).sum()`. That is a functional of the ORDER in which the alphabet is laid
out. For BPE tokens that order is an arbitrary artifact of merge frequency, and nothing about the
model or the lattice depends on it. F41 established this is the MONOTONE coupling, not the maximal
one, and that maximal is order-INVARIANT while monotone is not. So `order=` here is a first-class
parameter, and permuting it changes nothing about the model and everything about the coupling.

WHAT IS NOT HERE. Maximal coupling. It requires BOTH twins' distributions jointly, and this loop
samples one lattice at a time with twins batched as 2B -- so it cannot be expressed as a per-cell
`sampler(probs, u)`. That restructuring is #108's problem and `coupling_primary.py` is on it. At
|V|=2 it is not needed: F41 proves monotone IS maximal there, which is exactly what makes #106 a
rung rather than another measurement.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]

import numpy as np


def pick_tokens(tok, words):
    """Token ids for `words`, keeping only those that encode to EXACTLY one token.

    A word that splits into two tokens would put a multi-token unit on a single cell, which is a
    different construction. Silently dropping such words would shrink the alphabet without saying
    so -- the caller gets the survivors AND the rejects, and must decide.
    """
    keep, drop = [], []
    for w in words:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        (keep if len(ids) == 1 else drop).append((w, ids))
    return (np.array([i[0] for _, i in keep], dtype=np.int64),
            [w for w, _ in keep], [w for w, _ in drop])


def make_sampler(sub_ids, order=None):
    """Inverse-CDF over the sub-alphabet, laid out in `order`.

    `probs` arrives as a numpy (B, V) array over the FULL vocabulary (the adapter passes
    `as_torch=False` whenever a custom sampler is supplied). We select the sub-alphabet columns,
    renormalise, and run the same `(cdf < u).sum()` the production path runs -- but in the given
    order, which is the parameter #105 varies.

    Returns real token ids, so the lattice state stays in the model's own vocabulary and every
    downstream tool (decode, metrics, damage) works unchanged.
    """
    sub_ids = np.asarray(sub_ids, dtype=np.int64)
    k = len(sub_ids)
    order = np.arange(k) if order is None else np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(k)):
        raise ValueError(f"order must be a permutation of range({k})")
    laid_out = sub_ids[order]                       # the alphabet as the CDF will walk it

    def sampler(probs, u):
        p = np.asarray(probs, dtype=np.float64)[:, sub_ids]      # (B, k)
        p = p[:, order]
        s = p.sum(axis=1, keepdims=True)
        # A window can make every sub-alphabet token vanish under float64. Falling back to uniform
        # keeps the chain defined and is recorded by the caller via `degenerate_frac`; silently
        # renormalising a zero row would divide by zero and poison the lattice.
        bad = (s[:, 0] <= 0)
        p = np.where(bad[:, None], 1.0 / k, p / np.where(s > 0, s, 1.0))
        cdf = np.cumsum(p, axis=1)
        cdf = cdf / cdf[:, -1:]
        idx = (cdf < np.asarray(u, dtype=np.float64)[:, None]).sum(axis=1)
        return laid_out[np.clip(idx, 0, k - 1)]

    sampler.sub_ids, sampler.order, sampler.k = sub_ids, order, k
    return sampler


def sub_init(sub_ids, B, N, rng):
    """A ring drawn from the sub-alphabet -- the lattice never leaves the restricted support."""
    return rng.choice(np.asarray(sub_ids, dtype=np.int64), size=(B, N)).astype(np.int64)


def damage_on_sub(rule, sub_ids, order, *, T, r, B, N, settle, sweeps, seed, block=3):
    """`ar_probe.block_damage`'s protocol, restricted to the sub-alphabet.

    Mirrors it exactly -- same settle, same block flip at the ring centre, same CRN twin batching
    with a shared uniform stream, same roll -- except that the sampler and the initial state are
    restricted. Returns the per-replica damage array so the caller can average or not, following
    F101's finding that averaging is the thing to decide rather than assume.
    """
    from ar_ca import run
    smp = make_sampler(sub_ids, order)
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=r, T=T, sweeps=settle, scheme="none",
               init_state=sub_init(sub_ids, B, N, rng), seed=seed, sampler=smp)["final"]
    c = N // 2
    idx = [c + k for k in range(-(block // 2), block - block // 2)]
    flipped = base.copy()
    for j in idx:
        flipped[:, j] = rng.choice(np.asarray(sub_ids, dtype=np.int64), size=B)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    u2 = np.concatenate([u.reshape(sweeps * N, B)] * 2, axis=1).reshape(-1)
    c2 = run(rule, B=2 * B, N=N, r=r, T=T, sweeps=sweeps, scheme="none",
             init_state=np.concatenate([base, flipped], axis=0), seed=seed + 2,
             u_stream=u2, sampler=smp)
    snaps = c2["snaps"]
    diff = (snaps[:, :B] != snaps[:, B:])
    rolled = np.roll(diff, N // 2 - idx[len(idx) // 2], axis=2)
    return base, rolled


def lambda_of(rolled, N):
    """lambda_ca and ignition from a per-replica damage array, using the project's own estimator."""
    from lyapunov import lyap_from_cone, is_unignited
    from dev_transition_phase3 import FIT_KW
    cone = rolled.mean(axis=1)
    md = float(rolled[-1].mean())
    lam = float(lyap_from_cone(cone, N, **FIT_KW)[0])
    ign = float(np.mean([not is_unignited(mean_damage=float(rolled[-1, b].mean()))
                         for b in range(rolled.shape[1])]))
    return dict(lambda_ca=round(lam, 5), mean_damage=md, ignition=round(ign, 4),
                ignited=bool(not is_unignited(mean_damage=md)))


# Sub-alphabets used by the three issues. Chosen to be plausible single tokens with a leading
# space, which is how BPE vocabularies store mid-sentence words; `pick_tokens` verifies and
# reports any that do not survive rather than quietly shrinking the alphabet.
COLOURS = [" red", " green", " blue", " yellow", " black", " white"]
BINARY = [" 0", " 1"]
DIGITS = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
