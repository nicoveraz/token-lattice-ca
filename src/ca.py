"""Token-lattice CA: ring of N cells, each a vocab token.
Update rule = the trained windowed conditional (radius r), temperature T.
Modes: async (random-order single-site Glauber) and sync (all sites at once).
Common-random-number sampling supports damage-spreading twin runs."""
import contextlib
import json
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from model import CFG, center_logits, load
from lattice import run as _lattice_run
from sampling import inverse_cdf

MASK, UNK = 0, 1

# --- context (switchable so the same automaton drives word-level or BPE models) ---
# Defaults reproduce the word-level pilot exactly. A BPE experiment needs
#   data_dir="data_bpe", vocab=4096, init_lo=1
# because byte-level BPE has no <unk>, so id 1 is a real token and random init should span [1, V).
#
# SET THESE THROUGH `using()`, NOT BY ASSIGNMENT (#25). They were plain module globals that
# experiments assigned to directly -- `ca.DATA_DIR = "data_bpe"` -- and nothing ever put them back.
# Two experiments in one process therefore silently corrupted each other: whichever ran second
# inherited the first's vocabulary and init floor, and the corruption is invisible because a wrong
# `init_lo` still produces a perfectly well-formed lattice, just drawn from the wrong support. The
# assignment form remains readable for compatibility (`damage.py` reads `ca.INIT_LO`), but the
# scoped form is the one that restores.
DATA_DIR = "data"
VOCAB = None      # None -> fall back to CFG["vocab"]
INIT_LO = 2       # lowest token id used for random init (skips <mask>,<unk>)


@contextlib.contextmanager
def using(*, data_dir=None, vocab=None, init_lo=None):
    """Scope the lattice context, restoring the previous values on exit.

    Restores in a `finally`, so an experiment that raises cannot leave the module configured for a
    corpus the next one is not reading. Nests correctly: each frame restores what it found rather
    than resetting to the defaults, so an inner BPE block inside an outer BPE block does not drop
    the caller back to word-level on exit.
    """
    global DATA_DIR, VOCAB, INIT_LO
    prev = (DATA_DIR, VOCAB, INIT_LO)
    if data_dir is not None:
        DATA_DIR = data_dir
    if vocab is not None:
        VOCAB = vocab
    if init_lo is not None:
        INIT_LO = init_lo
    try:
        yield
    finally:
        DATA_DIR, VOCAB, INIT_LO = prev


def _vocab():
    return VOCAB if VOCAB is not None else CFG["vocab"]

@partial(jax.jit, static_argnums=(3,))
def _site_probs(params, lattice, idx, w, T):
    """lattice (B,N); idx (w,) ring indices with center masked -> probs (B,V)."""
    return _win_probs(params, lattice[:, idx], w, T)


def _win_probs(params, win, w, T):
    """Same as `_site_probs` but taking the ALREADY-SLICED window (B,w).

    Split out so the unified loop in `lattice.run` can hand the rule a window rather than
    the whole lattice. Integer gathering is exact, so slicing in numpy before the call is
    bit-identical to slicing inside jax (verified by tests/test_golden.py).
    """
    win = jnp.asarray(win).at[:, w // 2].set(MASK)
    logits = center_logits(params, win)
    logits = logits.at[:, MASK].set(-1e9)  # never emit <mask>
    return jax.nn.softmax(logits / T, axis=-1)

def _sample(probs, u):
    """Inverse-CDF sampling with external uniforms u (B,) -> tokens (B,).

    int32 is this caller's choice and is preserved deliberately; see sampling.inverse_cdf.
    """
    return inverse_cdf(probs, u, dtype=np.int32)

class ToyRule:
    """The toy JAX transformer as a `lattice.Rule` (symmetric masked-centre window)."""

    def __init__(self, params, init="random"):
        self.params, self.init = params, init

    def window(self, i, r, N):
        return (np.arange(i - r, i + r + 1) % N).astype(np.int32)

    def probs(self, win, T):
        return _win_probs(self.params, win, win.shape[1], T)

    def sample(self, probs, u):
        return _sample(probs, u)

    def random_lattice(self, rng, B, N):
        """Replicates the historical init order exactly (random ids, or corpus slices)."""
        if self.init == "random":
            return rng.integers(INIT_LO, _vocab(), size=(B, N)).astype(np.int32)
        ids = np.load(f"{DATA_DIR}/train_ids.npy")
        starts = rng.integers(0, len(ids) - N, size=B)
        return np.stack([ids[s:s + N] for s in starts]).astype(np.int32)


def run(params, B=8, N=48, r=2, T=1.0, sweeps=120, mode="async",
        init="random", seed=0, record_every=1, init_state=None, u_stream=None):
    """Returns dict with snapshots, activity, and per-sweep metrics.

    Thin shim over the unified loop (`lattice.run`); kept so existing experiment scripts
    keep working. init_state/u_stream are passed through untouched so the RNG is consumed
    in the historical order: init -> u_stream -> per-sweep permutations.
    """
    out = _lattice_run(ToyRule(params, init), B=B, N=N, r=r, T=T, sweeps=sweeps, mode=mode,
                       init=init, seed=seed, record_every=record_every,
                       init_state=init_state, u_stream=u_stream)
    out["params_r"] = out.pop("r")            # historical key name for this backend
    return out

# ---------- metrics ----------
_corpus_bi = {}
def corpus_bigrams():
    if DATA_DIR not in _corpus_bi:
        ids = np.load(f"{DATA_DIR}/train_ids.npy")
        _corpus_bi[DATA_DIR] = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
    return _corpus_bi[DATA_DIR]

def metrics(lat):
    """lat (B,N) -> dict of scalars averaged over lattices."""
    B, N = lat.shape
    bi = corpus_bigrams()
    ent, dist, biov = [], [], []
    for b in range(B):
        vals, cnts = np.unique(lat[b], return_counts=True)
        p = cnts / cnts.sum()
        ent.append(-(p * np.log2(p)).sum())
        dist.append(len(vals) / N)
        pairs = list(zip(lat[b, :-1].tolist(), lat[b, 1:].tolist())) + \
                [(int(lat[b, -1]), int(lat[b, 0]))]
        biov.append(np.mean([pr in bi for pr in pairs]))
    return dict(entropy=float(np.mean(ent)), distinct=float(np.mean(dist)),
                bigram_overlap=float(np.mean(biov)))

def decode(row, itos):
    return " ".join(itos[i] for i in row)
