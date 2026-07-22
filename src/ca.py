"""Token-lattice CA: ring of N cells, each a vocab token.
Update rule = the trained windowed conditional (radius r), temperature T.
Modes: async (random-order single-site Glauber) and sync (all sites at once).
Common-random-number sampling supports damage-spreading twin runs."""
import json
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from model import CFG, center_logits, load

MASK, UNK = 0, 1

# --- context (switchable so the same automaton drives word-level or BPE models) ---
# Defaults reproduce the word-level pilot exactly. A BPE experiment sets e.g.
#   ca.DATA_DIR = "data_bpe"; ca.VOCAB = 4096; ca.INIT_LO = 1
# before calling run()/metrics() (byte-level BPE has no <unk>, so id 1 is a real
# token and random init should span [1, V)).
DATA_DIR = "data"
VOCAB = None      # None -> fall back to CFG["vocab"]
INIT_LO = 2       # lowest token id used for random init (skips <mask>,<unk>)

def _vocab():
    return VOCAB if VOCAB is not None else CFG["vocab"]

@partial(jax.jit, static_argnums=(3,))
def _site_probs(params, lattice, idx, w, T):
    """lattice (B,N); idx (w,) ring indices with center masked -> probs (B,V)."""
    win = lattice[:, idx]
    win = win.at[:, w // 2].set(MASK)
    logits = center_logits(params, win)
    logits = logits.at[:, MASK].set(-1e9)  # never emit <mask>
    return jax.nn.softmax(logits / T, axis=-1)

def _sample(probs, u):
    """Inverse-CDF sampling with external uniforms u (B,) -> tokens (B,)."""
    cdf = np.cumsum(np.asarray(probs), axis=-1)
    cdf /= cdf[:, -1:]
    return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))],
                    dtype=np.int32)

def run(params, B=8, N=48, r=2, T=1.0, sweeps=120, mode="async",
        init="random", seed=0, record_every=1, init_state=None, u_stream=None):
    """Returns dict with snapshots, activity, and per-sweep metrics."""
    rng = np.random.default_rng(seed)
    w = 2 * r + 1
    V = _vocab()
    if init_state is not None:
        lat = init_state.copy()
    elif init == "random":
        lat = rng.integers(INIT_LO, V, size=(B, N)).astype(np.int32)
    else:  # corpus slices
        ids = np.load(f"{DATA_DIR}/train_ids.npy")
        starts = rng.integers(0, len(ids) - N, size=B)
        lat = np.stack([ids[s:s + N] for s in starts]).astype(np.int32)

    snaps, activity = [lat.copy()], []
    n_up = sweeps * N * B
    if u_stream is None:
        u_stream = rng.random(n_up)  # one uniform per (sweep,site,lattice)
    ui = 0

    for t in range(sweeps):
        prev = lat.copy()
        if mode == "async":
            order = rng.permutation(N)
            for i in order:
                idx = (np.arange(i - r, i + r + 1) % N).astype(np.int32)
                probs = _site_probs(params, jnp.asarray(lat), jnp.asarray(idx), w, T)
                u = u_stream[ui:ui + B]; ui += B
                lat[:, i] = _sample(probs, u)
        else:  # sync: all sites from the same previous state
            newlat = lat.copy()
            for i in range(N):
                idx = (np.arange(i - r, i + r + 1) % N).astype(np.int32)
                probs = _site_probs(params, jnp.asarray(prev), jnp.asarray(idx), w, T)
                u = u_stream[ui:ui + B]; ui += B
                newlat[:, i] = _sample(probs, u)
            lat = newlat
        activity.append((lat != prev).mean(axis=1))  # per-lattice fraction changed
        if (t + 1) % record_every == 0:
            snaps.append(lat.copy())

    return dict(snaps=np.array(snaps), activity=np.array(activity),
                final=lat, params_r=r, T=T, mode=mode)

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
