"""Shared helpers for the Phase-3 real-MLM probes."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import Counter
import numpy as np

MODELS = {                      # tag -> HF name
    "tiny": "prajjwal1/bert-tiny",
    "mini": "prajjwal1/bert-mini",
    "base": "bert-base-uncased",
}
RESDIR = "results/mlm"


def load_ref():
    return np.load("data_mlm/ref_ids.npy")


_REF_CACHE = {}
def ref_kgram_sets(k_max=4):
    """Return {k: set of k-grams present in the reference corpus} for k=2..k_max."""
    if k_max in _REF_CACHE:
        return _REF_CACHE[k_max]
    ids = load_ref().tolist()
    out = {}
    for k in range(2, k_max + 1):
        out[k] = set(tuple(ids[i:i + k]) for i in range(len(ids) - k + 1))
    _REF_CACHE[k_max] = out
    return out


def ref_trigram_counter():
    ids = load_ref().tolist()
    return Counter(tuple(ids[i:i + 3]) for i in range(len(ids) - 2))


def ring_kgrams(row, k):
    """k-grams of a ring (wraps around)."""
    n = len(row)
    ext = list(row) + list(row[:k - 1])
    return [tuple(ext[i:i + k]) for i in range(n)]


def repeat_collapse(row):
    """Collapse runs of identical tokens: [a,a,b,b,b,c] -> [a,b,c]. Removes the
    trivial 'my lord my lord' repetition that inflates k-gram overlap (A3)."""
    out = [row[0]]
    for t in row[1:]:
        if t != out[-1]:
            out.append(t)
    return out


def kgram_overlap(lat, ksets, collapse=False):
    """For each k, fraction of ring k-grams present in the reference set,
    averaged over lattices. Local (k=2) vs longer-range (k=3,4) corpus consistency.
    collapse=True first repeat-collapses each row (A3: repetition-robust variant)."""
    B, N = lat.shape
    rows = []
    for b in range(B):
        r = lat[b].tolist()
        rows.append(repeat_collapse(r) if collapse else r)
    out = {}
    for k, S in ksets.items():
        vals = []
        for r in rows:
            if len(r) < k:
                continue
            g = ring_kgrams(r, k)
            vals.append(np.mean([x in S for x in g]))
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def distinct_corpus_kgrams(lat, ksets):
    """Repetition-robust structure metric: count of DISTINCT lattice k-grams that
    appear in the corpus, normalized by N (per lattice, averaged). A lattice looping
    one corpus bigram 24x contributes 1, not 24 -- so periodic repetition ("my lord
    my lord") cannot inflate this the way raw position-weighted overlap does.
    Also returns the distinct-token fraction as a repetitiveness gauge."""
    B, N = lat.shape
    out = {}
    for k, S in ksets.items():
        vals = []
        for b in range(B):
            g = set(ring_kgrams(lat[b].tolist(), k))
            vals.append(len(g & S) / N)
        out[k] = float(np.mean(vals))
    out["distinct_tok"] = float(np.mean([len(np.unique(lat[b])) / N for b in range(B)]))
    return out


def order_param(lat, ref_bi):
    """Fraction of ring bigrams present in the reference corpus (the order param)."""
    B, N = lat.shape
    vals = []
    for b in range(B):
        pairs = list(zip(lat[b, :-1].tolist(), lat[b, 1:].tolist())) + \
                [(int(lat[b, -1]), int(lat[b, 0]))]
        vals.append(np.mean([pr in ref_bi for pr in pairs]))
    return float(np.mean(vals)), [float(np.mean([pr in ref_bi for pr in
        (list(zip(lat[b, :-1].tolist(), lat[b, 1:].tolist())) + [(int(lat[b, -1]), int(lat[b, 0]))])]))
        for b in range(B)]


def cone_front_velocity(cone, thresh=0.25):
    """Given a damage cone (sweeps+1, N) recentered on the flip, estimate the
    front half-width (sites with damage prob > thresh) per sweep, then the light-cone
    velocity = slope of half-width vs sweep over the early (ballistic) window."""
    S, N = cone.shape
    c = N // 2
    halfwidth = []
    for t in range(S):
        dmg = cone[t] > thresh
        idx = np.where(dmg)[0]
        if len(idx) == 0:
            halfwidth.append(0.0)
        else:
            halfwidth.append(float(max(abs(idx.max() - c), abs(c - idx.min()))))
    hw = np.array(halfwidth)
    # ballistic window: from sweep 1 until it saturates (reaches N/2) or 12 sweeps
    sat = np.argmax(hw >= (N // 2 - 1)) if (hw >= (N // 2 - 1)).any() else min(12, S - 1)
    end = max(2, min(sat if sat > 0 else 12, 12))
    v = float(np.polyfit(np.arange(end + 1), hw[:end + 1], 1)[0]) if end >= 2 else float("nan")
    return dict(halfwidth=hw.tolist(), velocity_sites_per_sweep=v, saturate_sweep=int(end))


def ensure_resdir():
    os.makedirs(RESDIR, exist_ok=True)


# ---------- coarse-grained long-range MI (A3, repetition-robust) ----------
_BUCKET = {}
def freq_buckets(nbuckets=16):
    """Map each token id -> a coarse frequency bucket (0=most frequent .. nbuckets-1
    =rarest/unseen), by corpus-frequency rank. Coarse-graining keeps the MI joint
    (nbuckets x nbuckets) well-sampled from a few thousand lattice sites."""
    if nbuckets in _BUCKET:
        return _BUCKET[nbuckets]
    ids = load_ref()
    from collections import Counter
    c = Counter(ids.tolist())
    ranked = [t for t, _ in c.most_common()]
    rank = {t: i for i, t in enumerate(ranked)}
    R = len(ranked)
    V = 30522
    bmap = np.full(V, nbuckets - 1, dtype=np.int32)     # unseen -> rarest bucket
    if R > 0:
        edges = np.linspace(0, np.log1p(R), nbuckets + 1)
        for t, r in rank.items():
            b = int(np.searchsorted(edges, np.log1p(r), side="right") - 1)
            bmap[t] = min(max(b, 0), nbuckets - 1)
    _BUCKET[nbuckets] = bmap
    return bmap


def _mi_plugin(a, b, K):
    """Plug-in MI (bits) of two integer arrays with alphabet size K."""
    joint = np.zeros((K, K))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = joint * (np.log2(joint) - np.log2(pa) - np.log2(pb))
    return float(np.nansum(m))


def coarse_mi_decay(lats, nbuckets=16, dmax=None, ring=True, seed=0):
    """lats: (M, N) int array of equilibrium lattices. Returns shuffle-debiased
    coarse MI(x_0 ; x_d) vs distance d (bits), and a decay length (first d where MI
    falls below half of MI at d=1). Debias: subtract MI of a globally shuffled pair
    set (the finite-sample bias floor for independent variables)."""
    bmap = freq_buckets(nbuckets)
    lats = np.asarray(lats)
    M, N = lats.shape
    bl = bmap[lats]                                     # (M,N) bucket field
    if dmax is None:
        dmax = N // 2
    rng = np.random.default_rng(seed)
    mis, mis_raw = [], []
    for d in range(1, dmax + 1):
        if ring:
            a = bl.reshape(-1)
            b = bmap[np.roll(lats, -d, axis=1)].reshape(-1)
        else:
            a = bl[:, :N - d].reshape(-1); b = bl[:, d:].reshape(-1)
        raw = _mi_plugin(a, b, nbuckets)
        sh = _mi_plugin(a, rng.permutation(b), nbuckets)
        mis_raw.append(raw)
        mis.append(max(0.0, raw - sh))
    mis = np.array(mis)
    half = mis[0] / 2 if mis[0] > 0 else 0.0
    below = np.where(mis < half)[0]
    length = int(below[0] + 1) if len(below) else dmax
    return dict(mi=mis.tolist(), mi_raw=mis_raw, decay_length=length,
                integrated=float(mis.sum()), mi_d1=float(mis[0]))
