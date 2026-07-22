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


def kgram_overlap(lat, ksets):
    """For each k, fraction of ring k-grams present in the reference set,
    averaged over lattices. Local (k=2) vs longer-range (k=3,4) corpus consistency."""
    B, N = lat.shape
    out = {}
    for k, S in ksets.items():
        vals = []
        for b in range(B):
            g = ring_kgrams(lat[b].tolist(), k)
            vals.append(np.mean([x in S for x in g]))
        out[k] = float(np.mean(vals))
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
