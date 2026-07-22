"""Phase 3: the token-lattice CA driven by a real pretrained masked LM (BERT).

Same automaton as ca.py (ring of N token cells, async random-order Glauber, CRN
sampling for damage/differential twins), but the rule p_r(x_i | x_{i +/- r}) is a
HuggingFace MLM's masked-center distribution at temperature T instead of the toy
transformer's.

Apparatus choice (windowing): BERT expects [CLS] ... [SEP]. The default wraps the
2r+1 window in CLS/SEP and masks the center; special_scheme="none" is the
no-special-tokens variant used in the apparatus-invariance arm.

Sampling mirrors ca.py exactly: numpy uniforms + inverse-CDF, so coupled twin
runs sharing (init, order, uniforms, model) diverge by exactly zero — the null
test that certifies the CRN coupling carries over unchanged.

All three target models (prajjwal1/bert-tiny, prajjwal1/bert-mini,
bert-base-uncased) share the bert-base-uncased WordPiece vocab (30522), so one
tokenizer serves all. bert-tiny/mini configs lack model_type, so we load the
explicit BertForMaskedLM class rather than AutoModel.
"""
import os
import numpy as np
import torch
from transformers import AutoTokenizer, BertForMaskedLM

TOK_NAME = "bert-base-uncased"          # shared vocab for tiny/mini/base
_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(TOK_NAME)
    return _TOK


def pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MLMRule:
    """Wraps an MLM as the CA rule family p_r(center | window)."""

    def __init__(self, model_name, device=None, fp16=True):
        self.name = model_name
        self.tok = get_tokenizer()
        self.device = device or pick_device()
        self.model = BertForMaskedLM.from_pretrained(model_name).eval()
        # fp16 helps on MPS for the larger models; keep tiny models fp32 for stability
        self.dtype = torch.float16 if (fp16 and self.device != "cpu") else torch.float32
        self.model = self.model.to(self.device, self.dtype)
        self.cfg = self.model.config
        self.V = self.tok.vocab_size
        self.MASK = self.tok.mask_token_id
        self.CLS = self.tok.cls_token_id
        self.SEP = self.tok.sep_token_id
        # forbidden emission: all special tokens + [unusedX] placeholders
        forb = set(self.tok.all_special_ids)
        toks = self.tok.convert_ids_to_tokens(list(range(self.V)))
        for i, t in enumerate(toks):
            if t is None or t.startswith("[unused") or (t.startswith("[") and t.endswith("]")):
                forb.add(i)
        self.forbidden = np.array(sorted(forb), dtype=np.int64)
        self.init_pool = np.array([i for i in range(self.V) if i not in forb], dtype=np.int64)
        self._forbid_t = torch.tensor(self.forbidden, device=self.device, dtype=torch.long)

    # ---- the rule -----------------------------------------------------------
    @torch.no_grad()
    def center_probs(self, win, T, scheme="cls_sep", as_torch=False):
        """win: (B, w) int array (any center value); center is masked here.
        Returns probs (B, V), special/unused tokens forbidden. as_torch keeps it on
        the device (avoids a (B,V) MPS->CPU transfer per site — the hot-path win)."""
        B, w = win.shape
        win = np.asarray(win, dtype=np.int64).copy()
        win[:, w // 2] = self.MASK
        if scheme == "cls_sep":
            cls = np.full((B, 1), self.CLS, np.int64)
            sep = np.full((B, 1), self.SEP, np.int64)
            seq = np.concatenate([cls, win, sep], axis=1)
            cpos = 1 + w // 2
        elif scheme == "none":
            seq = win
            cpos = w // 2
        else:
            raise ValueError(scheme)
        ids = torch.from_numpy(seq).to(self.device)
        logits = self.model(input_ids=ids).logits[:, cpos, :].float()
        logits[:, self._forbid_t] = -1e9
        probs = torch.softmax(logits / T, dim=-1)
        return probs if as_torch else probs.cpu().numpy()

    @torch.no_grad()
    def sample_device(self, probs_t, u):
        """On-device inverse-CDF (CRN): probs_t (B,V) torch, u (B,) numpy -> tokens
        (B,) numpy. Deterministic given (probs_t, u), so the null test stays exact."""
        u_t = torch.as_tensor(u, device=self.device, dtype=probs_t.dtype).unsqueeze(1)
        cdf = probs_t.cumsum(-1)
        cdf = cdf / cdf[:, -1:]
        return (cdf < u_t).sum(dim=1).to("cpu", torch.int64).numpy()

    def random_lattice(self, rng, B, N):
        return rng.choice(self.init_pool, size=(B, N)).astype(np.int64)


def _sample(probs, u):
    """Inverse-CDF sampling with external uniforms u (B,) -> tokens (B,). CRN."""
    cdf = np.cumsum(probs, axis=-1)
    cdf /= cdf[:, -1:]
    return np.array([np.searchsorted(cdf[b], u[b]) for b in range(len(u))],
                    dtype=np.int64)


def run(rule, B=16, N=48, r=2, T=1.0, sweeps=60, mode="async", scheme="cls_sep",
        init="random", seed=0, record_every=1, init_state=None, u_stream=None,
        sampler=None):
    """Ring CA driven by `rule`. Mirrors ca.run (async/sync, CRN uniforms)."""
    rng = np.random.default_rng(seed)
    w = 2 * r + 1
    if init_state is not None:
        lat = init_state.copy()
    else:
        lat = rule.random_lattice(rng, B, N)
    dev = sampler is None                               # default path samples on-device
    snaps, activity = [lat.copy()], []
    if u_stream is None:
        u_stream = rng.random(sweeps * N * B)
    ui = 0

    def step(src, i):
        nonlocal ui
        idx = (np.arange(i - r, i + r + 1) % N)
        u = u_stream[ui:ui + B]; ui += B
        if dev:
            return rule.sample_device(rule.center_probs(src[:, idx], T, scheme, as_torch=True), u)
        return sampler(rule.center_probs(src[:, idx], T, scheme), u)

    for t in range(sweeps):
        prev = lat.copy()
        if mode == "async":
            for i in rng.permutation(N):
                lat[:, i] = step(lat, i)
        else:  # sync
            newlat = lat.copy()
            for i in range(N):
                newlat[:, i] = step(prev, i)
            lat = newlat
        activity.append((lat != prev).mean(axis=1))
        if (t + 1) % record_every == 0:
            snaps.append(lat.copy())
    return dict(snaps=np.array(snaps), activity=np.array(activity),
                final=lat, r=r, T=T, mode=mode, scheme=scheme)


# ---------- reference-corpus metrics (proxy validation) ----------
def ref_bigrams(ref_ids):
    return set(zip(ref_ids[:-1].tolist(), ref_ids[1:].tolist()))


def metrics(lat, ref_bi):
    """lat (B,N) -> scalars. Order parameter = fraction of ring bigrams present in
    the reference corpus (WikiText proxy). NOTE proxy: BERT's pretraining corpus is
    not WikiText, so this is a lower bound on 'is this English bigram structure'."""
    B, N = lat.shape
    ent, dist, biov = [], [], []
    for b in range(B):
        vals, cnts = np.unique(lat[b], return_counts=True)
        p = cnts / cnts.sum()
        ent.append(-(p * np.log2(p)).sum())
        dist.append(len(vals) / N)
        pairs = list(zip(lat[b, :-1].tolist(), lat[b, 1:].tolist())) + \
                [(int(lat[b, -1]), int(lat[b, 0]))]
        biov.append(np.mean([pr in ref_bi for pr in pairs]))
    return dict(entropy=float(np.mean(ent)), distinct=float(np.mean(dist)),
                bigram_overlap=float(np.mean(biov)))
