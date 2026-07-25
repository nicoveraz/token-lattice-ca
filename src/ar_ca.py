"""Phase C: the token-lattice CA driven by an AUTOREGRESSIVE model (Pythia).

Same instrument, causal window: to resample cell i at radius r, feed the r cells
to its LEFT, x_{i-r..i-1}, to the AR model and take the next-token distribution at
the last position -- literally an order-r Markov approximation of the AR model.
This is the causal analog of the MLM's symmetric masked window (MLM = two-sided
window with a masked center; AR = one-sided left window predicting the next token).

If velocity∝r (F16/F21) and a finite repair/instability radius (F23) replicate
here, the phenomena are not artifacts of the MLM's globally-inconsistent
construction -- the key external-validity result (Phase C1).

Sampling is the same on-device inverse-CDF as mlm_ca (CRN, null test exact).
Apparatus (special-token) arm: prepend BOS (scheme="bos") vs nothing (scheme="none").
Pythia is a proper causal LM with a consistent joint, unlike the MLM -- state that.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lattice import run as _lattice_run


def pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ARRule:
    """Wraps an autoregressive LM as the causal CA rule p_r(x_i | x_{i-r..i-1})."""

    def __init__(self, model_name, device=None, fp16=True, revision=None):
        self.name = model_name if revision is None else f"{model_name}@{revision}"
        self.tok = AutoTokenizer.from_pretrained(model_name)   # tokenizer is revision-invariant
        self.device = device or pick_device()
        kw = {"revision": revision} if revision else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
        self.dtype = torch.float16 if (fp16 and self.device != "cpu") else torch.float32
        self.model = self.model.to(self.device, self.dtype)
        self.V = self.model.get_output_embeddings().weight.shape[0]
        self.BOS = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        forb = set(i for i in [self.tok.bos_token_id, self.tok.eos_token_id,
                               self.tok.pad_token_id, self.tok.unk_token_id] if i is not None)
        self.forbidden = np.array(sorted(forb), dtype=np.int64) if forb else np.array([], np.int64)
        self.init_pool = np.array([i for i in range(self.V) if i not in forb], dtype=np.int64)
        self._forbid_t = torch.tensor(self.forbidden, device=self.device, dtype=torch.long) \
            if len(self.forbidden) else None

    @torch.no_grad()
    def center_probs(self, win, T, scheme="none", as_torch=False):
        """win: (B, r) left-context ids -> next-token probs (B, V), specials forbidden."""
        win = np.asarray(win, dtype=np.int64)
        B = win.shape[0]
        if scheme == "bos":
            seq = np.concatenate([np.full((B, 1), self.BOS, np.int64), win], axis=1)
        elif scheme == "none":
            seq = win
        else:
            raise ValueError(scheme)
        ids = torch.from_numpy(seq).to(self.device)
        logits = self.model(input_ids=ids).logits[:, -1, :].float()
        if self._forbid_t is not None:
            logits[:, self._forbid_t] = -1e9
        probs = torch.softmax(logits / T, dim=-1)
        return probs if as_torch else probs.cpu().numpy()

    @torch.no_grad()
    def sample_device(self, probs_t, u):
        u_t = torch.as_tensor(u, device=self.device, dtype=probs_t.dtype).unsqueeze(1)
        cdf = probs_t.cumsum(-1)
        cdf = cdf / cdf[:, -1:]
        return (cdf < u_t).sum(dim=1).to("cpu", torch.int64).numpy()

    def random_lattice(self, rng, B, N):
        return rng.choice(self.init_pool, size=(B, N)).astype(np.int64)


class _ARAdapter:
    """Causal (left-window) counterpart of `_MLMAdapter`."""

    def __init__(self, rule, scheme, sampler):
        self.rule, self.scheme, self.sampler = rule, scheme, sampler
        self.dev = sampler is None

    def window(self, i, r, N):
        return np.arange(i - r, i) % N                   # r cells strictly to the LEFT

    def probs(self, win, T):
        return self.rule.center_probs(win, T, self.scheme, as_torch=self.dev)

    def sample(self, probs, u):
        return self.rule.sample_device(probs, u) if self.dev else self.sampler(probs, u)

    def random_lattice(self, rng, B, N):
        return self.rule.random_lattice(rng, B, N)


def run(rule, B=16, N=48, r=2, T=1.0, sweeps=60, scheme="none", init="random",
        seed=0, record_every=1, init_state=None, u_stream=None, sampler=None,
        mode="async"):
    """Causal ring CA: cell i resampled from p(x_i | x_{i-r..i-1}) (left window on the
    ring). Thin shim over the unified loop; `mode` now works here too (it was missing
    before the unification -- drift, not a decision)."""
    return _lattice_run(_ARAdapter(rule, scheme, sampler), B=B, N=N, r=r, T=T,
                        sweeps=sweeps, mode=mode, init=init, seed=seed,
                        record_every=record_every, init_state=init_state,
                        u_stream=u_stream, scheme=scheme)
