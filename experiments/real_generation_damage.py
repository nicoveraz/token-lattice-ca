"""Does the instrument measure the MODEL, or only the apparatus? (the external-validity test)

Every metric in this project is measured on a ring CA built FROM a model -- a windowed,
in-place-resampled lattice whose stationary measure is not the model's generative
distribution (the MLM joint is globally inconsistent; the AR ring is a truncated causal
kernel). So the instrument's central assumption -- that its error-propagation numbers say
something about the model -- is ASSUMED, not demonstrated. The white-box version of this
question already returned a clean negative (F26/F28/F29).

This runs the same damage-spreading protocol on the thing we actually care about: **real
autoregressive generation**. Inject one wrong token mid-continuation and ask whether the
model's own dynamics absorb it or let it take over.

Why this is better posed than the failed cross-level test (F31): that one compared a
TANGENT/infinitesimal quantity (lambda_top, power-iterated) against a FINITE discrete one
(lambda_ca), and the logistic epsilon-sweep showed those regimes disagree even with known
ground truth. Here both sides are finite, discrete, single-token perturbations of an
autoregressive process -- same regime, same units.

Protocol (mirrors block_damage's discipline):
  * CRN: both twins share ONE uniform stream, so any divergence is causal, not sampling
    noise. The null arm (no injection) MUST diverge by exactly zero -- asserted, not assumed.
  * Injection: at position k, force the perturbed twin to emit a different token.
  * Readout: per-position token disagreement over the continuation.
      P_persist   = fraction of trials still diverged in the final tail (the DP-style
                    survival/order parameter, the analogue of P_ignite)
      reconverged = fraction that returned to exact agreement (error absorbed)
    Reported as a PAIR, per F34: survival probability and conditional spread, never a
    single mixture statistic.

Writes results/real_generation_damage.json. Usage:
  caffeinate -i .venv/bin/python experiments/real_generation_damage.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
PROMPTS = [
    "The capital of France is Paris, and the largest city in Japan is",
    "In the morning the baker opened the shop, lit the oven, and began to",
    "She picked up the letter, read it twice, and then quietly",
    "Scientists studying the deep ocean have recently discovered that",
]
T = 0.7
N_GEN = 48          # continuation length
INJECT_AT = 8       # inject after this many generated tokens
TAIL = 8            # final tokens used to decide "still diverged"
SEEDS = list(range(8))
OUT = str(_ROOT / "results" / "real_generation_damage.json")


@torch.no_grad()
def gen_twins(model, tok, prompt_ids, u, inject_at=None, forbid=None):
    """Two CRN twins generated autoregressively from the SAME uniform stream.

    Returns (tokens_a, tokens_b). With inject_at=None the twins are identical by
    construction -- that is the null arm. With inject_at=k, twin B is forced to emit a
    DIFFERENT token at step k (the injected error), then both continue freely.
    """
    a = prompt_ids.clone()
    b = prompt_ids.clone()
    out_a, out_b = [], []
    for t_step in range(N_GEN):
        pa = torch.softmax(model(input_ids=a).logits[0, -1, :].float() / T, dim=-1)
        pb = torch.softmax(model(input_ids=b).logits[0, -1, :].float() / T, dim=-1)
        if forbid is not None:
            pa[forbid] = 0; pb[forbid] = 0
        ua = float(u[t_step])
        ca = torch.cumsum(pa, 0); ca = ca / ca[-1]
        cb = torch.cumsum(pb, 0); cb = cb / cb[-1]
        ta = int((ca < ua).sum())                    # inverse-CDF, shared uniform (CRN)
        tb = int((cb < ua).sum())
        if inject_at is not None and t_step == inject_at:
            # force a DIFFERENT token for twin b: next-most-likely under its own dist
            top = torch.topk(pb, 2).indices.tolist()
            tb = top[1] if top[0] == ta else top[0]
        out_a.append(ta); out_b.append(tb)
        a = torch.cat([a, torch.tensor([[ta]], device=a.device)], dim=1)
        b = torch.cat([b, torch.tensor([[tb]], device=b.device)], dim=1)
    return np.array(out_a), np.array(out_b)


def run_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    forbid = torch.tensor([i for i in {tok.eos_token_id, tok.bos_token_id,
                                       tok.pad_token_id} if i is not None],
                          device=DEV, dtype=torch.long)
    forbid = forbid if len(forbid) else None
    persisted, reconverged, div_frac, null_max = [], [], [], 0.0
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        for sd in SEEDS:
            u = np.random.default_rng(1000 * pi + sd).random(N_GEN)
            # --- null arm: no injection must give EXACTLY zero divergence ---
            na, nb = gen_twins(model, tok, ids, u, inject_at=None, forbid=forbid)
            null_max = max(null_max, float((na != nb).mean()))
            # --- damage arm ---
            ga, gb = gen_twins(model, tok, ids, u, inject_at=INJECT_AT, forbid=forbid)
            post = slice(INJECT_AT, N_GEN)
            diff = (ga[post] != gb[post])
            persisted.append(bool(diff[-TAIL:].any()))
            reconverged.append(bool(not diff[-TAIL:].any()))
            div_frac.append(float(diff.mean()))
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return dict(P_persist=float(np.mean(persisted)),
                P_reconverge=float(np.mean(reconverged)),
                mean_divergence_frac=float(np.mean(div_frac)),
                divergence_frac_if_persisted=float(
                    np.mean([d for d, p in zip(div_frac, persisted) if p]) if any(persisted) else np.nan),
                n_trials=len(persisted), null_max_divergence=null_max)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in MODELS:
        tag = name.split("/")[-1]
        if tag in res:
            print(f"[{tag}] SKIP"); continue
        t0 = time.time()
        r = run_model(name)
        r["secs"] = round(time.time() - t0, 1)
        # the guarantee: without an injection, CRN twins must be identical
        assert r["null_max_divergence"] == 0.0, (
            f"{tag}: NULL ARM DIVERGED ({r['null_max_divergence']}) -- CRN broken in real "
            "generation; every number here would be meaningless. Fix before interpreting.")
        res[tag] = r
        print(f"[{tag}] P_persist={r['P_persist']:.3f}  P_reconverge={r['P_reconverge']:.3f}  "
              f"div_frac={r['mean_divergence_frac']:.3f}  null={r['null_max_divergence']}  "
              f"({r['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    res["_note"] = ("Damage spreading on REAL autoregressive generation (not the ring CA). "
                    "P_persist is the DP-style survival probability of a single injected "
                    "token error; reported as a pair with conditional divergence per F34. "
                    "The null arm (no injection, shared uniforms) is asserted to be exactly 0.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
