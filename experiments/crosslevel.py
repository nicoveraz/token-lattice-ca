"""Issue #4 (flagship): cross-level validation.

Does the BLACK-BOX, weights-free token-space damping length track a WHITE-BOX
activation-space criticality measure across a real model ladder? If yes, the
instrument earns external significance: a cheap proxy for a property that
otherwise needs internals.

Two quantities per Pythia model:
  * WHITE-BOX  lambda_top = (1/L) log rho(J_{emb->h_L}): finite-depth top-Lyapunov
    exponent = depth-normalized log spectral radius of the input->final-hidden
    Jacobian, estimated by FINITE-DIFFERENCE POWER ITERATION (forward passes only,
    fp32). This is the SPARC error-propagation operator rho(F_T) (threshold >=1,
    i.e. lambda_top >= 0 = supercritical). Secondary: lambda_rand, mean per-layer
    log-expansion of a random perturbation (robustness check).
  * BLACK-BOX  D_norm at r=2, FIXED T=0.7 (no bimodal-T pooling; the audit's A6),
    5 seeds, +/- SE, via the same ar_probe block_damage / drift_floor as the paper.

Then Spearman(lambda_top, D_norm) across the ladder, reported honestly at n=6.
A null is a legitimate finding (the black-box instrument does NOT proxy white-box
criticality). Held to the seed-level standard the audit (A1) demands.

Incremental save + MPS cache flush (survives kills). Usage: crosslevel.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlm_lib import RESDIR, ensure_resdir

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
R, T_BB, SEEDS = 2, 0.7, [21, 22, 23, 24, 25]
N, B_BB = 48, 20
PROMPTS = [
    "The theory of dynamical systems studies how points in a state space evolve "
    "under iteration, and whether small perturbations grow or decay over time.",
    "In a crowded market the merchant weighed the copper coins carefully before "
    "handing the traveler a loaf of warm bread and a cup of water.",
    "Photosynthesis converts light energy into chemical energy, storing it in the "
    "bonds of glucose molecules that the plant later uses for growth and repair.",
]
# (tag, hf_name, size_M, B_blackbox)
LADDER = [
    ("pythia-14m",  "EleutherAI/pythia-14m",  14,   20),
    ("pythia-31m",  "EleutherAI/pythia-31m",  31,   20),
    ("pythia-70m",  "EleutherAI/pythia-70m",  70,   20),
    ("pythia-160m", "EleutherAI/pythia-160m", 160,  20),
    ("pythia-410m", "EleutherAI/pythia-410m", 410,  16),
    ("pythia-1b",   "EleutherAI/pythia-1b",   1000, 12),
    # second family (issue #5): GPT-2 small -> xl
    ("gpt2",        "gpt2",        124,  20),
    ("gpt2-medium", "gpt2-medium", 355,  16),
    ("gpt2-large",  "gpt2-large",  774,  12),
    ("gpt2-xl",     "gpt2-xl",     1558, 8),
]
OUT = None


# ---------------- white-box: finite-depth top-Lyapunov ----------------
def _final_hidden(model, emb):
    return model(inputs_embeds=emb, output_hidden_states=True).hidden_states[-1]


@torch.no_grad()
def white_box(name, n_iter=12, n_restart=2, rel_eps=1e-3, seed=0):
    """lambda_top (power-iteration spectral radius, depth-normalized) and
    lambda_rand (mean per-layer log-expansion of a random direction)."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tops, rands, Ls = [], [], []
    for text in PROMPTS:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        emb0 = model.get_input_embeddings()(ids)                 # (1,S,d)
        base_hs = model(inputs_embeds=emb0, output_hidden_states=True).hidden_states
        L = len(base_hs) - 1
        base = base_hs[-1]
        eps = rel_eps * emb0.norm()
        # --- top-Lyapunov via finite-difference power iteration ---
        for _ in range(n_restart):
            v = torch.randn(emb0.shape, generator=g).to(DEV, torch.float32)
            v = v / v.norm()
            rho = None
            for _ in range(n_iter):
                w = (_final_hidden(model, emb0 + eps * v) - base) / eps   # J v
                rho = w.norm().item()
                if rho < 1e-20:
                    break
                v = (w / w.norm()).reshape(emb0.shape)
            if rho and rho > 0:
                tops.append(np.log(rho) / L)
        # --- random-direction per-layer expansion (secondary) ---
        r = torch.randn(emb0.shape, generator=g).to(DEV, torch.float32)
        r = r / r.norm() * eps
        pert_hs = model(inputs_embeds=emb0 + r, output_hidden_states=True).hidden_states
        d = np.array([(pert_hs[l] - base_hs[l]).norm().item() for l in range(L + 1)])
        d = d / d[0]
        rands.append(np.polyfit(np.arange(L + 1), np.log(d + 1e-30), 1)[0])
        Ls.append(L)
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    return dict(lambda_top=float(np.mean(tops)), lambda_top_se=float(np.std(tops) / max(1, len(tops)**0.5)),
                lambda_rand=float(np.mean(rands)), lambda_rand_se=float(np.std(rands) / max(1, len(rands)**0.5)),
                L=int(np.mean(Ls)), n_top=len(tops))


# ---------------- black-box: damping length D_norm ----------------
def black_box(name, B):
    from ar_ca import ARRule
    from ar_probe import block_damage, drift_floor
    rule = ARRule(name)
    vals = []
    for sd in SEEDS:
        d = block_damage(rule, T_BB, R, block=3, B=B, N=N, settle=12, sweeps=26, seed=sd)
        d0, _ = drift_floor(rule, T_BB, R, B=B, N=N, settle=12, sweeps=26, seed=sd)
        vals.append(d["mean_damage"] / max(d0, 1e-3))
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    vals = np.array(vals)
    return dict(D_norm=float(vals.mean()), D_norm_se=float(vals.std() / len(vals)**0.5),
                seeds=SEEDS, vals=[round(float(v), 4) for v in vals])


def correlate(res):
    pts = [(res[t]["size_M"], res[t]["white"]["lambda_top"], res[t]["black"]["D_norm"])
           for t in res if isinstance(res[t], dict) and "white" in res[t]]
    if len(pts) < 3:
        return None
    size, wl, bl = zip(*pts)
    out = {"n": len(pts), "models": [t for t in res if isinstance(res[t], dict) and "white" in res[t]]}
    sp = stats.spearmanr(wl, bl)
    out["spearman_white_vs_black"] = dict(rho=round(float(sp.correlation), 3), p=round(float(sp.pvalue), 4))
    pr = stats.pearsonr(wl, bl)
    out["pearson_white_vs_black"] = dict(r=round(float(pr[0]), 3), p=round(float(pr[1]), 4))
    for lab, q in [("white_lambda_top", wl), ("black_D_norm", bl)]:
        s = stats.spearmanr(size, q)
        out[f"{lab}_vs_size"] = dict(rho=round(float(s.correlation), 3), p=round(float(s.pvalue), 4))
    return out


def main():
    global OUT
    ensure_resdir()
    OUT = f"{RESDIR}/crosslevel.json"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, name, sizeM, B in LADDER:
        if tag in res and "white" in res.get(tag, {}):
            print(f"[{tag}] SKIP (done)", flush=True); continue
        t0 = time.time()
        print(f"[{tag}] white-box ...", flush=True)
        w = white_box(name, seed=21)
        print(f"[{tag}] lambda_top={w['lambda_top']:+.4f}+/-{w['lambda_top_se']:.4f} "
              f"lambda_rand={w['lambda_rand']:+.4f} (L={w['L']})", flush=True)
        print(f"[{tag}] black-box (D_norm, r=2, T={T_BB}, {len(SEEDS)} seeds) ...", flush=True)
        b = black_box(name, B)
        print(f"[{tag}] D_norm={b['D_norm']:.4f}+/-{b['D_norm_se']:.4f}", flush=True)
        res[tag] = dict(size_M=sizeM, white=w, black=b, secs=round(time.time() - t0, 1))
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"[{tag}] done in {res[tag]['secs']}s", flush=True)
    corr = correlate(res)
    if corr:
        res["_correlation"] = corr
        json.dump(res, open(OUT, "w"), indent=1)
        print("\n=== CROSS-LEVEL CORRELATION (n=%d) ===" % corr["n"], flush=True)
        print("  white lambda_top vs black D_norm:  Spearman rho=%(rho)s p=%(p)s"
              % corr["spearman_white_vs_black"], flush=True)
        print("  (Pearson r=%(r)s p=%(p)s)" % corr["pearson_white_vs_black"], flush=True)
        print("  white_top vs size:  rho=%(rho)s p=%(p)s" % corr["white_lambda_top_vs_size"], flush=True)
        print("  black_Dnorm vs size: rho=%(rho)s p=%(p)s" % corr["black_D_norm_vs_size"], flush=True)
    print("CROSSLEVEL DONE", flush=True)


if __name__ == "__main__":
    main()
