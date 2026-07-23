"""Issue #15: de-confounded cross-level test on the RADIUS axis (control for E2).

E2 correlated white-box rho(F_T) vs black-box lambda_ca(T) ACROSS TEMPERATURE and found
a uniform ~-0.9 for every model/family -- a MECHANICAL confound (raise T: CA explores more
so lambda_ca up, softmax flattens so rho down), not cross-level signal. Here we hold T fixed
and sweep the RADIUS r instead: a rule knob not mechanically yoked to both measures the way
T is. Within each model, correlate:
  * WHITE-BOX  log rho(F_r): log top-singular-value of J[g] for an r-token window (T fixed).
  * BLACK-BOX  lambda_ca(r): token-space finite-size Lyapunov at radius r (T fixed).
If white and black track across r consistently across models/families, that is far harder to
dismiss as mechanical than the T-axis result. Pre-registered: is the r-axis relationship
model-specific (signal) or uniform-by-construction (still confounded)?

Merges into results/mlm/crosslevel_radius.json. Resumable, MPS flush, caffeinate-friendly.
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
from mlm_lib import RESDIR
from lyapunov import lyap_from_cone

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
R_GRID = [1, 2, 4, 8]
T_FIXED = 0.7
WINDOW_TEXT = ("the theory of dynamical systems studies how small perturbations grow or "
               "decay under iteration in a state space over time and whether the flow is "
               "chaotic or stable near its attractors and fixed points in practice today")
MODELS = [
    ("pythia-70m",  "EleutherAI/pythia-70m",  20),
    ("pythia-410m", "EleutherAI/pythia-410m", 16),
    ("pythia-1b",   "EleutherAI/pythia-1b",   12),
    ("gpt2",        "gpt2",                   20),
    ("gpt2-medium", "gpt2-medium",            16),
    ("gpt2-large",  "gpt2-large",             12),
]
OUT = f"{RESDIR}/crosslevel_radius.json"


def white_rho_radius(name, n_win=8, n_iter=8, rel_eps=1e-3, seed=0):
    """log top-singular-value of J[g_r] at fixed T, for each window radius r."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    Win = model.get_input_embeddings().weight
    emb = model.get_input_embeddings()
    ids = tok(WINDOW_TEXT, return_tensors="pt").input_ids[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def g_r(E):
        logits = model(inputs_embeds=E).logits[:, -1, :]
        p = torch.softmax(logits / T_FIXED, dim=-1)
        return p @ Win

    out = {}
    for r in R_GRID:
        starts = np.linspace(0, len(ids) - r - 1, n_win).astype(int)
        logsigs = []
        for s in starts:
            E0 = emb(ids[s:s + r].unsqueeze(0).to(DEV)).detach()      # (1,r,d)
            eps = rel_eps * E0.norm()
            with torch.no_grad():
                base = g_r(E0)
            v = torch.randn(E0.shape, generator=g).to(DEV, torch.float32); v = v / v.norm()
            sigma = None
            for _ in range(n_iter):
                with torch.no_grad():
                    a = (g_r(E0 + eps * v) - base) / eps
                na = a.norm()
                if na < 1e-20:
                    sigma = 0.0; break
                E1 = E0.detach().clone().requires_grad_(True)
                srr = (g_r(E1) * (a / na).detach()).sum()
                b = torch.autograd.grad(srr, E1)[0]
                sigma = float(na)
                nb = b.norm()
                if nb < 1e-20:
                    break
                v = (b / nb).detach()
            if sigma and sigma > 0:
                logsigs.append(np.log(sigma))
        out[r] = float(np.mean(logsigs))
        print(f"    [white] r={r}: log rho(F_r)={out[r]:+.4f}", flush=True)
    del model, Win, emb
    try: torch.mps.empty_cache()
    except Exception: pass
    return out


def black_lyap_radius(name, B):
    from ar_ca import ARRule
    from ar_probe import block_damage
    rule = ARRule(name)
    out = {}
    for r in R_GRID:
        ls = []
        for sd in (21, 22):
            d = block_damage(rule, T_FIXED, r, block=3, B=B, N=48, settle=12, sweeps=22, seed=sd, scheme="none")
            ls.append(lyap_from_cone(d["cone"], 48)[0])
        out[r] = float(np.mean(ls))
        print(f"    [black] r={r}: lambda_ca={out[r]:+.4f}", flush=True)
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, name, B in MODELS:
        if tag in res and "white_log_rho" in res[tag]:
            print(f"[{tag}] SKIP", flush=True); continue
        print(f"[{tag}] white rho(F_r) radius sweep (T={T_FIXED}) ...", flush=True)
        t0 = time.time()
        w = white_rho_radius(name)
        print(f"[{tag}] black lambda_ca(r) radius sweep ...", flush=True)
        b = black_lyap_radius(name, B)
        wv = [w[r] for r in R_GRID]; bv = [b[r] for r in R_GRID]
        pr = stats.pearsonr(wv, bv); sp = stats.spearmanr(wv, bv)
        res[tag] = dict(name=name, R_grid=R_GRID, T=T_FIXED,
                        white_log_rho={str(r): round(w[r], 4) for r in R_GRID},
                        black_lambda_ca={str(r): round(b[r], 4) for r in R_GRID},
                        within_pearson_r=round(float(pr[0]), 3), within_pearson_p=round(float(pr[1]), 4),
                        within_spearman_rho=round(float(sp.correlation), 3),
                        secs=round(time.time() - t0, 1))
        print(f"[{tag}] RADIUS-AXIS within-model: Pearson r={res[tag]['within_pearson_r']} "
              f"p={res[tag]['within_pearson_p']}  ({res[tag]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    rs = [res[t]["within_pearson_r"] for t, _, _ in MODELS if t in res and "white_log_rho" in res[t]]
    if len(rs) >= 2:
        res["_summary"] = dict(n_models=len(rs), per_model_r=rs,
                               median_r=round(float(np.median(rs)), 3),
                               n_positive=int(sum(r > 0 for r in rs)),
                               all_same_sign=bool(all(r > 0 for r in rs) or all(r < 0 for r in rs)))
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\n=== RADIUS-AXIS SUMMARY ({len(rs)} models) ===", flush=True)
        print(f"  per-model r: {rs}", flush=True)
        print(f"  median={res['_summary']['median_r']}  {res['_summary']['n_positive']}/{len(rs)} positive  "
              f"all-same-sign={res['_summary']['all_same_sign']}", flush=True)
        print("  (uniform across families => still mechanical; varied/model-specific => real signal)", flush=True)
    print("CROSSLEVEL_RADIUS DONE", flush=True)


if __name__ == "__main__":
    main()
