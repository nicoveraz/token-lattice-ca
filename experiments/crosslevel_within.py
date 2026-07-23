"""Issue #4, within-model design (the power boost over the n=6 cross-model test).

For each model we SWEEP TEMPERATURE and correlate, WITHIN that model:
  * WHITE-BOX  log rho(F_T): log top-singular-value of the Jacobian of the exact
    CA update's mean-field map  g_T(E) = sum_x p_T(x | E) * W_in[x]  -- the expected
    next-token INPUT embedding given the r-token window embeddings E, at temperature T.
    This is the analytic linearization of the SAME rule the black-box CA iterates
    (SPARC's rho(F_T)); estimated by power iteration on J^T J (finite-difference JVP +
    autograd VJP), averaged over real r-token windows, fp32.
  * BLACK-BOX  lambda_ca(T): token-space finite-size Lyapunov (damage-growth slope)
    at the same r, same T.
Sweeping T turns "n=6 cross-model scalars" into ~7 matched points PER model, immune
to the small-ladder / small-model-leverage problem. Positive within-model correlation,
consistent across models, is the strong form of the cross-level claim.

Merges into results/mlm/crosslevel_within.json. Resumable, MPS flush, caffeinate-friendly.
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
R = 2
T_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
WINDOW_TEXT = ("the theory of dynamical systems studies how small perturbations grow or "
               "decay under iteration in a state space over time and whether the flow is "
               "chaotic or stable near its attractors and fixed points in practice")
# (tag, hf_name, black-box B) -- both families, spanning sizes; xl skipped (backward cost)
MODELS = [
    ("pythia-70m",  "EleutherAI/pythia-70m",  20),
    ("pythia-410m", "EleutherAI/pythia-410m", 16),
    ("pythia-1b",   "EleutherAI/pythia-1b",   12),
    ("gpt2",        "gpt2",                   20),
    ("gpt2-medium", "gpt2-medium",            16),
    ("gpt2-large",  "gpt2-large",             12),
]
OUT = f"{RESDIR}/crosslevel_within.json"


def white_rho_sweep(name, n_win=8, n_iter=8, rel_eps=1e-3, seed=0):
    """log top-singular-value of J[g_T] at each T, averaged over real r-token windows."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    Win = model.get_input_embeddings().weight            # (V, d)
    emb = model.get_input_embeddings()
    ids = tok(WINDOW_TEXT, return_tensors="pt").input_ids[0]
    starts = np.linspace(0, len(ids) - R - 1, n_win).astype(int)
    windows = [emb(ids[s:s + R].unsqueeze(0).to(DEV)).detach() for s in starts]   # each (1,R,d)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def g_T(E, T):
        logits = model(inputs_embeds=E).logits[:, -1, :]  # (1,V)
        p = torch.softmax(logits / T, dim=-1)
        return p @ Win                                    # (1,d)

    out = {}
    for T in T_GRID:
        logsigs = []
        for E0 in windows:
            eps = rel_eps * E0.norm()
            with torch.no_grad():
                base = g_T(E0, T)
            v = torch.randn(E0.shape, generator=g).to(DEV, torch.float32)
            v = v / v.norm()
            sigma = None
            for _ in range(n_iter):
                with torch.no_grad():
                    a = (g_T(E0 + eps * v, T) - base) / eps          # J v  (1,d)
                na = a.norm()
                if na < 1e-20:
                    sigma = 0.0; break
                E1 = E0.detach().clone().requires_grad_(True)        # VJP: J^T a
                s = (g_T(E1, T) * (a / na).detach()).sum()
                b = torch.autograd.grad(s, E1)[0]                    # (1,R,d)
                sigma = float(na)                                    # ||J v|| -> sigma_max
                nb = b.norm()
                if nb < 1e-20:
                    break
                v = (b / nb).detach()
            if sigma and sigma > 0:
                logsigs.append(np.log(sigma))
        out[T] = float(np.mean(logsigs))
        print(f"    [white] T={T}: log rho(F_T)={out[T]:+.4f}", flush=True)
    del model, Win, emb
    try: torch.mps.empty_cache()
    except Exception: pass
    return out


def black_lyap_sweep(name, B):
    from ar_ca import ARRule
    from ar_probe import block_damage
    rule = ARRule(name)
    out = {}
    for T in T_GRID:
        ls = []
        for sd in (21, 22):
            d = block_damage(rule, T, R, block=3, B=B, N=48, settle=12, sweeps=22, seed=sd, scheme="none")
            ls.append(lyap_from_cone(d["cone"], 48)[0])
        out[T] = float(np.mean(ls))
        print(f"    [black] T={T}: lambda_ca={out[T]:+.4f}", flush=True)
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, name, B in MODELS:
        if tag in res and "white" in res[tag]:
            print(f"[{tag}] SKIP", flush=True); continue
        print(f"[{tag}] white-box rho(F_T) T-sweep ...", flush=True)
        t0 = time.time()
        w = white_rho_sweep(name)
        print(f"[{tag}] black-box lambda_ca(T) T-sweep ...", flush=True)
        b = black_lyap_sweep(name, B)
        wv = [w[T] for T in T_GRID]; bv = [b[T] for T in T_GRID]
        pr = stats.pearsonr(wv, bv); sp = stats.spearmanr(wv, bv)
        res[tag] = dict(name=name, T_grid=T_GRID,
                        white_log_rho={str(T): round(w[T], 4) for T in T_GRID},
                        black_lambda_ca={str(T): round(b[T], 4) for T in T_GRID},
                        within_pearson_r=round(float(pr[0]), 3), within_pearson_p=round(float(pr[1]), 4),
                        within_spearman_rho=round(float(sp.correlation), 3),
                        within_spearman_p=round(float(sp.pvalue), 4),
                        secs=round(time.time() - t0, 1))
        print(f"[{tag}] WITHIN-MODEL(T): Pearson r={res[tag]['within_pearson_r']} "
              f"p={res[tag]['within_pearson_p']}  Spearman rho={res[tag]['within_spearman_rho']} "
              f"p={res[tag]['within_spearman_p']}  ({res[tag]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    # pooled: stack all (standardized within model) points, and count consistent signs
    xs, ys, rs = [], [], []
    for tag, _, _ in MODELS:
        if tag not in res or "white" not in res[tag]:
            continue
        w = np.array([res[tag]["white_log_rho"][str(T)] for T in T_GRID])
        b = np.array([res[tag]["black_lambda_ca"][str(T)] for T in T_GRID])
        xs += list((w - w.mean()) / (w.std() + 1e-9)); ys += list((b - b.mean()) / (b.std() + 1e-9))
        rs.append(res[tag]["within_pearson_r"])
    if len(rs) >= 2:
        pr = stats.pearsonr(xs, ys)
        res["_pooled"] = dict(n_models=len(rs), per_model_r=rs,
                              median_within_r=round(float(np.median(rs)), 3),
                              n_positive=int(sum(r > 0 for r in rs)),
                              pooled_standardized_pearson_r=round(float(pr[0]), 3),
                              pooled_p=round(float(pr[1]), 6),
                              n_points=len(xs))
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\n=== WITHIN-MODEL POOLED ({len(rs)} models, {len(xs)} points) ===", flush=True)
        print(f"  per-model Pearson r: {rs}", flush=True)
        print(f"  {res['_pooled']['n_positive']}/{len(rs)} positive, median r={res['_pooled']['median_within_r']}", flush=True)
        print(f"  pooled standardized r={res['_pooled']['pooled_standardized_pearson_r']} "
              f"p={res['_pooled']['pooled_p']} (n={len(xs)})", flush=True)
    print("CROSSLEVEL_WITHIN DONE", flush=True)


if __name__ == "__main__":
    main()
