"""Mine for a positive cross-level signal: DEVELOPMENTAL design (issue #4 follow-up).

The cross-level proxy failed because black-box lambda_ca is kinematically model-INVARIANT
across trained sizes. But across TRAINING it should vary (a random-init model's conditional
is near-uniform; a trained one is sharp). So we sweep training checkpoints of a SINGLE model
and ask whether the two criticality levels CO-EVOLVE:
  * WHITE-BOX  lambda_top = (1/L) log rho(J_{emb->h_L})  at each checkpoint.
  * BLACK-BOX  lambda_ca (r=2, T=0.7) and D_norm at each checkpoint.
This is within-ONE-architecture (no family confound), across a natural axis (training) that
is not mechanically yoked to both measures the way temperature was. A positive, monotone
co-evolution would be genuine cross-level evidence.

Pre-registered: does white lambda_top track black lambda_ca across checkpoints (Pearson/
Spearman over ~10 steps)? Merges into results/mlm/crosslevel_dev.json. Resumable, MPS flush.
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
BASE = "EleutherAI/pythia-410m"
STEPS = ["step256", "step512", "step1000", "step2000", "step4000", "step8000",
         "step16000", "step32000", "step64000", "step143000"]
R, T_FIXED, B = 2, 0.7, 16
PROMPTS = [
    "The theory of dynamical systems studies how points in a state space evolve under "
    "iteration, and whether small perturbations grow or decay over time.",
    "In a crowded market the merchant weighed the copper coins carefully before handing "
    "the traveler a loaf of warm bread and a cup of water.",
]
OUT = f"{RESDIR}/crosslevel_dev.json"


@torch.no_grad()
def white_top(revision, n_dir=6, n_iter=10, rel_eps=1e-3, seed=21):
    tok = AutoTokenizer.from_pretrained(BASE)                 # tokenizer is revision-invariant
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=revision).eval().to(DEV, torch.float32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tops = []
    for text in PROMPTS:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        emb0 = model.get_input_embeddings()(ids)
        base = model(inputs_embeds=emb0, output_hidden_states=True).hidden_states
        L = len(base) - 1
        bf = base[-1]; eps = rel_eps * emb0.norm()
        for _ in range(n_dir):
            v = torch.randn(emb0.shape, generator=g).to(DEV, torch.float32); v = v / v.norm()
            rho = None
            for _ in range(n_iter):
                w = (model(inputs_embeds=emb0 + eps * v, output_hidden_states=True).hidden_states[-1] - bf) / eps
                rho = w.norm().item()
                if rho < 1e-20:
                    break
                v = (w / w.norm()).reshape(emb0.shape)
            if rho and rho > 0:
                tops.append(np.log(rho) / L)
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    return float(np.mean(tops)), float(np.std(tops) / max(1, len(tops) ** 0.5))


def black(revision):
    from ar_ca import ARRule
    from ar_probe import block_damage, drift_floor
    rule = ARRule(BASE, revision=revision)
    lam, dn = [], []
    for sd in (21, 22):
        d = block_damage(rule, T_FIXED, R, block=3, B=B, N=48, settle=12, sweeps=22, seed=sd, scheme="none")
        lam.append(lyap_from_cone(d["cone"], 48)[0])
        d0, _ = drift_floor(rule, T_FIXED, R, B=B, N=48, settle=12, sweeps=22, seed=sd, scheme="none")
        dn.append(d["mean_damage"] / max(d0, 1e-3))
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    return float(np.mean(lam)), float(np.mean(dn))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for step in STEPS:
        if step in res:
            print(f"[{step}] SKIP", flush=True); continue
        t0 = time.time()
        wt, wse = white_top(step)
        lam, dn = black(step)
        res[step] = dict(step=int(step.replace("step", "")), white_lambda_top=round(wt, 4),
                         white_se=round(wse, 4), black_lambda_ca=round(lam, 4),
                         black_D_norm=round(dn, 4), secs=round(time.time() - t0, 1))
        print(f"[{step}] white λ_top={wt:+.4f}  black λ_ca={lam:+.4f}  D_norm={dn:.4f}  ({res[step]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    steps = [s for s in STEPS if s in res]
    if len(steps) >= 4:
        wt = [res[s]["white_lambda_top"] for s in steps]
        lam = [res[s]["black_lambda_ca"] for s in steps]
        dn = [res[s]["black_D_norm"] for s in steps]
        stp = [res[s]["step"] for s in steps]
        out = {"model": BASE, "n": len(steps)}
        for nm, x in [("black_lambda_ca", lam), ("black_D_norm", dn)]:
            pr = stats.pearsonr(wt, x); sp = stats.spearmanr(wt, x)
            out[f"white_top_vs_{nm}"] = dict(pearson_r=round(float(pr[0]), 3), pearson_p=round(float(pr[1]), 4),
                                             spearman_rho=round(float(sp.correlation), 3), spearman_p=round(float(sp.pvalue), 4))
        for nm, x in [("white_lambda_top", wt), ("black_lambda_ca", lam), ("black_D_norm", dn)]:
            sp = stats.spearmanr(stp, x)
            out[f"{nm}_vs_step"] = dict(rho=round(float(sp.correlation), 3), p=round(float(sp.pvalue), 4))
        res["_dev"] = out
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\n=== DEVELOPMENTAL CROSS-LEVEL ({BASE}, n={len(steps)}) ===", flush=True)
        c = out["white_top_vs_black_lambda_ca"]
        print(f"  white λ_top vs black λ_ca:  Pearson r={c['pearson_r']} p={c['pearson_p']}  Spearman ρ={c['spearman_rho']} p={c['spearman_p']}", flush=True)
        c = out["white_top_vs_black_D_norm"]
        print(f"  white λ_top vs black D_norm: Pearson r={c['pearson_r']} p={c['pearson_p']}", flush=True)
        print(f"  vs training step:  white λ_top ρ={out['white_lambda_top_vs_step']['rho']}  "
              f"black λ_ca ρ={out['black_lambda_ca_vs_step']['rho']}  D_norm ρ={out['black_D_norm_vs_step']['rho']}", flush=True)
    print("CROSSLEVEL_DEV DONE", flush=True)


if __name__ == "__main__":
    main()
