"""Mine for a positive (masked side): the one regime where black-box criticality genuinely
varies across models. The n=3 BERT cross-level was strong (white lambda_top vs masked D_norm
r=-0.96, rho=-1) but n=3 and depth-confounded. Extend to n=6 BERT depths (L=2..24, same
family/tokenizer) at a MATCHED cell (r=2, T=0.7) to test whether it survives, and record
depth so the depth-mediation is explicit.

  WHITE  lambda_top = (1/L) log rho(J_{emb->h_L})  (BertModel depth-Lyapunov)
  BLACK  D_norm = D / D0 at (r=2, T=0.7), 3 seeds  (masked capacity quantity)

Pre-registered: does the strong lambda_top vs D_norm relationship survive to n=6? Is it
distinguishable from a pure depth (1/L) effect? Resumable, MPS flush.
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
from transformers import AutoTokenizer, BertModel
from mlm_lib import RESDIR

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
R, T_FIXED, N, B = 2, 0.7, 48, 24
SEEDS = [21, 22, 23]
TEXT = ("the theory of dynamical systems studies how small perturbations grow or decay under "
        "iteration in a state space and whether the flow is chaotic or stable near its fixed points")
# same-family BERT depth ladder (all share bert-base-uncased WordPiece)
MODELS = [
    ("bert-tiny",   "prajjwal1/bert-tiny"),
    ("bert-mini",   "prajjwal1/bert-mini"),
    ("bert-small",  "prajjwal1/bert-small"),
    ("bert-medium", "prajjwal1/bert-medium"),
    ("bert-base",   "bert-base-uncased"),
    ("bert-large",  "bert-large-uncased"),
]
OUT = f"{RESDIR}/masked_ladder.json"
_TOK = AutoTokenizer.from_pretrained("bert-base-uncased")


@torch.no_grad()
def white_top(name, n_dir=6, rel=1e-3, seed=21):
    m = BertModel.from_pretrained(name).eval().to(DEV, torch.float32)
    ids = _TOK(TEXT, return_tensors="pt").to(DEV)
    emb = m.get_input_embeddings()(ids["input_ids"])
    am = ids["attention_mask"]
    base = m(inputs_embeds=emb, attention_mask=am, output_hidden_states=True).hidden_states
    L = len(base) - 1; bf = base[-1]; eps = rel * emb.norm()
    g = torch.Generator().manual_seed(seed); tops = []
    for _ in range(n_dir):
        v = torch.randn(emb.shape, generator=g).to(DEV, torch.float32); v = v / v.norm(); rho = None
        for _ in range(10):
            w = (m(inputs_embeds=emb + eps * v, attention_mask=am, output_hidden_states=True).hidden_states[-1] - bf) / eps
            rho = w.norm().item()
            if rho < 1e-20:
                break
            v = (w / w.norm()).reshape(emb.shape)
        if rho and rho > 0:
            tops.append(np.log(rho) / L)
    del m
    try: torch.mps.empty_cache()
    except Exception: pass
    return float(np.mean(tops)), L


def black_dnorm(name):
    from mlm_ca import MLMRule
    from mlm_damage import block_damage, drift_floor
    rule = MLMRule(name)
    vals = []
    for sd in SEEDS:
        d = block_damage(rule, T_FIXED, R, block=3, B=B, N=N, settle=12, sweeps=26, seed=sd, scheme="cls_sep")
        d0, _ = drift_floor(rule, T_FIXED, R, B=B, N=N, settle=12, sweeps=26, seed=sd, scheme="cls_sep")
        vals.append(d["mean_damage"] / max(d0, 1e-3))
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    return float(np.mean(vals)), float(np.std(vals) / len(vals) ** 0.5)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, name in MODELS:
        if tag in res:
            print(f"[{tag}] SKIP", flush=True); continue
        t0 = time.time()
        wt, L = white_top(name)
        dn, se = black_dnorm(name)
        res[tag] = dict(name=name, L=L, white_lambda_top=round(wt, 4),
                        black_D_norm=round(dn, 4), D_norm_se=round(se, 4), secs=round(time.time() - t0, 1))
        print(f"[{tag}] L={L:2d}  white λ_top={wt:+.4f}  black D_norm={dn:.4f}±{se:.4f}  ({res[tag]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    tags = [t for t, _ in MODELS if t in res]
    if len(tags) >= 4:
        wt = np.array([res[t]["white_lambda_top"] for t in tags])
        dn = np.array([res[t]["black_D_norm"] for t in tags])
        L = np.array([res[t]["L"] for t in tags], float)
        pr = stats.pearsonr(wt, dn); sp = stats.spearmanr(wt, dn)
        # depth-mediation check: partial correlation of white vs black controlling for log L
        def resid(y, x): return y - np.polyval(np.polyfit(x, y, 1), x)
        rr = stats.pearsonr(resid(wt, np.log(L)), resid(dn, np.log(L)))
        out = dict(n=len(tags), models=tags,
                   white_vs_black=dict(pearson_r=round(float(pr[0]), 3), pearson_p=round(float(pr[1]), 4),
                                       spearman_rho=round(float(sp.correlation), 3), spearman_p=round(float(sp.pvalue), 4)),
                   white_vs_logL=dict(r=round(float(stats.pearsonr(wt, np.log(L))[0]), 3)),
                   black_vs_logL=dict(r=round(float(stats.pearsonr(dn, np.log(L))[0]), 3)),
                   partial_white_black_given_logL=dict(r=round(float(rr[0]), 3), p=round(float(rr[1]), 4)))
        res["_summary"] = out
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\n=== MASKED LADDER ({len(tags)} BERT depths) ===", flush=True)
        print(f"  white λ_top vs black D_norm: Pearson r={out['white_vs_black']['pearson_r']} "
              f"p={out['white_vs_black']['pearson_p']}  Spearman ρ={out['white_vs_black']['spearman_rho']}", flush=True)
        print(f"  white vs log L: r={out['white_vs_logL']['r']}   black vs log L: r={out['black_vs_logL']['r']}", flush=True)
        print(f"  PARTIAL (controlling log depth): r={out['partial_white_black_given_logL']['r']} "
              f"p={out['partial_white_black_given_logL']['p']}  <- is it more than a depth effect?", flush=True)
    print("MASKED_LADDER DONE", flush=True)


if __name__ == "__main__":
    main()
