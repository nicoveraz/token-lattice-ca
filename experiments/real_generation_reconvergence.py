"""Distributional reconvergence: is the token-identity saturation an artifact of the metric?

`real_generation_damage.py` finds P_persist = 1.000 and P_reconverge = 0.000 in real
autoregressive generation: an injected token error is NEVER absorbed. But token identity is
a harsh test -- after an injection the two continuations are no longer positionally aligned,
so "different token at position i" need not mean the model has failed to recover. Two
continuations can be near-identical in distribution while sharing few tokens.

This measures recovery where it is actually well defined: the model's own NEXT-TOKEN
DISTRIBUTION. At each step after the injection we compute the total-variation distance
between the twins' distributions, TV(p_a, p_b). If the perturbation is absorbed in any
meaningful sense, TV decays.

Crucially it is reported against a FLOOR, in the spirit of the project's diversity floor
(F23) and with the W2 coupling caveat respected -- the floor uses the SAME coupling as the
signal (both are twin generations under CRN; they differ only in whether an error was
injected or the contexts were independently seeded):
    TV_norm = TV(twins) / TV(independent continuations of the same prompt)
  TV_norm -> 0  : distributions reconverge; the model absorbs the error.
  TV_norm -> 1  : the twins are as far apart as two unrelated continuations; no recovery.

Reported as a trajectory over post-injection steps plus the tail value, per model.
Writes results/real_generation_reconvergence.json. CPU-friendly.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from provenance import rel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from real_generation_damage import MODELS, PROMPTS, T, N_GEN, INJECT_AT, TAIL, SEEDS

DEV = "cpu" if os.environ.get("FORCE_CPU") else ("mps" if torch.backends.mps.is_available() else "cpu")
OUT = str(_ROOT / "results" / "real_generation_reconvergence.json")


def _probs(model, ids, forbid):
    p = torch.softmax(model(input_ids=ids).logits[0, -1, :].float() / T, dim=-1)
    if forbid is not None:
        p = p.clone(); p[forbid] = 0; p = p / p.sum()
    return p


def _sample(p, u):
    c = torch.cumsum(p, 0); c = c / c[-1]
    return int((c < float(u)).sum())


@torch.no_grad()
def tv_trajectory(model, ids, u, forbid, inject_at, ids_b=None, u_b=None):
    """Generate twins and record TV(p_a, p_b) at every step.

    inject_at=k  -> twin b forced to a different token at step k (the damage arm).
    ids_b/u_b    -> independent prompt-continuation for the FLOOR arm (different stream).
    """
    a = ids.clone()
    b = (ids_b if ids_b is not None else ids).clone()
    tvs = []
    for t_step in range(N_GEN):
        pa = _probs(model, a, forbid)
        pb = _probs(model, b, forbid)
        tvs.append(0.5 * float((pa - pb).abs().sum()))
        ta = _sample(pa, u[t_step])
        tb = _sample(pb, (u_b if u_b is not None else u)[t_step])
        if inject_at is not None and t_step == inject_at:
            top = torch.topk(pb, 2).indices.tolist()
            tb = top[1] if top[0] == ta else top[0]
        a = torch.cat([a, torch.tensor([[ta]], device=a.device)], dim=1)
        b = torch.cat([b, torch.tensor([[tb]], device=b.device)], dim=1)
    return np.array(tvs)


def run_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    forbid = torch.tensor([i for i in {tok.eos_token_id, tok.bos_token_id,
                                       tok.pad_token_id} if i is not None],
                          device=DEV, dtype=torch.long)
    forbid = forbid if len(forbid) else None
    dmg, flr = [], []
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        for sd in SEEDS:
            u = np.random.default_rng(1000 * pi + sd).random(N_GEN)
            ub = np.random.default_rng(50000 + 1000 * pi + sd).random(N_GEN)
            dmg.append(tv_trajectory(model, ids, u, forbid, inject_at=INJECT_AT))
            # floor: same prompt, INDEPENDENT stream, no injection -> unrelated continuations
            flr.append(tv_trajectory(model, ids, u, forbid, inject_at=None, u_b=ub))
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    dmg = np.stack(dmg); flr = np.stack(flr)
    post = slice(INJECT_AT + 1, N_GEN)                    # strictly after the injection
    d_traj = dmg[:, post].mean(axis=0)
    f_traj = flr[:, post].mean(axis=0)
    norm = d_traj / np.maximum(f_traj, 1e-9)
    return dict(tv_damage_traj=[round(float(x), 4) for x in d_traj],
                tv_floor_traj=[round(float(x), 4) for x in f_traj],
                tv_norm_traj=[round(float(x), 4) for x in norm],
                tv_norm_tail=float(np.mean(norm[-TAIL:])),
                tv_damage_tail=float(np.mean(d_traj[-TAIL:])),
                tv_floor_tail=float(np.mean(f_traj[-TAIL:])),
                n_trials=int(dmg.shape[0]))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in MODELS:
        tag = name.split("/")[-1]
        if tag in res:
            print(f"[{tag}] SKIP"); continue
        t0 = time.time()
        r = run_model(name); r["secs"] = round(time.time() - t0, 1)
        res[tag] = r
        print(f"[{tag}] TV_norm(tail)={r['tv_norm_tail']:.3f}  "
              f"TV_damage={r['tv_damage_tail']:.3f}  TV_floor={r['tv_floor_tail']:.3f}  "
              f"({r['secs']}s)", flush=True)
        print(f"         TV_norm trajectory: {r['tv_norm_traj'][:10]} ...", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    res["_note"] = ("Distributional reconvergence after a single injected token error in REAL "
                    "AR generation. TV_norm = TV(twins)/TV(independent continuations). ->0 "
                    "means the model absorbs the error in distribution; ->1 means the twins "
                    "are as far apart as unrelated continuations. Companion to "
                    "real_generation_damage.py, which finds token-identity recovery is 0.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", rel(OUT))


if __name__ == "__main__":
    main()
