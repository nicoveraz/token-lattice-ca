"""New front (white-box substrate): the ACTIVATION-lattice light cone.

Apply the CA damage-spreading machinery to the residual stream instead of token space.
Perturb ONE token position's embedding under CRN twins (two forward passes differing only at
position p0; the null with eps=0 is exactly 0 by construction) and measure how the
perturbation propagates across POSITIONS and DEPTH:
    D_ell[p] = || h'_ell[p] - h_ell[p] || / ||eps||
This is a certified, model-SPECIFIC information-propagation cone (an empirical
Lieb-Robinson / effective-receptive-field map) -- driven by learned attention, not by the
depth/architecture that made the scalar white-box measure degenerate. fp32.

Outputs: (layers x distance) influence profile averaged over p0 and directions, a single-p0
(layers x position) heatmap, and summary metrics (cone width per layer; long-range mass).
Prototype: is there a real, model-specific signal? Usage: activation_cone.py [model]
"""
import sys, os, json, pathlib
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TEXT = ("The capital of France is Paris, and the capital of Japan is Tokyo. When John gave "
        "the book to Mary, she thanked him warmly and began to read it by the window.")


@torch.no_grad()
def damage_map(name, p0_list=(4, 8, 12), n_dir=8, rel=1e-3, seed=0):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV, torch.float32)
    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEV)
    S = ids.shape[1]
    emb0 = model.get_input_embeddings()(ids)                       # (1,S,d)
    base = model(inputs_embeds=emb0, output_hidden_states=True).hidden_states  # tuple L+1
    L = len(base) - 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    per_p0 = {}                                                    # p0 -> (L+1, S) mean damage
    for p0 in p0_list:
        acc = np.zeros((L + 1, S))
        for _ in range(n_dir):
            e = torch.randn(emb0.shape[2], generator=g).to(DEV, torch.float32)
            e = e / e.norm() * (rel * emb0[0, p0].norm())
            ep = emb0.clone(); ep[0, p0] = ep[0, p0] + e
            pert = model(inputs_embeds=ep, output_hidden_states=True).hidden_states
            for l in range(L + 1):
                acc[l] += ((pert[l][0] - base[l][0]).norm(dim=-1) / e.norm()).cpu().numpy()
        per_p0[p0] = acc / n_dir
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    return per_p0, L, S, [tok.decode(t) for t in ids[0]]


def analyze(name):
    per_p0, L, S, toks = damage_map(name)
    # distance-averaged profile: influence at (layer, distance=p-p0), causal so distance>=0
    maxd = S
    prof = np.zeros((L + 1, maxd)); cnt = np.zeros((L + 1, maxd))
    for p0, D in per_p0.items():
        for p in range(S):
            d = p - p0
            if d >= 0:
                prof[:, d] += D[:, p]; cnt[:, d] += 1
    prof = np.divide(prof, np.maximum(cnt, 1))
    # summary: effective cone width per layer (# positions with D > 0.1 * D at source), and
    # long-range mass (fraction of total downstream influence beyond distance 8)
    out = {"model": name, "L": L, "S": S}
    widths, lrmass = [], []
    for l in range(L + 1):
        row = prof[l]
        src = row[0] if row[0] > 1e-9 else max(row.max(), 1e-9)
        w = int((row > 0.1 * src).sum())
        far = row[8:].sum(); tot = row[1:].sum() + 1e-9
        widths.append(w); lrmass.append(round(float(far / tot), 3))
    out["cone_width_by_layer"] = widths
    out["longrange_massfrac_by_layer"] = lrmass
    # velocity: growth of cone width across depth (slope)
    out["cone_velocity"] = round(float(np.polyfit(np.arange(L + 1), widths, 1)[0]), 3)
    print(f"=== {name} (L={L}, S={S}) ===")
    print("cone width (D>0.1·src) by layer:", widths)
    print("long-range mass frac (>dist8) by layer:", lrmass)
    print("cone velocity (positions/layer):", out["cone_velocity"])
    # save a p0=8 heatmap + distance profile
    np.savez(str(ROOT / "results" / f"actcone_{name.split('/')[-1]}.npz"),
             heatmap=per_p0[8], profile=prof, widths=np.array(widths), tokens=np.array(toks, dtype=object))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
        im0 = ax[0].imshow(per_p0[8], aspect="auto", origin="lower", cmap="magma",
                           norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
        ax[0].axvline(8, color="cyan", lw=1, ls="--")
        ax[0].set_xlabel("token position (perturb p0=8, dashed)"); ax[0].set_ylabel("layer")
        ax[0].set_title(f"{name.split('/')[-1]}: influence of position 8\n(||Δh|| across positions × depth)")
        fig.colorbar(im0, ax=ax[0], shrink=0.8)
        im1 = ax[1].imshow(prof[:, :20], aspect="auto", origin="lower", cmap="magma",
                           norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
        ax[1].set_xlabel("distance downstream (p - p0)"); ax[1].set_ylabel("layer")
        ax[1].set_title("distance-averaged influence profile\n(the certified information cone)")
        fig.colorbar(im1, ax=ax[1], shrink=0.8)
        fig.tight_layout()
        fp = str(ROOT / "fig" / f"actcone_{name.split('/')[-1]}.png")
        fig.savefig(fp, bbox_inches="tight"); print("wrote", fp)
    except Exception as ex:
        print("plot skipped:", ex)
    json.dump(out, open(str(ROOT / "results" / f"actcone_{name.split('/')[-1]}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["EleutherAI/pythia-160m"]):
        analyze(m)
