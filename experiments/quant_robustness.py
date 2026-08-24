"""Does the escape destination survive 8- and 4-bit weight quantization? Discharges an owed test.

Registered in experiments/prereg_quant_robustness.json (frozen `728d162b...` before any quantized
cell existed).

WHAT THIS DISCHARGES. prereg_selfcont.json registered quantization robustness as OWED AND NOT RUN,
noting that argmax is brittle at near-ties and that the margin field exists to support a threshold
rule nobody had tested. F185 tested that rule against BFLOAT16 and refused to call it a quantization
test. This is the quantization test: 4- and 8-bit are categorically larger perturbations than
bfloat16, and they are the ones deployment actually applies.

THE QUANTIZER, AND THE ASYMMETRY IT CREATES. Weight-only symmetric per-output-channel round-to-
nearest over every nn.Linear, applied in place to a freshly loaded float32 model -- no download, no
new dependency. RTN is the WEAKEST standard quantizer; GPTQ and AWQ use calibration data and
preserve behaviour better. So surviving RTN implies surviving those, while FAILING RTN does not
imply failing them. That asymmetry is registered in the prereg and scopes every negative result here.

Embeddings, LayerNorms and biases are NOT quantized, and the count of Linear layers that were is
stored per cell -- "we quantized the model" is not a measurement, the module list is.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from selfcont_set import BATCH, BATCH_BIG, BIG, out_path

PREREG = "experiments/prereg_quant_robustness.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "quant_robustness.json"
CACHE = _ROOT / "results" / "quant_robustness_cells.json"
CELLS = PR["cells"]
BITS = PR["quantizer"]["bit_widths"]
DECISIVE_FP32 = 0.6355          # arm 1's decisive-pair agreement; the registered comparison point


def fakequant_(model, bits):
    """Symmetric per-output-channel RTN, in place. Returns the count of quantized Linear layers."""
    q = 2 ** (bits - 1) - 1
    n = 0
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight.data
                s = torch.clamp(W.abs().amax(dim=1, keepdim=True) / q, min=1e-12)
                mod.weight.data = torch.round(W / s).clamp(-q - 1, q) * s
                n += 1
    return n


@torch.no_grad()
def top1(model, ids, dev, batch):
    out = np.empty(len(ids), np.int64)
    for i in range(0, len(ids), batch):
        ch = ids[i:i + batch]
        x = torch.tensor(ch, dtype=torch.long, device=dev).view(-1, 1).repeat(1, 2)
        out[i:i + len(ch)] = model(input_ids=x).logits[:, -1].float().argmax(-1).cpu().numpy()
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for m in CELLS:
        src = json.load(open(out_path(m, "fp32")))
        pid = np.array(src["probe_token_ids"])
        idx = np.flatnonzero(pid >= 0)
        ids = [int(i) for i in pid[idx]]
        batch = BATCH_BIG if m in BIG else BATCH
        for bits in BITS:
            key = f"{m}@int{bits}"
            if key in cache:
                print(f"  {key:<40} cached", flush=True); continue
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
            n_lin = fakequant_(model, bits)
            n_emb = sum(1 for x in model.modules() if isinstance(x, torch.nn.Embedding))
            a = top1(model, ids, "cpu", batch)
            dec = [tok.decode([int(i)]) for i in a]
            del model; gc.collect()
            cache[key] = dict(cell=key, model=m, bits=bits, probe_positions=[int(i) for i in idx],
                              source_ids=ids, top1_id=[int(i) for i in a], top1_str=dec,
                              n_linear_quantized=n_lin, n_embedding_untouched=n_emb,
                              secs=round(time.time() - t0, 1))
            json.dump(cache, open(CACHE, "w"))
            print(f"  {key:<40} lin={n_lin:<4} emb_untouched={n_emb} ({cache[key]['secs']:.0f}s)",
                  flush=True)
    _verdict(cache)


def _verdict(cache):
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_quant_robustness.sha256").read().split()[0],
               _quantizer=PR["quantizer"], decisive_pair_fp32_agreement=DECISIVE_FP32, cells={})
    for m in CELLS:
        src = json.load(open(out_path(m, "fp32")))
        pid = np.array(src["probe_token_ids"]); idx = np.flatnonzero(pid >= 0)
        marg = np.array(src["margins_e4"], np.int64)
        sent = src.get("_unmeasured_sentinel", -2147483648)
        base_amax = np.array(src["argmax_ids"], np.int64)[pid[idx]]
        base_bit = (marg[pid[idx]] > 0) & (marg[pid[idx]] != sent)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m)
        base_str = np.array([tok.decode([int(i)]) for i in base_amax], dtype=object)
        for bits in BITS:
            key = f"{m}@int{bits}"
            if key not in cache:
                continue
            c = cache[key]
            q_str = np.array(c["top1_str"], dtype=object)
            q_id = np.array(c["top1_id"], np.int64)
            esc = ~base_bit                       # sources that escape at full precision
            n = int(esc.sum())
            agree = float(np.mean(base_str[esc] == q_str[esc])) if n else None
            q_bit = q_id == np.array(c["source_ids"], np.int64)
            ham = int((q_bit != base_bit).sum())
            # A HAMMING COUNT ON A SPARSE SET CANNOT BE READ ALONE, and this arm is where that
            # would have gone wrong. pythia-410m has 8 self-continuing tokens in the intersection,
            # so "Hamming 8" at int4 is consistent with perfect robustness AND with the set being
            # annihilated. It is annihilation: 0 of 8 kept. The decomposition is therefore stored
            # beside the count, and the KEPT FRACTION is what any robustness claim must quote.
            kept = int((base_bit & q_bit).sum())
            gained = int((~base_bit & q_bit).sum())
            lost = int((base_bit & ~q_bit).sum())
            nb = int(base_bit.sum())
            cnt = collections.Counter(q_str[esc].tolist())
            res["cells"][key] = dict(
                model=m, bits=bits, n_escaping_sources=n,
                escape_agreement_vs_fp32=None if agree is None else round(agree, 4),
                feature_A_hamming_vs_fp32=ham, n_probe_sources=int(len(idx)),
                feature_A_base_set_size=nb, feature_A_quant_set_size=int(q_bit.sum()),
                feature_A_kept=kept, feature_A_gained=gained, feature_A_lost=lost,
                feature_A_kept_fraction=round(kept / nb, 4) if nb else None,
                _feature_A_reading="quote the KEPT FRACTION, never the Hamming count: on a sparse "
                                   "set a small Hamming is consistent with total destruction.",
                modal_destination=cnt.most_common(1)[0][0] if n else None,
                modal_share=round(cnt.most_common(1)[0][1] / n, 4) if n else None,
                n_linear_quantized=c["n_linear_quantized"],
                n_embedding_untouched=c["n_embedding_untouched"], secs=c["secs"])
    anchor = "EleutherAI/pythia-410m"
    q8 = res["cells"].get(f"{anchor}@int8", {}).get("escape_agreement_vs_fp32")
    q4 = res["cells"].get(f"{anchor}@int4", {}).get("escape_agreement_vs_fp32")
    res["KQ1_fires"] = bool(q8 is not None and q8 <= DECISIVE_FP32)
    res["KQ2_fires"] = bool(q4 is not None and q4 <= DECISIVE_FP32)

    p = [f"QUANTIZATION ROBUSTNESS, registered in {PREREG} (sha256 {res['_prereg_sha256'][:12]}..., "
         f"frozen before any quantized cell existed). This DISCHARGES the test prereg_selfcont left "
         f"OWED; bfloat16 did not, because 4- and 8-bit are categorically larger and are what "
         f"deployment applies. Weight-only symmetric per-channel RTN over nn.Linear only; "
         f"embeddings, norms and biases untouched, counts per cell in the file. "]
    for key, c in sorted(res["cells"].items(), key=lambda kv: (kv[1]["bits"], kv[1]["model"])):
        p.append(f"{key.split('/')[-1]}: escape agreement vs fp32 {c['escape_agreement_vs_fp32']} "
                 f"over {c['n_escaping_sources']} sources; feature-A keeps {c['feature_A_kept']} of "
                 f"{c['feature_A_base_set_size']} ({c['feature_A_kept_fraction']}), modal share "
                 f"{c['modal_share']}. ")
    p.append("FEATURE A IS REPORTED AS A KEPT FRACTION AND NOT AS A HAMMING COUNT, because this arm "
             "is where the count would have misled: pythia-410m has 8 self-continuing tokens in the "
             "intersection, so its int4 Hamming of 8 is consistent with perfect robustness and is in "
             "fact total loss -- 0 of 8 kept. A sparse set makes a small Hamming meaningless on its "
             "own, which is the same defect F183 found at r=0.913 and F185 avoided by pairing. ")
    p.append(f"THE REGISTERED COMPARISON POINT is arm 1's decisive pair at {DECISIVE_FP32}. ")
    for lbl, fired, v, b in (("KQ1", res["KQ1_fires"], q8, 8), ("KQ2", res["KQ2_fires"], q4, 4)):
        if v is None:
            p.append(f"{lbl} NOT EVALUABLE: the {b}-bit anchor cell is missing. ")
        elif fired:
            p.append(f"{lbl} FIRES: {b}-bit weight rounding of pythia-410m moves the escape "
                     f"destination to {v}, at or below the {DECISIVE_FP32} that separates the "
                     f"corpus manipulation. The fingerprint is SCOPED TO FULL PRECISION and may not "
                     f"be claimed for quantized deployment. ")
        else:
            p.append(f"{lbl} does not fire: {b}-bit leaves agreement at {v}, above the "
                     f"{DECISIVE_FP32} the corpus manipulation produces. ")
    p.append("KQ3, BINDING ON ANY NEGATIVE ABOVE: RTN is the WEAKEST standard quantizer. Surviving "
             "it implies surviving GPTQ or AWQ; failing it does NOT imply failing them, because "
             "those calibrate. A fired kill condition here scopes the claim, it does not establish "
             "that a real quantized deployment breaks the fingerprint. ")
    p.append("STILL OWED and not touched by this arm: activation quantization, real serving stacks, "
             "and any deployed quantized checkpoint. REFUSALS: no p-value on five cells; no "
             "adjustment of the bit widths or the comparison point; no semantic reading.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
