"""4-bit at the granularity deployment actually uses. Narrows F187's KQ3 gap; does not close it.

Registered in experiments/prereg_quant_grouped.json (frozen `953d354e...` before any cell).

WHAT F187 LEFT OPEN. It quantized with symmetric PER-OUTPUT-CHANNEL round-to-nearest -- one scale
per output row -- and 4-bit destroyed the escape map at agreement 0.0098. KQ3 registered the
asymmetry that scopes that negative: RTN is the weakest standard quantizer, so failing it does not
imply failing a calibrated one. But per-channel RTN is also the coarsest possible granularity, and
nothing deployed uses it: GPTQ, AWQ and bitsandbytes all quantize over GROUPS of input channels,
g=128 being the near-universal choice. So the 4-bit negative confounded granularity with calibration,
and this separates them.

WHAT IS STILL NOT DISCHARGED, and the prereg says so before any number: grouping is not calibration.
GPTQ compensates quantization error through a Hessian-based update and AWQ through activation-aware
scaling, and neither is implemented here. This narrows the owed item from "granularity and
calibration" to "calibration alone". Activation quantization and real serving stacks are untouched.
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
from quant_robustness import top1, DECISIVE_FP32

PREREG = "experiments/prereg_quant_grouped.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "quant_grouped.json"
CACHE = _ROOT / "results" / "quant_grouped_cells.json"
CELLS, BITS, G = PR["cells"], PR["bit_widths"], PR["group_size"]


def groupquant_(model, bits, group):
    """Symmetric round-to-nearest over contiguous groups of `group` input channels, in place.

    One scale per (output row, input group) instead of one per output row. That is the granularity
    GPTQ, AWQ and bitsandbytes all quantize at; it is NOT their calibration, which is the half this
    run does not implement.
    """
    q = 2 ** (bits - 1) - 1
    n = 0
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight.data
                out_f, in_f = W.shape
                pad = (-in_f) % group
                Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
                Wg = Wp.reshape(out_f, -1, group)
                s = torch.clamp(Wg.abs().amax(dim=2, keepdim=True) / q, min=1e-12)
                Wg = torch.round(Wg / s).clamp(-q - 1, q) * s
                Wp = Wg.reshape(out_f, -1)
                mod.weight.data = (Wp[:, :in_f] if pad else Wp).contiguous()
                n += 1
    return n


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for m in CELLS:
        pid = np.array(json.load(open(out_path(m, "fp32")))["probe_token_ids"])
        idx = np.flatnonzero(pid >= 0)
        ids = [int(i) for i in pid[idx]]
        for bits in BITS:
            key = f"{m}@int{bits}g{G}"
            if key in cache:
                print(f"  {key:<42} cached", flush=True); continue
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
            n_lin = groupquant_(model, bits, G)
            a = top1(model, ids, "cpu", BATCH_BIG if m in BIG else BATCH)
            dec = [tok.decode([int(i)]) for i in a]
            del model; gc.collect()
            cache[key] = dict(cell=key, model=m, bits=bits, group=G,
                              probe_positions=[int(i) for i in idx], source_ids=ids,
                              top1_id=[int(x) for x in a], top1_str=dec,
                              n_linear_quantized=n_lin, secs=round(time.time() - t0, 1))
            json.dump(cache, open(CACHE, "w"))
            print(f"  {key:<42} lin={n_lin:<4} ({cache[key]['secs']:.0f}s)", flush=True)
    _verdict(cache)


def _verdict(cache):
    from transformers import AutoTokenizer
    prev = json.load(open(_ROOT / "results" / "quant_robustness.json"))["cells"]
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_quant_grouped.sha256").read().split()[0],
               group_size=G, decisive_pair_fp32_agreement=DECISIVE_FP32, cells={})
    for m in CELLS:
        src = json.load(open(out_path(m, "fp32")))
        pid = np.array(src["probe_token_ids"]); idx = np.flatnonzero(pid >= 0)
        marg = np.array(src["margins_e4"], np.int64)
        sent = src.get("_unmeasured_sentinel", -2147483648)
        base_bit = (marg[pid[idx]] > 0) & (marg[pid[idx]] != sent)
        tok = AutoTokenizer.from_pretrained(m)
        base_str = np.array([tok.decode([int(i)]) for i in
                             np.array(src["argmax_ids"], np.int64)[pid[idx]]], dtype=object)
        for bits in BITS:
            key = f"{m}@int{bits}g{G}"
            if key not in cache:
                continue
            c = cache[key]
            q_str = np.array(c["top1_str"], dtype=object)
            q_bit = np.array(c["top1_id"], np.int64) == np.array(c["source_ids"], np.int64)
            esc = ~base_bit
            n = int(esc.sum())
            kept = int((base_bit & q_bit).sum()); nb = int(base_bit.sum())
            cnt = collections.Counter(q_str[esc].tolist())
            res["cells"][key] = dict(
                model=m, bits=bits, group=G, n_escaping_sources=n,
                escape_agreement_vs_fp32=round(float(np.mean(base_str[esc] == q_str[esc])), 4) if n else None,
                feature_A_kept=kept, feature_A_base_set_size=nb,
                feature_A_kept_fraction=round(kept / nb, 4) if nb else None,
                modal_share=round(cnt.most_common(1)[0][1] / n, 4) if n else None,
                per_channel_agreement=prev.get(f"{m}@int{bits}", {}).get("escape_agreement_vs_fp32"),
                n_linear_quantized=c["n_linear_quantized"], secs=c["secs"])
    a = "EleutherAI/pythia-410m"
    g4 = res["cells"].get(f"{a}@int4g{G}", {}).get("escape_agreement_vs_fp32")
    g8 = res["cells"].get(f"{a}@int8g{G}", {}).get("escape_agreement_vs_fp32")
    p8 = prev.get(f"{a}@int8", {}).get("escape_agreement_vs_fp32")
    res["KG1_grouping_does_not_rescue_four_bit"] = bool(g4 is not None and g4 <= DECISIVE_FP32)
    res["KG2_grouping_rescues_it"] = bool(g4 is not None and g4 > DECISIVE_FP32)
    res["KG3_eight_bit_regressed"] = bool(g8 is not None and p8 is not None and g8 < p8)

    p = [f"GROUPED QUANTIZATION at g={G}, registered in {PREREG} (sha256 "
         f"{res['_prereg_sha256'][:12]}...). Same estimator, same cells, same probe set as F187; the "
         f"ONLY change is granularity -- one scale per (output row, group of {G} input channels) "
         f"instead of one per row. That is the granularity GPTQ, AWQ and bitsandbytes quantize at. "]
    for key, c in sorted(res["cells"].items(), key=lambda kv: (kv[1]["bits"], kv[1]["model"])):
        p.append(f"{key.split('/')[-1]}: {c['escape_agreement_vs_fp32']} against per-channel "
                 f"{c['per_channel_agreement']}; feature-A keeps {c['feature_A_kept']} of "
                 f"{c['feature_A_base_set_size']} ({c['feature_A_kept_fraction']}). ")
    if res["KG3_eight_bit_regressed"]:
        p.append(f"KG3 FIRES -- grouped 8-bit ({g8}) is BELOW per-channel 8-bit ({p8}). Grouping is "
                 f"strictly finer, so this is an implementation error, not a finding, and nothing "
                 f"below should be read. ")
    if res["KG1_grouping_does_not_rescue_four_bit"]:
        p.append(f"KG1 FIRES: grouped 4-bit reaches {g4}, still at or below the {DECISIVE_FP32} the "
                 f"corpus manipulation produces. GRANULARITY IS NOT WHAT KILLED 4-BIT. The paper's "
                 f"full-precision scoping stands unchanged, and the remaining hope for 4-bit rests "
                 f"entirely on calibration, which this run does not implement. ")
    elif res["KG2_grouping_rescues_it"]:
        p.append(f"KG2 FIRES: grouped 4-bit reaches {g4}, above {DECISIVE_FP32}. F187's 4-bit "
                 f"negative was an artefact of the coarsest possible granularity and must be "
                 f"restated -- at DEPLOYMENT granularity 4-bit survives. This still says nothing "
                 f"about calibrated quantizers. ")
    p.append("WHAT REMAINS OWED, unchanged and narrowed rather than closed: grouping is not "
             "calibration. GPTQ compensates quantization error with a Hessian-based update and AWQ "
             "with activation-aware scaling, and neither is implemented here. The owed item survives "
             "as calibration alone; activation quantization and real serving stacks are untouched. "
             "REFUSALS: no p-value on five cells; no adjustment of the group size, bit widths or "
             "comparison point; feature A reported as a kept fraction and never as a Hamming count.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
