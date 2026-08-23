"""The self-continuation SET: for every token t, does argmax p(. | t, t) == t?

Registered in experiments/prereg_selfcont.json (frozen `3af2e81e...` before any model was loaded),
over the probe strings frozen one commit earlier in experiments/probe_strings_selfcont.json.

WHY THE FEATURE CHANGED FROM A SCALAR TO A SET. fingerprint/PROGRAM.md's frozen battery is a handful
of SCALARS, and Gate 0 already reported the problem with them: the profile is effectively
low-dimensional and bands models into strong/weak-attractor groups rather than identifying them
(4/14 leave-one-out). Three findings since say the scalars are reading the wrong half of the object.
F179: six of seven models across two families and a 22x span of scale land on the SAME endpoint
token, so WHERE trajectories go barely varies. F166: that token is a model x prefix INTERACTION, not
a model property. F172: within one corpus three models share an endpoint token while phi spans 0.036
to 0.458 -- the corpus sets the destination, the weights decide whether it self-continues. So the
discriminative half is self-continuation, and its natural form is not a summary statistic but the SET
of tokens that have it: deterministic, and high-dimensional where the scalars are not.

THE ESTIMATOR IS IMPORTED, NOT RESTATED. The per-token quantity is exactly
newline_margin_freeze.margin(model, dev, prefix_ids=[], nl=t), generalised from '\\n' to arbitrary t.
That function is imported and used as the REFERENCE ORACLE: it runs one token at a time, the
production path runs batches of 256, and floating-point reduction order differs between them. The
batched path is therefore checked against the oracle on a fixed sample per model, and the check is
reported rather than assumed -- an estimator must reproduce a known answer before its verdict counts.
That same check doubles as the batch-invariance diagnostic, which matters here because argmax is
brittle at near-ties and a near-tie can flip on reduction order alone.

WHAT IS MEASURED. The model's ENTIRE own vocabulary, not just the probe set. The probe intersection
is an index into it, so the cross-model comparison and the within-family signature come from one
measurement. No sampling anywhere, no census seeds, no random starts: this estimator is
deterministic, and the script asserts bit-for-bit reproducibility on a repeated call before writing.

ONE MODEL IS MEASURED ON THE PROBE SET ALONE. gpt-neo-2.7B pages against a 16GB machine and runs at
651 ms/token where gpt-neo-1.3B runs at 6.6 -- nine hours for its vocabulary. Since every registered
estimand is defined over the intersection, that cell measures the probe tokens only; see PROBE_ONLY
below. The choice was made from a wall-clock benchmark before any bit of that model was read, and it
keeps the model IN the cohort where the registered alternative, K4, would have dropped it.

Usage:  caffeinate -dimsu .venv/bin/python -u experiments/selfcont_set.py
        (resumable, one file per model, keyed by results/selfcont_set_<model>.json)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from newline_margin_freeze import margin as oracle_margin     # imported, never reimplemented

PROBES = _ROOT / "experiments" / "probe_strings_selfcont.json"
PREREG = "experiments/prereg_selfcont.json"
RESULTS = _ROOT / "results"

# (repo id, family, dtype tag). The bf16 cell is the registered precision control: the SAME weights
# read at a second numeric precision, which is the only near-identical control the cache can supply.
COHORT = [
    ("EleutherAI/pythia-70m", "Pythia", "fp32"),
    ("EleutherAI/pythia-160m", "Pythia", "fp32"),
    ("EleutherAI/pythia-410m", "Pythia", "fp32"),
    ("EleutherAI/pythia-410m-deduped", "Pythia", "fp32"),
    ("EleutherAI/pythia-1b", "Pythia", "fp32"),
    ("EleutherAI/gpt-neo-125m", "GPT-Neo", "fp32"),
    ("EleutherAI/gpt-neo-1.3B", "GPT-Neo", "fp32"),
    ("EleutherAI/gpt-neo-2.7B", "GPT-Neo", "fp32"),
    ("RWKV/rwkv-4-169m-pile", "RWKV", "fp32"),
    ("RWKV/rwkv-4-430m-pile", "RWKV", "fp32"),
    ("state-spaces/mamba-130m-hf", "Mamba", "fp32"),
    ("state-spaces/mamba-370m-hf", "Mamba", "fp32"),
    ("EleutherAI/pythia-410m", "Pythia", "bf16"),
]

BATCH = 256
BATCH_BIG = 32          # 2.7B in float32 is a 9.9GB checkpoint on a 16GB machine
BIG = {"EleutherAI/gpt-neo-2.7B"}

# PROBE-ONLY COVERAGE, and the reason is measured rather than asserted. gpt-neo-2.7B runs at 651
# ms/token against gpt-neo-1.3B's 6.6 -- a 100x slowdown for a 2x model, which is paging a 9.9GB
# checkpoint against ~10GB of usable RAM, not compute. Its whole vocabulary would take 9 hours.
# Every estimand registered in prereg_selfcont.json is defined over the INTERSECTION, so measuring
# this model on the probe tokens alone keeps it in every registered comparison at 1/14th the cost
# and forfeits only the unregistered within-family arm, which is stated as a coverage limit rather
# than left to be noticed. Decided from the wall-clock benchmark, before any bit of this model was
# read, and it PRESERVES the cohort where the alternative -- K4 -- would have shrunk it.
PROBE_ONLY = {"EleutherAI/gpt-neo-2.7B"}
SENTINEL = -2147483648   # margins_e4 value for a token this cell did not measure
N_ORACLE = 96           # tokens re-measured one at a time through the imported estimator
N_REPEAT = 1024         # tokens re-measured through the batched path, for the determinism assert
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}


def cell_key(m, dt):
    return m if dt == "fp32" else f"{m}@{dt}"


def out_path(m, dt):
    return RESULTS / f"selfcont_set_{cell_key(m, dt).replace('/', '__').replace('@', '__')}.json"


@torch.no_grad()
def batched(model, ids, dev, batch):
    """margin and argmax for every t in `ids`, from the two-token state (t, t).

    Same quantity as newline_margin_freeze.margin with an empty prefix; batched because this runs
    over whole vocabularies. Float32 is forced on the logits before the reduction so a bf16 cell
    differs from its fp32 twin only in the WEIGHTS, not in how the comparison is done.
    """
    marg = np.empty(len(ids), np.float64)
    amax = np.empty(len(ids), np.int64)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        x = torch.tensor(chunk, dtype=torch.long, device=dev).view(-1, 1).repeat(1, 2)
        lg = model(input_ids=x).logits[:, -1].float()
        rows = torch.arange(len(chunk))
        own = lg[rows, torch.tensor(chunk)]
        amax[i:i + len(chunk)] = lg.argmax(dim=-1).cpu().numpy()
        lg[rows, torch.tensor(chunk)] = -float("inf")
        marg[i:i + len(chunk)] = (own - lg.max(dim=-1).values).cpu().numpy()
    return marg, amax


def resolve_probes(tok, strings):
    """Which frozen strings encode to EXACTLY one token under this tokenizer? -1 if not exactly one."""
    ids = []
    for s in strings:
        try:
            enc = tok(s, add_special_tokens=False)["input_ids"]
        except Exception:
            enc = []
        ids.append(int(enc[0]) if len(enc) == 1 else -1)
    return ids


def measure(m, fam, dt, probe):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev, t0 = "cpu", time.time()
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m).eval().to(dev, DTYPE[dt])
    batch = BATCH_BIG if m in BIG else BATCH

    V_logits = int(model.get_output_embeddings().weight.shape[0])
    V_own = min(V_logits, len(tok))     # ids at or above len(tok) are untrained padding rows and
                                        # would manufacture structure; excluded, and both are recorded
    probe_ids = resolve_probes(tok, probe["strings_list"])
    full = m not in PROBE_ONLY
    measured = (list(range(V_own)) if full
                else sorted({i for i in probe_ids if 0 <= i < V_own}))

    dense_m = np.full(V_own, float(SENTINEL))
    dense_a = np.full(V_own, -1, np.int64)
    mm, aa = batched(model, measured, dev, batch)
    dense_m[measured], dense_a[measured] = mm, aa

    # DETERMINISM, asserted rather than assumed. A fixed slice plus a fixed stride across the
    # MEASURED ids, re-measured through the same path at the same batch size, must be bit-identical.
    step = max(1, len(measured) // N_REPEAT)
    rep_ids = sorted(set(measured[:min(N_REPEAT, len(measured))] + measured[::step]))
    rmarg, ramax = batched(model, rep_ids, dev, batch)
    same = np.array_equal(rmarg, dense_m[rep_ids]) and np.array_equal(ramax, dense_a[rep_ids])
    if not same:
        bad = int(np.sum(rmarg != dense_m[rep_ids]))
        raise AssertionError(
            f"{cell_key(m, dt)} IS NOT DETERMINISTIC: {bad} of {len(rep_ids)} re-measured tokens "
            f"differ bit-for-bit at the same batch size. Every claim in prereg_selfcont.json assumes "
            f"this estimator needs no seeds; if this fires the whole design is wrong, not the run.")

    # THE ORACLE: newline_margin_freeze.margin, one token at a time. Not expected to be bit-identical
    # -- batching changes the reduction order -- so what is required is that the BIT agrees, and the
    # margin gap is reported. This is also the batch-invariance diagnostic.
    o_ids = sorted({measured[int(k)] for k in
                    np.linspace(0, len(measured) - 1, N_ORACLE).astype(int)})
    gaps, flips = [], []
    for t in o_ids:
        om, oa = oracle_margin(model, dev, [], int(t))
        gaps.append(abs(om - dense_m[t]))
        if (om > 0) != (dense_m[t] > 0) or oa != dense_a[t]:
            flips.append(dict(token=int(t), oracle_margin=round(om, 6),
                              batched_margin=round(float(dense_m[t]), 6),
                              oracle_argmax=int(oa), batched_argmax=int(dense_a[t])))

    marg, amax = dense_m, dense_a
    bits = marg > 0                     # the sentinel is negative, so unmeasured is never a hit
    # MARGINS ARE STORED AS SCALED INTEGERS, and the reason is a guard elsewhere in the repo.
    # tests/test_findings_numbers.py builds its pool of "numbers this project can trace to" from
    # every float in every results/*.json. Thirteen cells x 50k float margins would have added ~300k
    # literals to a baseline of 72k, so any two- to four-decimal number written in findings.md would
    # then match by coincidence -- weakening a repo-wide guard for the whole project, to store a
    # per-token array nobody quotes. Integers do not enter that pool. The scale is 1e-4 logits and
    # the rounding NEVER sends a nonzero margin to zero, because the sign of the margin IS the
    # self-continuation bit and must stay exact; `self_continuing_ids` is computed from the
    # unrounded values and remains authoritative.
    m4 = np.where(marg == 0, 0,
                  np.sign(marg) * np.maximum(1.0, np.round(np.abs(marg) * 1e4))).astype(np.int64)
    m4[marg == float(SENTINEL)] = SENTINEL
    del model
    gc.collect()

    return dict(
        model=m, family=fam, dtype=dt, cell=cell_key(m, dt),
        _preregistration_file=PREREG,
        _prereg_sha256=open(_ROOT / "experiments" / "prereg_selfcont.sha256").read().split()[0],
        _probe_strings_file="experiments/probe_strings_selfcont.json",
        _probe_strings_sha256=probe["strings_sha256"],
        _estimator="newline_margin_freeze.margin(prefix=[], nl=t), imported; batched for whole "
                   "vocabularies and checked against the imported one-at-a-time path below",
        _deterministic=True,
        _determinism_check=dict(n_tokens=len(rep_ids), bit_for_bit_identical=True,
                                note="re-measured through the same path at the same batch size"),
        _oracle_check=dict(n_tokens=len(o_ids), max_abs_margin_gap=round(float(max(gaps)), 8),
                           median_abs_margin_gap=round(float(np.median(gaps)), 8),
                           n_bit_or_argmax_disagreements=len(flips), disagreements=flips[:20],
                           batch=batch,
                           note="the batched path vs the imported one-at-a-time estimator. Bits must "
                                "agree; margins need not be bit-identical because batching changes "
                                "the floating-point reduction order. Doubles as the batch-invariance "
                                "diagnostic: argmax is brittle at near-ties."),
        vocab_logits=V_logits, vocab_tokenizer=int(len(tok)), vocab_measured=V_own,
        coverage=("full_vocabulary" if full else "probe_only"), n_measured=len(measured),
        _unmeasured_sentinel=SENTINEL,
        _coverage_note=("every token of this model's own vocabulary was measured" if full else
                        "PROBE-ONLY: only the tokens the frozen probe strings resolve to were "
                        "measured, because this checkpoint pages on a 16GB machine and its whole "
                        "vocabulary would take 9 hours. Every REGISTERED estimand is defined over "
                        "the intersection and is unaffected; the within-family arm over tokens "
                        "outside the probe set is not available for this cell and is reported as a "
                        "coverage limit."),
        n_self_continuing=int(bits.sum()),
        # denominator is what was MEASURED, not the vocabulary. For a probe-only cell the two
        # differ by 14x, and dividing by the vocabulary would report a fraction of a population
        # this cell never looked at.
        self_continuing_fraction=round(float(bits.sum()) / len(measured), 6),
        self_continuing_ids=[int(i) for i in np.flatnonzero(bits)],
        margins_e4=[int(x) for x in m4],
        _margin_scale="1e-4 logits, stored as integers; see the comment in measure(). A nonzero "
                      "margin never rounds to zero, so sign(margins_e4) == the self-continuation "
                      "bit exactly, and self_continuing_ids is computed from the unrounded values.",
        argmax_ids=[int(x) for x in amax],
        probe_token_ids=probe_ids,
        n_probe_single_token=int(sum(1 for i in probe_ids if i >= 0)),
        secs=round(time.time() - t0, 1))


def main():
    probe = json.load(open(PROBES))
    probe["strings_list"] = [e["s"] for e in probe["strings"]]
    RESULTS.mkdir(exist_ok=True)
    failed = []
    for m, fam, dt in COHORT:
        p = out_path(m, dt)
        if p.exists():
            print(f"  {cell_key(m, dt):<34} cached", flush=True)
            continue
        try:
            res = measure(m, fam, dt, probe)
        except AssertionError:
            raise                       # determinism is not a load failure; it kills the design
        except Exception as e:
            failed.append(dict(cell=cell_key(m, dt), error=type(e).__name__, detail=str(e)[:200]))
            print(f"  {cell_key(m, dt):<34} LOAD FAILED {type(e).__name__}", flush=True)
            continue
        res["_analysis_provenance"] = stamp(__file__)
        json.dump(res, open(p, "w"), indent=1)
        print(f"  {res['cell']:<34} V={res['vocab_measured']:<6} "
              f"{'full' if res['coverage'] == 'full_vocabulary' else 'PROBE'} "
              f"n={res['n_measured']:<6} "
              f"self-cont {res['n_self_continuing']:>5} ({res['self_continuing_fraction']:.4f})  "
              f"probe 1-token {res['n_probe_single_token']:>4}/{len(probe['strings_list'])}  "
              f"oracle gap {res['_oracle_check']['max_abs_margin_gap']:.2e} "
              f"flips {res['_oracle_check']['n_bit_or_argmax_disagreements']}  "
              f"({res['secs']:.0f}s)", flush=True)

    if failed:
        json.dump(dict(failed=failed, _analysis_provenance=stamp(__file__)),
                  open(RESULTS / "selfcont_set_failures.json", "w"), indent=1)
        print(f"\n  K4: {len(failed)} cell(s) unusable and NAMED, not dropped: "
              f"{[f['cell'] for f in failed]}. Every gate that reads cohort shape must be "
              f"re-evaluated on the survivors.", flush=True)
    print("\nwrote", rel(str(RESULTS)) + f"/selfcont_set_*.json")


if __name__ == "__main__":
    main()
