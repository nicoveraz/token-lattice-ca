"""PHASE 1 of matched-entropy: measure entropy, freeze the pairing, compute NO dphi.

Registered in experiments/prereg_matched_entropy.json, whose note is the whole reason this is a
separate script and a separate commit:

    "The pairing must be frozen to disk before any dphi is computed; a pairing chosen after seeing
     dphi would guarantee whichever answer was wanted."

So this file measures induced conditional entropy and nothing else. It writes the pairing and stops.
`matched_entropy.py` reads that file and is the only thing that runs a census. If the two were one
script the ordering would rest on the author's discipline rather than on the commit graph.

WHAT IS MEASURED. For each model and each candidate 9-token prefix, the mean entropy of
p(. | prefix, x1, x2) over the SAME 96 two-token starts the census uses -- same RNG, same seeds, so
the two quantities describe the same forward passes. That is the correction F158 had to make after
comparing against an unmeasured mechanism.

DOCUMENT TYPE comes from the dataset's own `pile_set_name` metadata, not from a score we invent. H2
asks whether content matters beyond entropy, so a pair must differ in type; using the corpus's own
label keeps that from being a judgement of ours.

THE PAIRING RULE, fixed here and stated before any entropy is read: among candidate pairs for a model
that (a) differ in pile_set_name and (b) have induced-entropy shifts within the registered 0.10 nats,
take the pair whose shifts are LARGEST in magnitude, ties broken by lowest row index. Largest-shift
is chosen because K3 kills the comparison if neither arm moves phi, so the rule that best avoids
comparing two nothings is the one that picks the prefixes most likely to move it. It refers only to
entropy. It cannot refer to dphi, because no dphi exists when this runs.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, hashlib, itertools, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from argmax_census_hardened import N_STARTS, CENSUS_SEEDS

PREREG = "experiments/prereg_matched_entropy.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "matched_entropy_pairing.json"
SHA = _ROOT / "results" / "matched_entropy_pairing.sha256"

MODELS = ["EleutherAI/pythia-410m-deduped", "HuggingFaceTB/SmolLM-1.7B", "Qwen/Qwen1.5-1.8B",
          "Qwen/Qwen2.5-1.5B-Instruct", "sapienzanlp/Minerva-3B-base-v1.0",
          "tiiuae/Falcon3-1B-Base"]
LENGTH = PR["frozen"]["geometry"]["prefix_len"]
TOL = 0.10                     # nats; the registered matching tolerance
N_ROWS = 40                    # Pile rows taken in INDEX order, before any text is read
BATCH = 96


@torch.no_grad()
def mean_entropy(model, dev, prefix, starts):
    """Mean entropy of p(. | prefix, x1, x2) over the census's own starts. One batched pass."""
    ctx = [list(prefix) + list(s) for s in starts]
    ent = []
    for i in range(0, len(ctx), BATCH):
        x = torch.tensor(ctx[i:i + BATCH], dtype=torch.long, device=dev)
        lg = model(input_ids=x).logits[:, -1].float()
        p = torch.softmax(lg, dim=-1).double()
        ent.append(float((-(p * torch.log(p.clamp_min(1e-300))).sum(-1)).mean()))
    return float(np.mean(ent))


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    rows = [dict(row=i, text=ds[i]["text"],
                 pile_set_name=(ds[i].get("meta") or {}).get("pile_set_name", "unknown"))
            for i in range(N_ROWS)]

    res = dict(_preregistration_file=PREREG,
               _phase="1 of 2 -- entropy and pairing only. NO dphi is computed here, and none exists "
                      "when this runs. matched_entropy.py is the only script that runs a census.",
               _pairing_rule=("among pairs differing in pile_set_name whose induced-entropy shifts "
                              f"are within {TOL} nats, take the pair with the largest |shift|, ties "
                              "by lowest row index. Refers to entropy only."),
               n_rows_scanned=N_ROWS, prefix_len=LENGTH, tolerance_nats=TOL,
               n_starts=N_STARTS, census_seeds=CENSUS_SEEDS, models=MODELS, entropy={}, pairs={},
               unmatched=[])

    for m in MODELS:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
        V = int(getattr(model.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        # the census's own starts, from the census's own RNG -- same forward passes, per the prereg
        starts = []
        for cs in CENSUS_SEEDS:
            rng = np.random.default_rng(cs)
            starts += [[int(x) for x in rng.choice(pool, size=2)] for _ in range(N_STARTS)]

        cand = {}
        for r in rows:
            ids = tok(r["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= LENGTH:
                cand[r["row"]] = dict(ids=[int(x) for x in ids[:LENGTH]],
                                      pile_set_name=r["pile_set_name"])
        raw_e = mean_entropy(model, "cpu", [], starts)
        ent = {}
        for k, c in cand.items():
            ent[k] = round(mean_entropy(model, "cpu", c["ids"], starts) - raw_e, 6)
        del model; gc.collect()
        res["entropy"][m] = dict(raw_mean_entropy=round(raw_e, 6), n_candidates=len(cand),
                                 shift={str(k): v for k, v in ent.items()},
                                 pile_set_name={str(k): c["pile_set_name"] for k, c in cand.items()},
                                 secs=round(time.time() - t0, 1))

        best = None
        for a, b in itertools.combinations(sorted(ent), 2):
            if cand[a]["pile_set_name"] == cand[b]["pile_set_name"]:
                continue
            if abs(ent[a] - ent[b]) > TOL:
                continue
            score = min(abs(ent[a]), abs(ent[b]))
            key = (-score, a, b)
            if best is None or key < best[0]:
                best = (key, a, b)
        if best is None:
            res["unmatched"].append(m)
            print(f"  {m:<34} NO MATCHED PAIR within {TOL} nats", flush=True)
            continue
        _, a, b = best
        res["pairs"][m] = dict(
            A=dict(row=a, ids=cand[a]["ids"], pile_set_name=cand[a]["pile_set_name"],
                   entropy_shift=ent[a]),
            B=dict(row=b, ids=cand[b]["ids"], pile_set_name=cand[b]["pile_set_name"],
                   entropy_shift=ent[b]),
            shift_gap=round(abs(ent[a] - ent[b]), 6))
        print(f"  {m:<34} rows {a}/{b}  shifts {ent[a]:+.3f}/{ent[b]:+.3f} "
              f"(gap {abs(ent[a]-ent[b]):.3f})  types "
              f"{cand[a]['pile_set_name']}/{cand[b]['pile_set_name']}  "
              f"({res['entropy'][m]['secs']:.0f}s)", flush=True)

    payload = json.dumps(res["pairs"], sort_keys=True).encode()
    res["pairing_sha256"] = hashlib.sha256(payload).hexdigest()
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    SHA.write_text(f"{res['pairing_sha256']}  matched_entropy_pairing.json:pairs  "
                   f"(frozen before any dphi was computed)\n")
    print(f"\n  {len(res['pairs'])} pairs frozen, {len(res['unmatched'])} model(s) unmatched")
    print(f"  pairing sha256 {res['pairing_sha256'][:32]}...")
    print("\nwrote", rel(str(OUT)), "and", rel(str(SHA)))


if __name__ == "__main__":
    main()
