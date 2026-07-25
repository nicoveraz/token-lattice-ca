"""Phase 1.1 -- generate golden files BEFORE any refactor of the simulation core.

Runs each of the three current backends (`ca`, `mlm_ca`, `ar_ca`) on a small, fully
deterministic configuration -- fixed `init_state` AND fixed `u_stream`, so nothing depends
on RNG draw order inside the loop -- and stores `snaps`, `activity` and `final`.

`tests/test_golden.py` then asserts BIT-IDENTICAL output after the refactor. If a backend
cannot be made bit-identical, that is a STOP-and-report condition, not a reason to relax
the assertion to `allclose`.

Usage:  .venv/bin/python tests/make_golden.py [--force]
"""
import sys, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import os
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

GOLD = ROOT / "tests" / "golden"
# One config for all three backends. Small enough to run in seconds.
CFG_RUN = dict(B=4, N=48, r=2, T=0.7, sweeps=8, seed=71)
TOY_CKPT = str(ROOT / "ckpt" / "final.npz")
MLM_NAME = "prajjwal1/bert-tiny"
AR_NAME = "EleutherAI/pythia-14m"


def _streams(B, N, sweeps, vocab_lo, vocab_hi, seed=12345):
    """Fixed init lattice + fixed uniform stream, shared by every backend."""
    rng = np.random.default_rng(seed)
    init = rng.integers(vocab_lo, vocab_hi, size=(B, N)).astype(np.int64)
    u = np.random.default_rng(seed + 1).random(sweeps * N * B)
    return init, u


def _save(tag, out):
    GOLD.mkdir(parents=True, exist_ok=True)
    path = GOLD / f"{tag}.npz"
    np.savez_compressed(path, snaps=np.asarray(out["snaps"]),
                        activity=np.asarray(out["activity"]), final=np.asarray(out["final"]))
    print(f"  wrote {path}  snaps={np.asarray(out['snaps']).shape} "
          f"activity={np.asarray(out['activity']).shape}")


def gen_ca():
    from model import CFG, load
    import ca
    params = load(TOY_CKPT)
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 2, CFG["vocab"])
    out = ca.run(params, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
                 sweeps=CFG_RUN["sweeps"], mode="async", seed=CFG_RUN["seed"],
                 init_state=init.astype(np.int32), u_stream=u)
    _save("ca", out)


def gen_mlm():
    from mlm_ca import MLMRule, run
    rule = MLMRule(MLM_NAME)
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 0, len(rule.init_pool))
    init = rule.init_pool[init]                      # map into the legal emission pool
    out = run(rule, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
              sweeps=CFG_RUN["sweeps"], mode="async", scheme="cls_sep",
              seed=CFG_RUN["seed"], init_state=init, u_stream=u)
    _save("mlm", out)


def gen_ar():
    from ar_ca import ARRule, run
    rule = ARRule(AR_NAME)
    init, u = _streams(CFG_RUN["B"], CFG_RUN["N"], CFG_RUN["sweeps"], 0, len(rule.init_pool))
    init = rule.init_pool[init]
    out = run(rule, B=CFG_RUN["B"], N=CFG_RUN["N"], r=CFG_RUN["r"], T=CFG_RUN["T"],
              sweeps=CFG_RUN["sweeps"], scheme="none", seed=CFG_RUN["seed"],
              init_state=init, u_stream=u)
    _save("ar", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing golden files")
    a = ap.parse_args()
    if GOLD.exists() and any(GOLD.glob("*.npz")) and not a.force:
        print(f"golden files already exist in {GOLD}; refusing to overwrite without --force.")
        print("(They are the pre-refactor reference. Regenerating them would defeat the test.)")
        return
    print("generating golden files (pre-refactor reference):")
    for name, fn in [("ca", gen_ca), ("mlm", gen_mlm), ("ar", gen_ar)]:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
