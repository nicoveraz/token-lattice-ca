"""Smoke test: drive the token-lattice CA with bert-tiny. Verify (1) it runs on
MPS, (2) low-T produces English-like local structure while high-T is soup,
(3) the CRN null coupling is exactly zero (harness intact on the torch path)."""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import mlm_ca
from mlm_ca import MLMRule, run

t0 = time.time()
rule = MLMRule("prajjwal1/bert-tiny")
print(f"loaded {rule.name} on {rule.device}/{rule.dtype} in {time.time()-t0:.1f}s "
      f"| V={rule.V} forbidden={len(rule.forbidden)} init_pool={len(rule.init_pool)}")

for T in [0.7, 2.0]:
    t1 = time.time()
    out = run(rule, B=6, N=32, r=2, T=T, sweeps=25, scheme="cls_sep", seed=1)
    act = float(out["activity"][-5:].mean())
    print(f"\n--- T={T}  ({time.time()-t1:.1f}s, activity_final={act:.3f}) ---")
    for b in range(2):
        print("  ", repr(rule.tok.decode(out["final"][b].tolist())))

# null CRN coupling: identical rule/init/order/uniforms -> exactly zero divergence
B, N, sweeps = 4, 32, 8
rng = np.random.default_rng(7)
init = rule.random_lattice(rng, B, N)
u = np.random.default_rng(9).random(sweeps * N * B)
a = run(rule, B=B, N=N, r=2, T=0.8, sweeps=sweeps, init_state=init, seed=5, u_stream=u)
b = run(rule, B=B, N=N, r=2, T=0.8, sweeps=sweeps, init_state=init, seed=5, u_stream=u)
d = (a["snaps"] != b["snaps"]).mean(axis=(1, 2))
print(f"\nNULL CRN divergence max = {d.max():.6f}  (must be 0)")
assert d.max() == 0.0
print("SMOKE OK")
