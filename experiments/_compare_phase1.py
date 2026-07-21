"""Compare Phase-1 reproduced results (working tree) against the committed pilot
baseline (git HEAD). Confirms findings F1-F9 qualitatively. Run from repo root:
    .venv/bin/python experiments/_compare_phase1.py
"""
import json, subprocess, io, sys
import numpy as np


def head_bytes(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"],
                          capture_output=True).stdout


def head_json(path):
    return json.loads(head_bytes(path).decode())


def head_jsonl(path):
    return [json.loads(l) for l in head_bytes(path).decode().splitlines() if l.strip()]


def head_npz(path):
    return np.load(io.BytesIO(head_bytes(path)))


def cur_json(path):
    return json.load(open(path))


print("=" * 72)
print("PHASE 1 REPRODUCTION vs COMMITTED PILOT BASELINE (git HEAD)")
print("=" * 72)

# ---- F1/F2: sweep order parameter & activity, radius collapse -------------
print("\n[F1/F2] sweep: order parameter (bigram_final) & activity vs (r,T)")
cur = {(x["mode"], x["r"], x["T"]): x for x in
       [json.loads(l) for l in open("results/summary.jsonl")]}
base = {(x["mode"], x["r"], x["T"]): x for x in head_jsonl("results/summary.jsonl")}
RS, TS = [1, 2, 4, 8, 16], [0.3, 0.7, 1.0, 1.5, 2.5]
print("  order parameter (corpus-bigram fraction), async:")
print("     T   " + "".join(f"  r={r:<2d}" for r in RS) + "   | radius spread @each T")
maxdiff = 0.0
for T in TS:
    row = [cur[("async", r, T)]["bigram_final"] for r in RS]
    spread = max(row) - min(row)
    print(f"   {T:>4} " + "".join(f" {v:5.2f}" for v in row) + f"   | {spread:.3f}")
    for r in RS:
        maxdiff = max(maxdiff, abs(cur[("async", r, T)]["bigram_final"]
                                   - base[("async", r, T)]["bigram_final"]))
print(f"  order param drop T0.3 -> T2.5 (r=2): "
      f"{cur[('async',2,0.3)]['bigram_final']:.2f} -> {cur[('async',2,2.5)]['bigram_final']:.2f}"
      f"   (pilot F1: 1.00 -> 0.14)")
print(f"  max |repro - pilot| over all (r,T) bigram_final = {maxdiff:.3f}")

# ---- F5: sync manufactures period-2 -------------------------------------
print("\n[F5] period-2 fraction async vs sync (analysis.json)")
a = cur_json("results/analysis.json")["period2_fraction"]
try:
    ab = head_json("results/analysis.json")["period2_fraction"]
except Exception:
    ab = {}
for T in [0.3, 0.7, 1.0]:
    print(f"   T={T}: async={a.get(f'async_T{T}')}  sync={a.get(f'sync_T{T}')}"
          f"   (pilot: sync~0.84 vs async~0.05 at T=0.3)")

# ---- F2: damage cones ----------------------------------------------------
print("\n[F2/F8] damage cones: total damage & width vs (T,r) (analysis.json)")
dmg = cur_json("results/analysis.json")["damage"]
for k in sorted(dmg):
    print(f"   {k}: width={dmg[k]['final_width_sites']:>2} sites  "
          f"total_damage={dmg[k]['total_damage']:.3f}")

# ---- F3: census recovery -------------------------------------------------
print("\n[F3] census: top-50 trigram overlap & Spearman rho vs corpus")
cen = cur_json("results/census.json")["census"]
cenb = head_json("results/census.json")["census"]
for T in ["0.3", "0.7", "1.0"]:
    c, b = cen[T], cenb[T]
    print(f"   T={T}: overlap50={c['overlap50']:.3f} (pilot {b['overlap50']:.3f})  "
          f"rho={c['spearman']:.3f} (pilot {b['spearman']:.3f})  "
          f"baseline={c['baseline_overlap50']:.3f}")

# ---- F6: melting ---------------------------------------------------------
print("\n[F6] melting: corpus-text identity retention (T=0.3, final sweep)")
mel = cur_json("results/census.json")["melts"]
for T in ["0.3", "1.0", "2.5"]:
    k = mel[T]
    print(f"   T={T}: retention {k[0]:.2f} -> {k[-1]:.2f} over {len(k)-1} sweeps")

# ---- F9: differential ----------------------------------------------------
print("\n[F9] differential: null / apparatus / model arms")
d = cur_json("results/differential.json")
for T in [0.3, 0.7]:
    print(f"   T={T}: NULL max divergence = {d[f'null_T{T}']:.6f}  (MUST be 0)")
    print(f"          apparatus:cdfperm  d[5,20,end] = {d[f'apparatus:cdfperm_T{T}']}")
    for tag in ["step0", "step1000", "step3000", "step5000"]:
        print(f"          model:{tag:8s}    d[5,20,end] = {d[f'model:{tag}_T{T}']}")

# ---- F7: crystallization (reproduced-reduced) ----------------------------
print("\n[F7] crystallization: order/census/val crystallize by step 1000")
try:
    cr = cur_json("results/crystal.json")
    steps = ["0", "1000", "2000", "6000"]
    print("   step  bigram_T0.3  census_ov50  val_acc  melt_ret")
    for s in steps:
        r = cr[s]
        print(f"   {s:>5}  {r.get('bigram_T0.3', float('nan')):>10.3f}  "
              f"{r.get('census_overlap50', float('nan')):>10.3f}  "
              f"{r.get('val_acc', float('nan')):>6.3f}  "
              f"{r.get('melt_retention', float('nan')):>7.3f}")
except Exception as e:
    print("   (crystal.json not available:", e, ")")

print("\n" + "=" * 72)
print("DONE")
