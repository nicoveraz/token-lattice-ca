"""The temperature window (item G of the camera-ready restructure): floor, window, ceiling.

Two panels at display size, figstyle conventions. Every plotted value is recomputed from
results/dev_transition_temp.json (and the T=0.7 point from the main grid, exactly as that file's
own summary imports it) -- nothing is typed in.

  (a) lambda_ca at the two ends of training (step256 vs step143000), per temperature, means over
      IGNITED runs (F42: lambda is undefined for a run whose damage never ignited), the same
      filter the number manifest uses for the same quantities.
  (b) ignition fraction at the two ends, per temperature, means over all runs. This is the
      mechanism panel: the floor (damage barely propagates at T=0.3 either end), the swing inside
      the window, and the ceiling (already super-critical pre-training at T=0.9, 1.1).

Writes fig/tempwindow.png.
Usage:  .venv/bin/python experiments/fig_tempwindow.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import use_classic_r, series, finish, RULE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
from lyapunov import run_ignited as ignited  # noqa: E402  -- ONE implementation of F42's predicate

PRE_STEP, POST_STEP = 256, 143000


def main():
    tmp = json.load(open(ROOT / "results" / "dev_transition_temp.json"))
    runs = [v for v in tmp["runs"].values() if isinstance(v, dict) and "lambda_ca" in v]
    temps = sorted({r["T"] for r in runs})

    lam, ign = {}, {}
    for T in temps:
        for step, key in ((PRE_STEP, "pre"), (POST_STEP, "post")):
            sel = [r for r in runs if r["T"] == T and r["step"] == step]
            lam[(T, key)] = float(np.mean([r["lambda_ca"] for r in sel if ignited(r)]))
            ign[(T, key)] = float(np.mean([r["ignition_prob"] for r in sel]))

    # T=0.7 from the main grid, exactly as dev_transition_temp.json's own summary imports it
    s7 = tmp["summary"]["0.7"]
    assert s7["source"] == "dev_transition_phase3"
    lam[(0.7, "pre")], lam[(0.7, "post")] = s7["pre_mean"], s7["plateau_mean"]
    # ignition for 0.7 from the phase3 runs at the same geometry, if the field exists there
    p3 = json.load(open(ROOT / "results" / "dev_transition_phase3.json"))
    p3r = [v for v in p3["runs"].values()
           if isinstance(v, dict) and v.get("N") == 48 and "ignition_prob" in v]
    have_ign7 = False
    for step, key in ((PRE_STEP, "pre"), (POST_STEP, "post")):
        sel = [r["ignition_prob"] for r in p3r if r["step"] == step]
        if sel:
            ign[(0.7, key)] = float(np.mean(sel))
            have_ign7 = True

    lam_T = sorted({t for t, _ in lam})
    ign_T = sorted({t for t, _ in ign})

    use_classic_r()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.7))

    for key, si in (("pre", 0), ("post", 1)):
        ax1.plot(lam_T, [lam[(t, key)] for t in lam_T],
                 **series(si, f"step {PRE_STEP if key == 'pre' else POST_STEP}"))
        ax2.plot(ign_T, [ign[(t, key)] for t in ign_T], **series(si, None))
    ax1.axhline(0, color=RULE, lw=0.6)
    ax1.set_xlabel("temperature $T$")
    ax1.set_ylabel(r"$\lambda_{\mathrm{ca}}$ (ignited runs)")
    ax1.set_title("(a) the window", loc="left")
    ax1.legend(loc="upper left")
    ax2.set_xlabel("temperature $T$")
    ax2.set_ylabel("ignition fraction")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("(b) floor and ceiling", loc="left")
    for ax in (ax1, ax2):
        ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.1])

    # sanity against the paper's literals -- fail loudly rather than draw a wrong figure.
    # The camera-ready fixed the submitted "0.81" to the derivable 0.80 (stored mean 0.8047);
    # asserted here like the others now that paper and file agree.
    assert round(ign[(0.3, "pre")], 2) == 0.20 and round(ign[(0.3, "post")], 2) == 0.21
    assert round(ign[(0.5, "pre")], 2) == 0.23 and round(ign[(0.5, "post")], 2) == 0.80
    assert round(ign[(0.9, "pre")], 2) == 0.65 and round(ign[(1.1, "pre")], 2) == 0.98
    assert round(lam[(0.9, "pre")], 2) == 0.19 and round(lam[(1.1, "pre")], 2) == 0.30
    print("ignition T=0.7 plotted:", have_ign7)
    finish(fig, ROOT / "fig" / "tempwindow.png")


if __name__ == "__main__":
    main()
