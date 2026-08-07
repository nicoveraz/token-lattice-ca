"""The manufactured-transition figure: the collapse, the intervention, and two controls.

WHAT THIS FIGURE HAS TO CARRY. Section 4 of the arXiv paper claims the low-temperature transition
is a property of the PROBE, and the claim rests on four facts that are usually four figures. They
fit in one because `attractor_construction.json` measured them on a common grid:

  * the collapse itself      -- pythia-410m under the autoregressive construction, top-1 share
                                rising sharply as T falls
  * an intervention that
    kills it                 -- the same model with one BOS token prepended: the map's DOMAIN
                                changes, and the collapse largely goes with it
  * a MODEL control          -- gpt2-medium, whose argmax map has no attracting fixed point, run
                                through the identical construction
  * a CONSTRUCTION control   -- two masked-LM models, whose native task is this update, showing
                                no collapse at any temperature

The second panel (r=4) is the boundary claim: the degeneracy occupies small radius only, so at
r=4 every arm should look alike. A reader who doubts that the transition belongs to the probe has
to explain why the construction control is flat and why one prepended token moves it so far.

MONOCHROME, BY MARKER AND DASH. See figstyle: a reviewer with a greyscale printer is a reviewer
who cannot read a hue-encoded series, and five arms is exactly where the temptation to reach for
colour is strongest.

Usage:
    .venv/bin/python experiments/fig_manufactured.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
_sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT / "src")]

import json

import numpy as np
from figstyle import use_classic_r, finish, RULE

SRC = _ROOT / "results" / "attractor_construction.json"
OUT = _ROOT / "fig" / "manufactured.png"

# (arm, model) -> (display label, marker, fill, dash). Order fixes the legend and the z-order.
ARMS = [
    (("ar-none", "EleutherAI/pythia-410m"), ("pythia-410m (AR)",       "o", "white", "-")),
    (("ar-bos",  "EleutherAI/pythia-410m"), ("pythia-410m + BOS",      "s", "black", "--")),
    (("ar-none", "gpt2-medium"),            ("gpt2-medium (AR)",       "^", "white", ":")),
    (("mlm",     "bert-base-uncased"),      ("bert-base (MLM)",        "D", "0.55",  "-.")),
    (("mlm",     "prajjwal1/bert-medium"),  ("bert-medium (MLM)",      "v", "0.80",  "-.")),
]


def series(runs, arm, model, r):
    """(temperatures, top-1 shares) for one arm at one radius, sorted by temperature."""
    pts = [(v["T"], v["top1_share"]) for v in runs.values()
           if v["arm"] == arm and v["model"] == model and v["r"] == r
           and v.get("top1_share") is not None]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main():
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    runs = json.load(open(SRC))["runs"]
    radii = sorted({v["r"] for v in runs.values()})

    use_classic_r()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(radii), figsize=(5.5, 2.3), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, radii):
        for (arm, model), (label, mk, mfc, ls) in ARMS:
            x, y = series(runs, arm, model, r)
            if len(x) == 0:
                continue
            ax.plot(x, y, marker=mk, mfc=mfc, mec="black", ls=ls, color="black",
                    ms=3.4, lw=0.9, label=label)
        # Explicit ticks at the MEASURED temperatures. A log axis auto-labels its minor ticks,
        # which at three data points produces an overlapping smear that hides the one number a
        # reader needs (T = 0.436, where the transition sits).
        ax.set_xscale("log")
        temps = sorted({v["T"] for v in runs.values()})
        ax.set_xticks(temps)
        ax.set_xticklabels([f"{t:g}" for t in temps])
        ax.minorticks_off()
        ax.set_xlabel("temperature $T$")
        ax.set_title(f"$r = {r}$")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=RULE, lw=0.6, ls=(0, (1, 3)))

    axes[0].set_ylabel("top-1 token share")
    axes[0].legend(frameon=False, loc="upper right", handlelength=2.6)
    finish(fig, OUT)

    # The numbers the caption quotes, printed so the caption can be checked against the data
    # rather than against memory. This is the same reason every figure script in this repo emits
    # its own headline values.
    print("\n  arm-by-arm top-1 share at the lowest temperature:")
    tmin = min(v["T"] for v in runs.values())
    for (arm, model), (label, *_ ) in ARMS:
        v = [x for x in runs.values()
             if x["arm"] == arm and x["model"] == model and x["T"] == tmin and x["r"] == radii[0]]
        if v:
            print(f"    r={radii[0]} T={tmin}  {label:<22} top1={v[0]['top1_share']:.4f}  "
                  f"dominant={v[0]['dominant_token']!r}  attractor={v[0]['has_attractor']}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
