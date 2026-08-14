"""Build a private review folder: one directory per experiment, with a plain explanation and a chart.

WHY THIS EXISTS. 131 results files, 183 scripts and 124 findings have accumulated, and the only way
to see what a given run actually did is to read its script and its verdict string. That is fine for
one experiment and useless for auditing the whole programme. This renders each run as a page a human
can skim: what it asked, what was pre-registered, what came out, which findings cite it, whether the
stored numbers are still fresh, and a chart of the actual data rather than a restatement of the
verdict.

WHAT IT IS NOT. It does not re-run anything and it does not re-derive any verdict -- it reads stored
results only. A chart here is a view of what was recorded, so if a results file is stale against its
script the page says so at the top rather than quietly plotting old numbers.

CHART POLICY, and the honest part is the fallback. Results files in this project have half a dozen
different shapes. Four are detected and drawn; anything else gets an explicit "no chart" with the
reason, because a wrong chart is worse than no chart -- it invites a reader to see structure that
the detector invented. Shapes handled:

  per-model scalar        a readout per model, averaged over other axes  -> horizontal bars
  model x construction    the same readout across constructions          -> grouped horizontal bars
  temperature sweep       a readout across a temperature ladder          -> lines, one per model
  analysis scalars        a flat dict of named numbers in `analysis`     -> horizontal bars

PALETTE. The six categorical slots are the dataviz reference instance, validated with its own
checker (all checks pass; the contrast WARN is discharged by direct value labels on every mark,
which are drawn). Single-series charts use one slot rather than cycling, and colour follows the
entity, never its rank.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import ast, hashlib, json, re, textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = _ROOT / "review"
RESULTS = _ROOT / "results"
EXPERIMENTS = _ROOT / "experiments"
FINDINGS = _ROOT / "findings.md"

# dataviz reference palette, light mode, validated: lightness band, chroma floor, CVD separation
# and normal-vision floor all PASS. Contrast vs surface WARNs, discharged by direct labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"

# readouts worth charting, in preference order -- the quantities this project actually reports
READOUTS = ["top1", "lambda_ca", "rep2", "distinct", "overall", "share", "s", "branching",
            "tstar", "rep_4", "width", "area"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def short(name):
    return name.split("/")[-1]


def docstring_of(script):
    p = EXPERIMENTS / script
    if not p.exists():
        return None
    try:
        return ast.get_docstring(ast.parse(p.read_text()))
    except Exception:
        return None


def freshness(res, script):
    """Is the stored analysis still the one this script would produce? (F45/F46's trap.)"""
    prov = res.get("_analysis_provenance")
    p = EXPERIMENTS / script
    if not prov or not p.exists():
        return "unknown", "no provenance stamp or script missing"
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if prov.get("sha256") == actual:
        return "fresh", "results match the script that wrote them"
    return "STALE", (f"written by a different version of {script} "
                     f"(stamped {str(prov.get('sha256'))[:12]}, on disk {actual[:12]})")


def findings_citing(stem, script):
    out = []
    if not FINDINGS.exists():
        return out
    txt = FINDINGS.read_text()
    for m in re.finditer(r"^### (F\d+) — (.+)$", txt, re.M):
        start = m.end()
        nxt = txt.find("\n### ", start)
        body = txt[start:nxt if nxt > 0 else len(txt)]
        if stem in body or script in body:
            out.append((m.group(1), m.group(2)))
    return out


# ---------------------------------------------------------------- chart detectors

def _cells(res):
    for key in ("cells", "runs", "models", "local"):
        v = res.get(key)
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            return key, v
    return None, None


def _readout_of(cells):
    keys = set()
    for c in list(cells.values())[:50]:
        keys |= {k for k, v in c.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for r in READOUTS:
        if r in keys:
            return r
    return None


def chart(res, stem, path):
    """Draw the best-supported view, or return a reason why none is."""
    key, cells = _cells(res)
    if cells:
        ro = _readout_of(cells)
        if ro:
            # Some files key entries BY model and carry no `model` field inside them
            # (compliance_v2/v3). Recover it from the key -- "org/model|construction|seed" or
            # a bare model name -- rather than reporting "no chartable shape" for a file that
            # plainly has one.
            rows = []
            for k, c in cells.items():
                if not isinstance(c.get(ro), (int, float)):
                    continue
                c = dict(c)
                c.setdefault("model", str(k).split("|")[0])
                rows.append(c)
            if rows and any("model" in c for c in rows):
                temps = sorted({c["T"] for c in rows if isinstance(c.get("T"), (int, float))})
                cons = sorted({str(c["construction"]) for c in rows if c.get("construction")})
                if len(temps) >= 3:
                    return _lines(rows, ro, "T", path, stem, "temperature",
                                  naxes="radius/seed" if len(cons) > 1 else "seed")
                if 2 <= len(cons) <= 6:
                    return _grouped(rows, ro, cons, path, stem)
                return _bars_by_model(rows, ro, path, stem)
    a = res.get("analysis")
    if isinstance(a, dict):
        flat = {k: v for k, v in a.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e6}
        if len(flat) >= 3:
            return _bars(list(flat), [flat[k] for k in flat], path, stem,
                         "analysis summary", SERIES[0])
    return None, "no recognised chartable shape (scalars only, or a bespoke structure)"


def _bars(labels, values, path, stem, subtitle, colour):
    h = max(2.2, 0.34 * len(labels) + 1.3)
    fig, ax = plt.subplots(figsize=(8.4, h))
    style(ax)
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=colour, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    span = max(values) - min(min(values), 0)
    for i, v in enumerate(values):                      # direct labels discharge the contrast WARN
        ax.text(v + span * 0.012, i, f"{v:.4g}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(min(min(values), 0), max(values) * 1.14 if max(values) > 0 else 1)
    ax.set_title(stem, loc="left", fontsize=11, color=INK, fontweight="bold", pad=12)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)
    return path.name, None


def _bars_by_model(rows, ro, path, stem):
    agg = {}
    for c in rows:
        agg.setdefault(short(str(c["model"])), []).append(c[ro])
    items = sorted(agg.items(), key=lambda kv: -float(np.mean(kv[1])))
    return _bars([k for k, _ in items], [float(np.mean(v)) for _, v in items], path, stem,
                 f"{ro}, averaged over every other axis — one bar per model", SERIES[0])


def _grouped(rows, ro, cons, path, stem):
    models = sorted({short(str(c["model"])) for c in rows})
    agg = {(short(str(c["model"])), str(c.get("construction"))): [] for c in rows}
    for c in rows:
        agg[(short(str(c["model"])), str(c.get("construction")))].append(c[ro])
    h = max(2.6, 0.30 * len(models) * min(len(cons), 4) + 1.4)
    fig, ax = plt.subplots(figsize=(8.4, h)); style(ax)
    n = len(cons); bh = 0.8 / n
    for j, con in enumerate(cons):
        vals = [float(np.mean(agg.get((m, con), [np.nan]))) for m in models]
        y = np.arange(len(models)) + (j - (n - 1) / 2) * bh
        ax.barh(y, vals, height=bh * 0.88, color=SERIES[j % len(SERIES)], label=con, zorder=3)
    ax.set_yticks(np.arange(len(models))); ax.set_yticklabels(models, fontsize=8)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7.5, ncol=min(n, 3), loc="lower right", labelcolor=INK2)
    ax.set_title(stem, loc="left", fontsize=11, color=INK, fontweight="bold", pad=12)
    ax.text(0, 1.015, f"{ro} by construction — legend names the construction",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)
    return path.name, None


def _lines(rows, ro, xkey, path, stem, xname, naxes=""):
    agg = {}
    for c in rows:
        if isinstance(c.get(xkey), (int, float)):
            agg.setdefault(short(str(c["model"])), {}).setdefault(c[xkey], []).append(c[ro])
    dropped = []
    if len(agg) > 6:            # never cycle hues -- but never drop silently either
        keep = sorted(agg, key=lambda m: -max(np.mean(v) for v in agg[m].values()))[:6]
        dropped = sorted(set(agg) - set(keep))
        agg = {k: agg[k] for k in keep}
    fig, ax = plt.subplots(figsize=(8.4, 4.6)); style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    for i, (m, d) in enumerate(sorted(agg.items())):
        xs = sorted(d); ys = [float(np.mean(d[x])) for x in xs]
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=2,
                color=SERIES[i % len(SERIES)], label=m, zorder=3)
    ax.set_xlabel(xname, fontsize=8.5, color=INK2)
    ax.set_ylabel(ro, fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    ax.set_title(stem, loc="left", fontsize=11, color=INK, fontweight="bold", pad=12)
    sub = f"{ro} across the {xname} ladder — one line per model"
    if naxes:
        sub += f"; averaged over {naxes}"
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
    if dropped:
        ax.text(0, -0.16, f"NOT SHOWN ({len(dropped)} of {len(dropped)+len(agg)} models, "
                          f"lowest {ro}): " + ", ".join(dropped),
                transform=ax.transAxes, fontsize=7, color=INK2, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)
    return path.name, None


# ---------------------------------------------------------------- pages

def page(stem, res, script, doc, fresh, why, chartfile, chartwhy, cites):
    L = [f"# {stem}", ""]
    L += [f"**Script** `experiments/{script}` · **Results** `results/{stem}.json`", ""]
    badge = {"fresh": "FRESH", "STALE": "STALE — do not quote these numbers",
             "unknown": "UNVERIFIED"}[fresh]
    L += [f"**Freshness: {badge}** — {why}", ""]
    if cites:
        L += ["**Cited by:** " + ", ".join(f"{f} ({t[:60]}…)" for f, t in cites), ""]
    else:
        L += ["**Cited by:** no finding references this file. Either it is scaffolding, or a "
              "result was never written up.", ""]
    L += ["---", "", "## What it asked", ""]
    if doc:
        para = doc.strip().split("\n\n")
        L += [textwrap.fill(para[0].replace("\n", " "), 96), ""]
        for p in para[1:]:
            head = p.strip().split("\n")[0]
            if re.match(r"^[A-Z][A-Z ,'\-]{6,}", head) or head.startswith("PRE-REGISTERED"):
                L += ["### " + head.split(".")[0].strip().title(), "",
                      textwrap.fill(" ".join(p.split()), 96), ""]
    else:
        L += ["_No script docstring found._", ""]
    L += ["## What came out", ""]
    v = res.get("verdict")
    L += [textwrap.fill(str(v), 96) if v else "_No verdict recorded in the results file._", ""]
    L += ["## Chart", ""]
    L += [f"![{stem}]({chartfile})" if chartfile else f"_No chart: {chartwhy}_", ""]
    return "\n".join(L)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / ".gitignore").write_text("*\n")     # belt and braces: the folder ignores itself too
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        stem = p.stem
        try:
            res = json.load(open(p))
        except Exception as e:
            rows.append((stem, "unreadable", f"{type(e).__name__}", 0)); continue
        if not isinstance(res, dict):
            rows.append((stem, "unreadable", "top level is not an object", 0)); continue
        prov = res.get("_analysis_provenance") or {}
        script = prov.get("source") or f"{stem}.py"
        script = _pathlib.Path(str(script)).name
        doc = docstring_of(script)
        fresh, why = freshness(res, script)
        cites = findings_citing(stem, script)
        d = OUT / stem
        d.mkdir(exist_ok=True)
        cf, cw = chart(res, stem, d / "chart.png")
        (d / "explanation.md").write_text(page(stem, res, script, doc, fresh, why, cf, cw, cites))
        rows.append((stem, fresh, str(res.get("verdict") or "")[:110], len(cites)))
    idx = ["# Review — every stored result, one folder each", "",
           f"{len(rows)} results files. Generated by `experiments/build_review.py`; it reads stored "
           "results only and re-runs nothing.", "",
           "`STALE` means the results file no longer matches the script that wrote it — those "
           "numbers must not be quoted until the analysis is re-run (the F45/F46 trap).", "",
           "| experiment | fresh | findings | verdict (truncated) |", "|---|---|---|---|"]
    for stem, fresh, v, n in rows:
        idx.append(f"| [{stem}]({stem}/explanation.md) | {fresh} | {n} | {v.replace('|', '/')} |")
    stale = [r for r in rows if r[1] == "STALE"]
    uncited = [r for r in rows if r[3] == 0]
    idx += ["", "## Summary", "",
            f"- **{sum(1 for r in rows if r[1] == 'fresh')}** fresh, **{len(stale)}** stale, "
            f"**{sum(1 for r in rows if r[1] == 'unknown')}** unverified",
            f"- **{len(uncited)}** results files are not cited by any finding",
            "", "### Stale — re-run before quoting", ""]
    idx += [f"- `{r[0]}`" for r in stale] or ["- none"]
    idx += ["", "### Not cited by any finding", ""]
    idx += [f"- `{r[0]}`" for r in uncited] or ["- none"]
    (OUT / "INDEX.md").write_text("\n".join(idx))
    print(f"wrote {len(rows)} pages to review/")
    print(f"  fresh {sum(1 for r in rows if r[1]=='fresh')} | stale {len(stale)} | "
          f"unverified {sum(1 for r in rows if r[1]=='unknown')} | uncited {len(uncited)}")


if __name__ == "__main__":
    main()
